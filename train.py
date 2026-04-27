VOXEL_SIZE = 1 / 128 # 3D Space defined into 128^3 voxels world
NUM_GS = 4 # Each anchor point will spawn 4 individual Gaussians, Each Gaussian will have 14 features to optimize
ALIGN_RATIO = 5 # Control how far a Gaussian can move from its anchor point
O_BIAS = .1
SCA_BIAS = VOXEL_SIZE
N_training_VIEWS = 8 # Views for training
N_VIEWS = 8 # Views for validation

N_STEPS = 400
EVAL_PER = 20
PROFILE = True
VERSION = 'gsplat'
MODE = 0
SCENE = '31a2c91c43'

from contextlib import nullcontext
import cv2
from gc import collect
import os
import numpy as np
from tqdm import trange
import argparse

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
IS_NPU_BACKEND = VERSION in {'1015', '1030', '1212', '1230'}
if PROFILE and IS_NPU_BACKEND: os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
if IS_NPU_BACKEND: import torch_npu

from sample import depth_to_world, voxel_downsample, tsdf_fusion


if VERSION not in {'gsplat', 'torchsplat'}:
    DEVICE = torch.device('npu')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
DTYPE = torch.float

XYZ_SCALE = torch.tensor(ALIGN_RATIO * VOXEL_SIZE, device=DEVICE, dtype=DTYPE) # Control how far a Gaussian can move from its anchor point. 
# The actual movement is XYZ_SCALE * tanh(feature[..., :3]), which is bounded within [-XYZ_SCALE, XYZ_SCALE]. 
O_BIAS = -(1 / torch.tensor(O_BIAS, device=DEVICE, dtype=DTYPE) - 1).log()
SCA_BIAS = torch.expm1(torch.tensor(SCA_BIAS, device=DEVICE, dtype=DTYPE)).log()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default=VERSION, help='version for training and visualization')
    parser.add_argument('--scene', type=str, default=SCENE, help='scene name for training')
    parser.add_argument('--profile', action='store_true', help='whether to profile the code')
    parser.add_argument('--mode', type=int, default=MODE, help='algorithm mode for gsplat: 0 for opacity-aware, 1 for snugbox, 2 for accutile')
    return parser.parse_args()


def main():

    arg = args_parse()
    MODE = arg.mode
    VERSION = arg.version
    print("Training Version:", VERSION)
    print("Training Mode:", MODE)
    
    if VERSION == '1015':
        from meta_gauss_render.ascend_gauss_render import AscendGaussRender
        from gaussian_render_npu import render_metagauss_perview
    elif VERSION == '1030':
        from rasterizer_1030 import Rasterizer
    elif VERSION == '1212':
        from rasterizer_1212 import Rasterizer
    elif VERSION == '1230':
        from rasterizer_1230 import Rasterizer
    elif VERSION == 'torchsplat':
        from torch_splat.rendering import torch_rasterization
    else:  # VERSION == 'gsplat'
        from gsplat import rasterization

    print('Loading data')
    import datetime
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(f'vis-{VERSION}-{time}', exist_ok=True)

    # 1. Loading Data

    # 32 indexes use for training selected from 64 views
    idx32 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
    image_npz = np.load(f'/data06/z84301856/scannetpp_v2_opencv_pinhole/supervise_view_fix64_viewInfo_normV14/{SCENE}_viewInfo.npz')
    image_nv_npz = np.load(f'/data06/z84301856/scannetpp_v2_opencv_pinhole/novel_view_fix8_viewInfo_normV14/{SCENE}_viewInfo.npz')
    depth_npz = np.load(f'/data06/z84301856/scannetpp_v2_opencv_pinhole/geometry_priors/irregular_pts_align_unipts_normV14/mapanything_pcd_normext_imgonly_64view_v2_edgepreserve_csv/mapanything_imgonly_v2_32view/depth_pred/{SCENE}.npz') # Only take 32 views for training, which are the same views as idx32

    print('Copying data')
    # Training Set: images shape (64, 3, 1168, 1752), intrinsics: [32,3,3], extrinsics: [32,4,4]; Depth Maps of 32 images, shape = (32, 336, 518)
    # Validation Set: 8 views for validation, H= 336, W=518: images_nv, intrinsics_nv: [8,3,3], extrinsics_nv: [8,4,4]

    images = image_npz['ori_img_tensor']
    images = torch.tensor(images, device=DEVICE, dtype=torch.float) # Shape = (64, 3, 1168, 1752)
    images = F.interpolate(images, scale_factor=.5, mode='area') # Shrink to Shape = (64, 3, 584, 876)

    intrinsics = image_npz['normed_intrinsics_norm128_v15_fix32'] # Shape = (32, 3, 3)
    intrinsics = torch.tensor(intrinsics, device=DEVICE, dtype=DTYPE)
    extrinsics = image_npz['extrinsics_Tnorm_norm128_v15_fix32'] # Shape = (32, 4, 4)
    extrinsics = torch.tensor(extrinsics, device=DEVICE, dtype=DTYPE)

    images_nv = image_nv_npz['ori_img_tensor'] # Shape = (8, 3, 1168, 1752)
    images_nv = torch.tensor(images_nv, device=DEVICE, dtype=torch.float) # Shape = (8, 3, 1168, 1752)
    images_nv = F.interpolate(images_nv, scale_factor=.5, mode='area') # Shrink to Shape = (8, 3, 584, 876)

    intrinsics_nv = image_nv_npz['normed_intrinsics_norm128_v15_fix32'] # Shape = (8, 3, 3)
    intrinsics_nv = torch.tensor(intrinsics_nv, device=DEVICE, dtype=DTYPE)
    extrinsics_nv = image_nv_npz['extrinsics_Tnorm_norm128_v15_fix32'] # Shape = (8, 4, 4)
    extrinsics_nv = torch.tensor(extrinsics_nv, device=DEVICE, dtype=DTYPE)


    # Force the computer to delete old copies immediately to free up memory
    collect()
    if IS_NPU_BACKEND:
        torch_npu.npu.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # 2. Sampling anchors

    print('Sampling anchors')
    # scale = image_npz['scale'].item()
    # depth_gt = image_npz['depthmap_1168x1752_norm128_v15_fix32'][:, 0]
    # depth_gt = torch.tensor(depth_gt, device=DEVICE, dtype=DTYPE) * scale
    
    # Depth Maps of 32 images, : 32 views for training, H= 584, W=876
    depth_ma = depth_npz['data']
    depth_ma = torch.tensor(depth_ma, device=DEVICE, dtype=DTYPE) # shape = (32, 584, 876)
    # points = depth_to_world(depth_ma, intrinsics[idx32], extrinsics[idx32])
    # anchors = voxel_downsample(points, VOXEL_SIZE)


    # From depth maps to sampled points near the surface using TSDF fusion.
    # Result: set of anchors samples (x,y,z) representing where objects likely to be:  (M, 3)
    anchors = tsdf_fusion(depth_ma, intrinsics[idx32], extrinsics[idx32], VOXEL_SIZE) # Likely torch.Size([297064, 3])

    # print(torch.cuda.memory_summary())
    # torch.cuda.reset_peak_memory_stats()

    del depth_ma
    collect()
    if IS_NPU_BACKEND:
        torch_npu.npu.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    # 3. Training Gaussians

    # features = (M, NUM_GS, 14)
    # Initialize 14 random features for each Gaussian: (3 for xyz, 1 for opacity, 3 for scale, 4 for rotation quaternion, 3 for RGB color)
    feature = .1 * torch.randn(anchors.shape[0], NUM_GS, 14, device=DEVICE, dtype=DTYPE) # Shape (M, NUM_GS, 14) (M: number of anchors)

    save_dict = {
        "anchors": anchors.cpu(),
        "feature": feature.cpu(),
        "voxel_size": VOXEL_SIZE,
        "num_gs": NUM_GS,
    }   


    save_path = "data/anchors_features.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    torch.save(save_dict, save_path)
    print(f"Saved to {save_path}")
    
    # Set optimizer to update the features of the Gaussians
    optim = torch.optim.Adam([feature.requires_grad_()], .01)

    if VERSION != '1230':
        intrinsics[:, :1] *= images.shape[3]
        intrinsics[:, 1:2] *= images.shape[2]
        intrinsics_nv[:, :1] *= images_nv.shape[3]
        intrinsics_nv[:, 1:2] *= images_nv.shape[2]
    if VERSION == '1015': gauss_render = AscendGaussRender(
        width=images.shape[3],
        height=images.shape[2],
        active_sh_degree=0,
        isect_mode='flashgs',
        cpu_radix_sort=False
    )
    elif VERSION == '1030': rasterizer = Rasterizer()
    elif VERSION == '1212': rasterizer = Rasterizer()
    elif VERSION == '1230': rasterizer = Rasterizer()
    elif VERSION == 'torchsplat': pass
    else: pass  # VERSION == 'gsplat'

    pbar = trange(N_STEPS, desc=f'Num GS = {anchors.shape[0] * NUM_GS}')
    

    writer = SummaryWriter(f'tensorboard-{VERSION}/run-{time}')


    # with torch_npu.profiler.profile(with_modules=True, on_trace_ready=handler) if PROFILE else nullcontext():
    prof = None
    if PROFILE:
        if IS_NPU_BACKEND:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                            torch_npu.profiler.ExportType.Text # Saves the results in human-readable text format.
                            ],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2, # Collect detailed performance data, task-level scheduling data.
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization # Measures how welll the AI Cores are being utilized
                )
            prof = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU,
                            torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=True,
                record_shapes=False,
                profile_memory=True,
                schedule=torch_npu.profiler.schedule(wait=10, warmup=10, active=1, repeat=0, skip_first=0), # Skip 10 steps, warmup for 10 steps, then profile only 1 step.
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'profile-{VERSION}') # Saves the profiling results when ready
            )
        else:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=False,
                profile_memory=True,
                schedule=torch.profiler.schedule(wait=10, warmup=10, active=1, repeat=0, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'profile-{VERSION}')
            )

        prof.start()


    for i in pbar:
        # feature[..., :3]: xyz shape (M, NUM_GS, 3)
        # feature[..., 3:4]: opacity shape (M, NUM_GS, 1)
        # feature[..., 4:7]: scale shape (M, NUM_GS, 3)
        # feature[..., 7:11]: rotation quaternion shape (M, NUM_GS, 4)
        # feature[..., 11:]: RGB color shape (M, NUM_GS, 3)

        xyz = anchors[:, None] + XYZ_SCALE * feature[..., :3].tanh()
        o = (feature[..., 3:4] + O_BIAS).sigmoid()
        sca = F.softplus(feature[..., 4:7] + SCA_BIAS)
        quat = F.normalize(feature[..., 7:11], dim=2)
        rgb = feature[..., 11:]

        # views = (N_training,): take 8 views from 64 views
        views = torch.randint(0, images.shape[0], (N_training_VIEWS,), device=DEVICE)


        if VERSION == '1015':
            gaussian = torch.cat((xyz, o, sca, quat, rgb), 2).flatten(0, 1)
            render = [render_metagauss_perview(
                gauss_render,
                gaussian,
                extrinsics[v:v+1],
                intrinsics[v:v+1],
                resolution=images.shape[2:]
            )['render_image'] for v in views]
            render = torch.stack(render).clamp(0, 1)
        elif VERSION == '1030':
            splats = {
                'means': xyz.reshape(-1, 3),
                'opacities': o.reshape(-1),
                'scales': sca.reshape(-1, 3),
                'quats': quat.reshape(-1, 4)
            }
            colors = rgb.reshape(-1, 1, 3)
            render = [rasterizer.ascend_rasterize_splats(
                camtoworlds=extrinsics[v:v+1],
                Ks=intrinsics[v:v+1],
                width=images.shape[3],
                height=images.shape[2],
                splats=splats,
                colors=colors,
                sh_degree=0
            )[0] for v in views]
            render = torch.cat(render).movedim(3, 1).clamp(0, 1)
        elif VERSION == '1212':
            splats = {
                'means': xyz.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'opacities': o.reshape(-1), # Shape (M*NUM_GS,)
                'scales': sca.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'quats': quat.reshape(-1, 4), # Shape (M*NUM_GS, 4)
                'sh0': rgb.reshape(-1, 1, 3) # Shape (M*NUM_GS, 1, 3), treat RGB as SH0 for color output
            }

            # Render the splats into 2D images from the selected views using the rasterizer.
            render = [rasterizer.ascend_rasterize_splats(
                w2c=extrinsics[v:v+1],
                Ks=intrinsics[v:v+1],
                width=images.shape[3],
                height=images.shape[2],
                tile_size=32,
                splats=splats,
                active_sh_degree=0
            )[0] for v in views]


            render = torch.cat(render).movedim(3, 1).clamp(0, 1) # Shape (8, 3, H, W)
            # print("Render shape (C_VIEWS, 3, H, W):", render.shape)


        elif VERSION == '1230':
            splats = {
                'mean': xyz.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'opacity': o.reshape(-1), # Shape (M*NUM_GS,)
                'scale': sca.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'rotation': quat.reshape(-1, 4), # Shape (M*NUM_GS, 4)
                'color': rgb.reshape(-1, 1, 3) # Shape (M*NUM_GS, 1, 3)
            }
            # Render the splats into 2D images from the selected views using the rasterizer.
            render = [rasterizer.ascend_rasterize_splats(
                    w2c=extrinsics[v:v+1], # (1,4,4)
                    Knorm=intrinsics[v:v+1], # (1,3,3)
                    width=images.shape[3],
                    height=images.shape[2],
                    tile_size=32,
                    splats=splats,
                    active_sh_degree=0
                )[0] for v in views]
            render = torch.cat(render).clamp(0, 1)    
            
            # render = rasterizer.ascend_rasterize_splats(
            #     w2c=extrinsics[views],       # Shape (8, 4, 4)
            #     Knorm=intrinsics[views],     # Shape (8, 3, 3)
            #     width=images.shape[3],
            #     height=images.shape[2],
            #     tile_size=32,
            #     splats=splats,           # The splats are shared for all cameras views
            #     active_sh_degree=0
            # )[0]
            # render = render.clamp(0, 1) # Shape (8, 3, H, W)
            # print("Render shape (C_VIEWS, 3, H, W):", render.shape)


        elif VERSION == 'torchsplat':
            splats = {
                'mean': xyz.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'opacity': o.reshape(-1), # Shape (M*NUM_GS,)
                'scale': sca.reshape(-1, 3), # Shape (M*NUM_GS, 3)
                'rotation': quat.reshape(-1, 4), # Shape (M*NUM_GS, 4)
                'color': rgb.reshape(-1, 1, 3) # Shape (M*NUM_GS, 1, 3)
            }
            # Render the splats into 2D images from the selected views using the rasterizer.
            render = [torch_rasterization(
                    means=xyz.reshape(-1, 3),
                    quats=quat.reshape(-1, 4),
                    scales=sca.reshape(-1, 3),
                    opacities=o.reshape(-1),
                    colors=rgb.reshape(-1, 3),
                    viewmats=extrinsics[v:v+1],
                    Ks=intrinsics[v:v+1],
                    width=images.shape[3],
                    height=images.shape[2],
                    render_mode='RGB+ED'
                    )[0][..., :3].movedim(3, 1).clamp(0, 1) for v in views]
            render = torch.cat(render) # Return the shape of (C_VIEWS, H, W, 3)

            # print("Shape Analysis of Rendered Image from torchsplat Rasterizer")
            # print(render.shape)



        else:  # VERSION == 'gsplat'
            render = rasterization(
                means=xyz.reshape(-1, 3),
                quats=quat.reshape(-1, 4),
                scales=sca.reshape(-1, 3),
                opacities=o.reshape(-1),
                colors=rgb.reshape(-1, 3),
                viewmats=extrinsics[views],
                Ks=intrinsics[views],
                width=images.shape[3],
                height=images.shape[2],
                render_mode='RGB+ED',
                algorithm_mode = MODE
            )[0][..., :3].movedim(3, 1).clamp(0, 1)

        
        # Start training step: compute loss, backpropagate, and update features
        optim.zero_grad()
        # Loss L1 between the rendered image and the ground truth image from the selected views.
        loss = F.l1_loss(render, images[views])
        loss.backward()
        optim.step()

        if PROFILE and prof is not None:
            prof.step()

            if DEVICE.type == 'cuda':
                peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
            else:
                peak_alloc = peak_reserved = 0.0

            # print(f"Step {i} Peak GPU memory allocated: {peak_alloc:.2f} MB")
            # print(f"Step {i} Peak GPU memory reserved: {peak_reserved:.2f} MB")

            # torch.cuda.reset_peak_memory_stats()


        # Change to inference_mode, calculate PSNR (Peak Signal-to-Noise Ratio) between rendered images and ground truth images
        # Metric Calculation: PSNR = -10 * log10(MSE), where MSE is the mean squared error between the rendered image and the ground truth image.
        # Higher PSNR -> lower error

        with torch.inference_mode(): psnr = -10 * F.mse_loss(render, images[views], reduction='none').mean((1, 2, 3)).log10_().mean().item()
        writer.add_scalar('train PSNR', psnr, i, new_style=True)

        # Do validation every EVAL_PER steps (i% EVAL_PER == 0)
        if not i % EVAL_PER:
            with torch.inference_mode():
                if VERSION == '1015':
                    render_nv = [render_metagauss_perview(
                        gauss_render,
                        gaussian,
                        extrinsics_nv[v:v+1],
                        intrinsics_nv[v:v+1],
                        resolution=images_nv.shape[2:]
                    )['render_image'] for v in range(N_VIEWS)]
                    render_nv = torch.stack(render_nv).clamp(0, 1)
                elif VERSION == '1030':
                    render_nv = [rasterizer.ascend_rasterize_splats(
                        camtoworlds=extrinsics_nv[v:v+1],
                        Ks=intrinsics_nv[v:v+1],
                        width=images_nv.shape[3],
                        height=images_nv.shape[2],
                        splats=splats,
                        colors=colors,
                        sh_degree=0
                    )[0] for v in range(N_VIEWS)]
                    render_nv = torch.cat(render_nv).movedim(3, 1).clamp(0, 1)
                elif VERSION == '1212':
                    render_nv = [rasterizer.ascend_rasterize_splats(
                        w2c=extrinsics_nv[v:v+1],
                        Ks=intrinsics_nv[v:v+1],
                        width=images_nv.shape[3],
                        height=images_nv.shape[2],
                        tile_size=32,
                        splats=splats,
                        active_sh_degree=0
                    )[0] for v in range(N_VIEWS)]
                    render_nv = torch.cat(render_nv).movedim(3, 1).clamp(0, 1)
                elif VERSION == '1230':
                    render_nv = [rasterizer.ascend_rasterize_splats(
                        w2c=extrinsics_nv[v:v+1],
                        Knorm=intrinsics_nv[v:v+1],
                        width=images_nv.shape[3],
                        height=images_nv.shape[2],
                        tile_size=32,
                        splats=splats,
                        active_sh_degree=0
                    )[0] for v in range(N_VIEWS)]
                    render_nv = torch.cat(render_nv).clamp(0, 1)
                
                elif VERSION == 'torchsplat':
                    render_nv = [torch_rasterization(
                        means=xyz.reshape(-1, 3),
                        quats=quat.reshape(-1, 4),
                        scales=sca.reshape(-1, 3),
                        opacities=o.reshape(-1),
                        colors=rgb.reshape(-1, 3),
                        viewmats=extrinsics_nv[v:v+1],
                        Ks=intrinsics_nv[v:v+1],
                        width=images_nv.shape[3],
                        height=images_nv.shape[2],
                        render_mode='RGB+ED'
                    )[0][..., :3].movedim(3, 1).clamp(0, 1) for v in range(N_VIEWS)]
                    render_nv = torch.cat(render_nv)

                    
                else:  # VERSION == 'gsplat'
                    render_nv = rasterization(
                        means=xyz.reshape(-1, 3),
                        quats=quat.reshape(-1, 4),
                        scales=sca.reshape(-1, 3),
                        opacities=o.reshape(-1),
                        colors=rgb.reshape(-1, 3),
                        viewmats=extrinsics_nv,
                        Ks=intrinsics_nv,
                        width=images_nv.shape[3],
                        height=images_nv.shape[2],
                        render_mode='RGB+ED',
                        algorithm_mode = MODE
                    )[0][..., :3].movedim(3, 1).clamp(0, 1)

            psnr_nv = -10 * F.mse_loss(render_nv, images_nv, reduction='none').mean((1, 2, 3)).log10_().mean().item()
            writer.add_scalar('val PSNR', psnr_nv, i, new_style=True)

            addiotnal_views = torch.cat([views, torch.randint(0, images.shape[0], (N_VIEWS - N_training_VIEWS,), device=DEVICE)])

            render_broadcasted = render.repeat(N_VIEWS // render.shape[0], 1, 1, 1)
            # render_broadcasted = render.expand(N_VIEWS, -1, -1, -1)

            vis = torch.stack((images[addiotnal_views], render_broadcasted, images_nv, render_nv)).mul_(255).round_().byte()
            vis = vis.permute(0, 3, 1, 4, 2)[..., [2, 1, 0]].flatten(0, 1).flatten(1, 2).cpu().numpy()
            vis = cv2.resize(vis, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'vis-{VERSION}-{time}/{i:04}.png', vis, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        pbar.set_postfix({'train PSNR': psnr, 'val PSNR': psnr_nv})

    if PROFILE and prof is not None: 
        prof.stop() # Flushes the buffers and writes the data to the disk
    

    # Save the final Gaussian features and anchors after training
    save_dict = {
        "anchors": anchors.cpu(),
        "feature": feature.cpu(),
        "voxel_size": VOXEL_SIZE,
        "num_gs": NUM_GS,
    }
    save_path = f"data/{VERSION}_{MODE}/anchors_features_trained_{time}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    torch.save(save_dict, save_path)
    print(f"Saved to {save_path}")

    writer.close()


if __name__ == '__main__': main()