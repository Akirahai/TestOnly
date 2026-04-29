"""Cache the per-scene .npz inputs as a single .pt blob.

First call to `load_scene(scene)` reads the three .npz files (supervise views,
novel views, mapanything depth predictions), converts every used field to a
torch tensor, and writes the bundle to `{cache_dir}/{scene}_data.pt`. Subsequent
calls just torch.load the .pt and skip the npz parse entirely, which dominates
cold-start time on the larger scenes.

Tensors are stored on CPU. Caller is responsible for `.to(device)`.
"""
import os
import numpy as np
import torch


_SUPERVISE_DIR = '/data06/z84301856/scannetpp_v2_opencv_pinhole/supervise_view_fix64_viewInfo_normV14'
_NOVEL_DIR     = '/data06/z84301856/scannetpp_v2_opencv_pinhole/novel_view_fix8_viewInfo_normV14'
_DEPTH_DIR     = ('/data06/z84301856/scannetpp_v2_opencv_pinhole/geometry_priors/'
                  'irregular_pts_align_unipts_normV14/'
                  'mapanything_pcd_normext_imgonly_64view_v2_edgepreserve_csv/'
                  'mapanything_imgonly_v2_32view/depth_pred')

_DEFAULT_CACHE_DIR = 'data/cache'

_K_INTR = 'normed_intrinsics_norm128_v15_fix32'
_K_EXTR = 'extrinsics_Tnorm_norm128_v15_fix32'
_K_IMG  = 'ori_img_tensor'


def load_scene(scene, cache_dir=_DEFAULT_CACHE_DIR):
    """Return a dict of CPU tensors for the given scene, hitting the .pt cache when possible.

    Keys returned:
        train_img   [64, 3, 1168, 1752]   uint8/float (whatever the npz holds)
        train_K     [32, 3, 3]
        train_extr  [32, 4, 4]
        novel_img   [8, 3, 1168, 1752]
        novel_K     [8, 3, 3]
        novel_extr  [8, 4, 4]
        depth_pred  [32, H, W]            mapanything depth prediction
    """
    cache_path = os.path.join(cache_dir, f'{scene}_data.pt')
    if os.path.exists(cache_path):
        print(f'[data_cache] loading cached scene from {cache_path}')
        return torch.load(cache_path, map_location='cpu')

    print(f'[data_cache] cache miss, parsing .npz files for scene {scene!r}')
    image_npz    = np.load(os.path.join(_SUPERVISE_DIR, f'{scene}_viewInfo.npz'))
    image_nv_npz = np.load(os.path.join(_NOVEL_DIR,     f'{scene}_viewInfo.npz'))
    depth_npz    = np.load(os.path.join(_DEPTH_DIR,     f'{scene}.npz'))

    out = {
        'train_img':  torch.from_numpy(np.asarray(image_npz[_K_IMG])),
        'train_K':    torch.from_numpy(np.asarray(image_npz[_K_INTR])),
        'train_extr': torch.from_numpy(np.asarray(image_npz[_K_EXTR])),
        'novel_img':  torch.from_numpy(np.asarray(image_nv_npz[_K_IMG])),
        'novel_K':    torch.from_numpy(np.asarray(image_nv_npz[_K_INTR])),
        'novel_extr': torch.from_numpy(np.asarray(image_nv_npz[_K_EXTR])),
        'depth_pred': torch.from_numpy(np.asarray(depth_npz['data'])),
    }

    os.makedirs(cache_dir, exist_ok=True)
    torch.save(out, cache_path)
    print(f'[data_cache] wrote cache to {cache_path}')
    return out
