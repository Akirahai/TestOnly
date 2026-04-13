TorchSplat: Pure PyTorch 3D Gaussian Rasterization

TorchSplat is a complete, native PyTorch implementation of a 3D Gaussian Splatting (3DGS) rasterizer.

Traditionally, 3DGS relies heavily on custom CUDA kernels (e.g., diff-gaussian-rasterization or gsplat) for performance. TorchSplat implements the entire forward pass—including EWA projection, tile-based intersection sorting, Spherical Harmonics (SH) evaluation, and $\alpha$-compositing—purely using PyTorch tensor operations.

This provides a highly readable, easily debuggable, and deeply introspectable pipeline with native Autograd support, ideal for researchers and developers experimenting with the core mechanics of Gaussian Splatting.

🏗️ System Architecture

The project is modularized into 5 core components, orchestrated by a central wrapper.

File

Component

Description

rasterizer_torchsplat.py

Main Wrapper

Contains the Rasterizer class, the primary API entry point. Initializes image bounds, padding, and handles input validation.

rendering.py

Pipeline Orchestrator

Coordinates the workflow. Passes data between projection, tile intersection, and pixel rasterization phases. Supports multiple render modes (RGB, D, ED).

EWA_fully_fused_proj_packed.py

Projection Engine

Transforms 3D Gaussians into 2D camera space. Reconstructs 3D covariance, applies affine transformations via a Jacobian, and computes 2D bounding radii.

rasterization_utils.py

Rasterization Core

Maps projected Gaussians to $16 \times 16$ screen tiles. Calculates pixel-wise transmittance and $\alpha$-blending to yield final image arrays.

sh_utils.py

Color Evaluation

Computes view-dependent Spherical Harmonics (up to Degree 4) to determine the exact RGB value of a Gaussian from a specific camera angle.

⚙️ How It Works: The Pipeline

TorchSplat accurately mimics the standard 3DGS rendering pipeline in four distinct stages:

1. Elliptical Weighted Average (EWA) Projection

Gaussians are defined in 3D by their mean $\mu$, scale $S$, and rotation $R$ (quaternion). The projection engine computes the 3D covariance matrix $\Sigma_{3D} = R S S^T R^T$. Using a first-order Taylor approximation with the camera's Jacobian $J$ and view matrix $W$, it computes the 2D screen-space covariance:


$$\Sigma_{2D} = J W \Sigma_{3D} W^T J^T$$


A low-pass filter is applied to prevent aliasing.

2. Tile Intersection & Sorting

To optimize rendering, the image is divided into $16 \times 16$ pixel tiles. The engine calculates the bounding box of each 2D Gaussian and identifies overlapping tiles. Intersections are encoded into IDs containing the tile ID and Gaussian depth, allowing for an efficient global sort to establish a front-to-back rendering order.

3. Spherical Harmonics (SH)

Instead of static colors, Gaussians use SH coefficients to represent view-dependent lighting and reflections. The eval_sh function computes the final color based on the camera's viewing direction relative to the Gaussian.

4. Alpha Compositing

Pixels are evaluated by accumulating the active Gaussians that overlap them. The engine provides three different implementations in rasterization_utils.py (ranging from educational nested loops to highly vectorized pixel-grid operations). The contribution $\alpha$ of a Gaussian at pixel $(x, y)$ is:


$$\sigma = \frac{1}{2} (A \Delta x^2 + C \Delta y^2) + B \Delta x \Delta y$$

$$\alpha = \text{opacity} \times e^{-\sigma}$$


Colors are accumulated front-to-back using $C = \sum_{i} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$.

🚀 Quick Start

Initialize the Rasterizer and pass your camera intrinsics, extrinsics, and Gaussian parameters.

import torch
from rasterizer_torchsplat import Rasterizer

# 1. Initialize the rasterizer wrapper
rasterizer = Rasterizer(white_bkgd=True)

# 2. Define camera properties
width, height = 800, 600
tile_size = 16

world_to_camera_matrix = torch.eye(4, device="cuda")
intrinsics_matrix = torch.tensor([
    [800.0, 0.0, 400.0], 
    [0.0, 800.0, 300.0], 
    [0.0, 0.0, 1.0]
], device="cuda")

# 3. Prepare your Gaussian dictionary (Mock data shown here)
num_gaussians = 10000
splats = {
    "mean": torch.randn((num_gaussians, 3), device="cuda"),
    "rotation": torch.randn((num_gaussians, 4), device="cuda"), # Quaternions
    "scale": torch.rand((num_gaussians, 3), device="cuda") * 0.1,
    "opacity": torch.rand((num_gaussians, 1), device="cuda"),
    "color": torch.rand((num_gaussians, 3), device="cuda") # Or SH coeffs
}

# 4. Render the image
render_colors, render_alphas = rasterizer.gpu_rasterize_splats(
    w2c=world_to_camera_matrix,
    Knorm=intrinsics_matrix,
    width=width,
    height=height,
    tile_size=tile_size,
    splats=splats,
    active_sh_degree=0 # Increase if using SH coefficients
)

print(f"Rendered Image Shape: {render_colors.shape}")


⚠️ Notes & Limitations

Performance vs. Introspection: Because this engine relies on dense PyTorch tensor operations rather than segmented CUDA reductions and atomic operations, it uses significantly more memory. It is designed for algorithmic clarity, debugging, and experimentation rather than real-time rendering of massive (1M+ splats) scenes.

Batched Rendering: The underlying EWA_fully_fused_proj_packed.py supports batched camera projections, enabling multi-view evaluation simultaneously.

Rendering Modes: Switch between RGB, D (Depth), and ED (Expected Depth) via the render_mode argument in the pipeline.
