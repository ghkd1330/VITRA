#!/usr/bin/env python3
"""Exit 0 if PyTorch3D mesh rasterization works on CUDA, 1 if rebuild needed, 2 skip (no GPU)."""
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(2)

try:
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes

    verts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device="cuda")
    meshes = Meshes(verts=verts, faces=faces.unsqueeze(0))
    rasterize_meshes(meshes, image_size=64, blur_radius=0.0, faces_per_pixel=1)
except RuntimeError as e:
    msg = str(e).lower()
    if "not compiled with gpu" in msg or "no kernel image" in msg:
        sys.exit(1)
    raise
sys.exit(0)
