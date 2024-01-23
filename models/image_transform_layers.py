import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Differentiable affine transformation
class AffineTransform(nn.Module):
    def __init__(self, identity_transform):
        super().__init__()

        def makep(x):
            x = torch.tensor(x).float()
            return nn.Parameter(x, requires_grad=False)

        self.scale = makep(1. if identity_transform else random.gauss(1, 0.02))
        self.transX = makep(0. if identity_transform else random.randint(-2, 2) / 255.)
        self.transY = makep(0. if identity_transform else random.randint(-2, 2) / 255.)

        self.theta = nn.Parameter(torch.tensor([
            [self.scale, 0, self.transX],
            [0, self.scale, self.transY]
        ])[None, ...])

    def forward(self, x):
        with torch.no_grad():
            grid = F.affine_grid(self.theta.repeat(x.shape[0],1,1), x.size(), align_corners=False).to(device)
        return F.grid_sample(x, grid, align_corners=False)


# Differentiable rotation transformation
class RotationTransform(nn.Module):
    def __init__(self, identity_transform):
        super().__init__()

        def makep(x):
            x = torch.tensor(x).float()
            return nn.Parameter(x, requires_grad=False)

        self.angle = makep(0. if identity_transform else random.randint(-2, 2) * np.pi/180) #-3, 3
        self.theta = nn.Parameter(torch.tensor([[torch.cos(self.angle), -torch.sin(self.angle), 0],
                             [torch.sin(self.angle), torch.cos(self.angle), 0]])[None])

    def forward(self, x):
        with torch.no_grad():
            grid = F.affine_grid(self.theta.repeat(x.shape[0],1,1), x.size(), align_corners=False).to(device)
        return F.grid_sample(x, grid, align_corners=False)
