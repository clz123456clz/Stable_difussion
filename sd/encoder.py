import torch 
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(128, 256),

            VAE_ResidualBlock(256, 256),

            # (B, 256, H/2, W/2) -> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),

            VAE_ResidualBlock(512, 512),

            # (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_AttentionBlock(512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            nn.GroupNorm(num_groups=32, num_channels=512),

            # (B, 512, H/8, W/8) -> (B, 512, H/8, W/8)
            nn.SiLU(),

            # (B, 512, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (B, 8, H/8, W/8) -> (B, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (B, 3(in_channels), H, W); noise: (B, 8(out_channels), H/8, W/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (padding_left, padding_right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # x: (B, 8, H/8, W/8) -> two tensors of shape (B, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        log_variance = torch.clamp(log_variance, min=-30.0, max=20.0)

         # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        variance = log_variance.exp()

        # (B, 4, H/8, W/8) -> (B, 4, H/8, W/8)
        stdev = variance.sqrt()

        # z ~ N(0, 1) -> x ~ N(mean, variance)
        # x = mean + stdev * z
        x = mean + stdev * noise

        # scale the output by a constant factor
        x *= 0.18215

        return x

