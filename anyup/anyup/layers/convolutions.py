from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8,
                 pad_mode="zeros", norm_fn=None, activation_fn=nn.SiLU, use_conv_shortcut=False):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
            N(out_channels),
            activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
