from torch import nn
from typing import Union, Tuple


class ResConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, num_blocks: int = 2):
        super(ResConv, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks

        sub_blocks = []
        for i in range(num_blocks):
            sub_blocks.append(nn.Conv2d(in_channels=channels, out_channels=channels,
                                     kernel_size=kernel_size, padding="same"))
            sub_blocks.append(nn.ELU())

        self.sub_blocks = nn.ModuleList(sub_blocks)

    def forward(self, x):
        output = x
        for module in self.sub_blocks:
            output = module(output)

        # skip connection
        return x + output


class ConvElu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 activation: bool = True):
        super(ConvElu, self).__init__()

        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        if self.activation:
            self.elu = nn.ELU()

    def forward(self, x):
        if not self.activation:
            return self.conv(x)
        return self.elu(self.conv(x))


class UpConvElu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 input_padding: Union[int, Tuple[int, int]],
                 output_padding: Union[int, Tuple[int, int]],
                 activation: bool = True):
        super(UpConvElu, self).__init__()

        self.activation = activation
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=input_padding, output_padding=output_padding)
        if self.activation:
            self.elu = nn.ELU()

    def forward(self, x):
        if not self.activation:
            return self.up(x)
        return self.elu(self.up(x))


class UpsampleAndSmooth(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 out_height: int, out_width: int, smooth_kernel_size: int = 5,
                 activation: bool = True):
        super(UpsampleAndSmooth, self).__init__()

        self.activation = activation
        upsampler = nn.Upsample(size=(out_height, out_width), mode='bilinear')
        smoother = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             padding="same", kernel_size=smooth_kernel_size)
        self.up = nn.Sequential(upsampler, smoother)
        if self.activation:
            self.elu = nn.ELU()

    def forward(self, x):
        x = self.up(x)
        if not self.activation:
            return x
        return self.elu(x)


class Reshaper(nn.Module):
    def __init__(self, out_shape: Tuple[int, int, int]):
        super(Reshaper, self).__init__()

        # out_shape is a tuple of (C, H, W)
        self.out_shape = out_shape

    def forward(self, x):
        return x.reshape((-1, self.out_shape[0], self.out_shape[1], self.out_shape[2]))
