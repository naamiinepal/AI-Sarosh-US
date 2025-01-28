import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2D(nn.Module):
    def __init__(self, kernel_size, activation_fn):
        """
        Implements a separable 2D convolution in PyTorch.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the depthwise kernel.
            stride (int or tuple): Stride of the depthwise convolution.
            padding (int or tuple): Padding for the depthwise convolution.
            dilation (int or tuple): Dilation rate for atrous convolution.
            bias (bool): Whether to include a bias term in the pointwise convolution.
        """
        super(SeparableConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn

        # Depthwise convolution


    def forward(self, x, in_channels, out_channels, multiplier, stride=1, padding="same", dilation=1, bias=True):
        depthwise = nn.Conv2d(
            in_channels,
            in_channels*multiplier,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Ensures depthwise operation
            bias=False  # No bias for depthwise convolution
        )

        # Pointwise convolution
        pointwise = nn.Conv2d(
            in_channels*multiplier,
            out_channels,
            kernel_size=1,  # Pointwise uses 1x1 kernel
            stride=1,
            padding=0,
            bias=bias
        )
        # Apply depthwise convolution
        x = depthwise(x)

        # Apply pointwise convolution
        x = pointwise(x)
        
        if self.activation_fn:
            x = self.activation_fn(x)

        return x
