from torch import nn



def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

"""
This function creates a standard convolutional layer followed by optional Batch Normalization (bn) and Rectified Linear Unit (ReLU) activation.
Parameters:
in_channels: Number of input channels.
out_channels: Number of output channels.
kernel_size: Size of the convolutional kernel (default is 3).
padding: Padding added to all sides of the input (default is 1).
bn: Boolean indicating whether to include Batch Normalization (default is True).
dilation: Dilation rate for the convolution (default is 1).
stride: Stride of the convolution (default is 1).
relu: Boolean indicating whether to include ReLU activation (default is True).
bias: Boolean indicating whether to include bias in the convolution (default is True).

"""

def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

"""
This function creates a depthwise separable convolutional layer, which is a lightweight alternative to standard convolutions. It performs spatial convolutions independently for each channel and then combines them. It consists of a depthwise convolution followed by 1x1 pointwise convolution.
Parameters:
Same as the conv function, plus:
groups: Equal to the number of input channels for depthwise convolution.
"""

def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )

"""
Similar to conv_dw, but without Batch Normalization. It uses the ELU (Exponential Linear Unit) activation instead of ReLU.
Parameters:
Same as the conv_dw function.
"""