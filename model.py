import torch
from typing import Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
from utils import autopad


class ConvBlock(nn.Module):
    
    def __init__(self, 
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Union[int, None] = None,
                 activation: bool = True,
                 **kwargs): 
        super(ConvBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=autopad(k=kernel_size, p=padding), **kwargs)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)
        if self.training:
            out = self.dropout(out)
        return out

class MaxPooling(nn.Module): 

    def __init__(self,
                 input_channels: int,
                 kernel_size: Tuple[int, int] = (2, 2),
                 padding: Union[int, None] = None,
                 **kwargs):
        super(MaxPooling, self).__init__()
        self.input_channels = input_channels
        self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                 padding=autopad(k=kernel_size, p=padding),
                                 **kwargs)
                                 
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.pool(x)
        return out

class ConvTranspose(nn.Module): 
    
    def __init__(self,
                 input_channels: int,
                 output_channels: int, 
                 kernel_size: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (2, 2),
                 padding: Union[int, None] = None,
                 **kwargs):
        super(ConvTranspose, self).__init__()
        self.input_channels = input_channels 
        self.output_channels = output_channels
        self.transpose = nn.ConvTranspose2d(in_channels=self.input_channels,
                                            out_channels=self.output_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=autopad(k=kernel_size, p=padding),
                                            **kwargs)
        
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor
                ) -> torch.Tensor:
        out = self.transpose(x1)
        diffY = x2.size()[2] - out.size()[2]
        diffX = x2.size()[3] - out.size()[3]

        out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        out = torch.cat([x2, out], dim=1)
        return out

class DcBlock(nn.Module): 

    def __init__(self,
                 corresponding_filters: int,
                 input_channels: int, 
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Union[int, None] = None,
                 alpha: float = 1.67,
                 **kwargs):
        super(DcBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = None
        self.w = corresponding_filters * alpha
        self.add_channels = 0 if not kwargs.get('add_channels') else kwargs.get('add_channels')
        self.filters = [int(self.w*0.167), int(self.w*0.333), int(self.w*0.5)]
        self.left_module = nn.ModuleList()
        self.right_module = nn.ModuleList()

        for i, filter in enumerate(self.filters): 
            if i == 0: 
                self.left_module.append(module=ConvBlock(input_channels=self.input_channels + self.add_channels,
                                                         output_channels=filter,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=autopad(k=kernel_size, p=padding)))
                
                self.right_module.append(module=ConvBlock(input_channels=self.input_channels + self.add_channels,
                                                          output_channels=filter,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          padding=autopad(k=kernel_size, p=padding)))
            else: 
                self.left_module.append(module=ConvBlock(input_channels=self.left_module[i-1].output_channels,
                                                         output_channels=filter,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=autopad(k=kernel_size, p=padding)))
                self.right_module.append(module=ConvBlock(input_channels=self.right_module[i-1].output_channels,
                                                          output_channels=filter,
                                                          kernel_size=kernel_size,
                                                          stride=stride,
                                                          padding=autopad(k=kernel_size, p=padding)))

            self.input_channels, self.output_channels = filter, filter
        
        self.output_channels = sum(self.filters)
        self.bn = nn.BatchNorm2d(num_features=self.output_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out_left = []
        out_right = []

        for i, left, right in zip(range(3), self.left_module, self.right_module):
            if i == 0:
                out_left.append(left(x))
                out_right.append(left(x))
            else: 
                out_left.append(left(out_left[i-1]))
                out_right.append(right(out_right[i-1]))
                
        out1 = torch.cat(tensors=out_left, dim=1)
        out1 = self.bn(out1)
        out2 = torch.cat(tensors=out_right, dim=1)
        out2 = self.bn(out2)
        out = torch.add(out1, out2)    
        out = F.relu(out)
        out = self.bn(out)

        return out

class ResPath(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 output_channels: int,
                 length: int,
                 padding: Union[int, None] = None): 
        super(ResPath, self).__init__()
        self.length = length
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = ConvBlock(input_channels=self.input_channels,
                               output_channels=self.output_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=autopad(k=(1, 1), p=padding),
                               activation=False)
        
        self.conv2 = ConvBlock(input_channels=self.input_channels,
                               output_channels=self.output_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=autopad(k=(3, 3), p=padding))
        
        self.bn = nn.BatchNorm2d(num_features=self.conv2.output_channels)
        
        self.module = nn.ModuleList()

        for i in range(self.length-1):
            self.module.append(module=ConvBlock(input_channels=self.output_channels,
                                                output_channels=self.output_channels,
                                                kernel_size=(1, 1),
                                                stride=(1, 1),
                                                padding=autopad(k=(1, 1), p=padding)))
            self.module.append(module=ConvBlock(input_channels=self.output_channels,
                                                output_channels=self.output_channels,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=autopad(k=(3, 3), p=padding)))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        shortcut = x
        shortcut = self.conv1(shortcut)
        out = self.conv2(x)
        out = torch.add(shortcut, out)
        out = F.relu(out)
        out = self.bn(out)

        for i in range(self.length-1):
            shortcut = out
            shortcut = self.module[i](shortcut)
            out = self.module[i+1](out)
            out = torch.add(shortcut, out)
            out = F.relu(out)
            out = self.bn(out)

        return out

class DcUnet(nn.Module):

    def __init__(self,
                 input_channels: int):
        super(DcUnet, self).__init__()
        self.dc_block_1 = DcBlock(corresponding_filters=32, 
                                  input_channels=input_channels)
        self.pool1 = MaxPooling(input_channels=self.dc_block_1.output_channels)
        self.res_path_1 = ResPath(input_channels=self.dc_block_1.output_channels,
                                  output_channels=32,
                                  length=4)
        self.dc_block_2 = DcBlock(corresponding_filters=32*2, 
                                  input_channels=self.pool1.input_channels)
        self.pool2 = MaxPooling(input_channels=self.dc_block_2.output_channels)
        self.res_path_2 = ResPath(input_channels=self.dc_block_2.output_channels,
                                  output_channels=32*2,
                                  length=3)
                                  
        self.dc_block_3 = DcBlock(corresponding_filters=32*4, 
                                  input_channels=self.pool2.input_channels)
        self.pool3 = MaxPooling(input_channels=self.dc_block_3.output_channels)
        self.res_path_3 = ResPath(input_channels=self.dc_block_3.output_channels,
                                  output_channels=32*4,
                                  length=2)

        self.dc_block_4 = DcBlock(corresponding_filters=32*8, 
                                  input_channels=self.pool3.input_channels)
        self.pool4 = MaxPooling(input_channels=self.dc_block_4.output_channels)
        self.res_path_4 = ResPath(input_channels=self.dc_block_4.output_channels,
                                  output_channels=32*8,
                                  length=1)
        self.dc_block_5 = DcBlock(corresponding_filters=32*16,
                                  input_channels=self.pool4.input_channels)

        self.up1 = ConvTranspose(input_channels=self.dc_block_5.output_channels,
                                 output_channels=32*8*2)
        self.dc_block_6 = DcBlock(corresponding_filters=32*8,
                                  input_channels=self.up1.output_channels,
                                  add_channels=256)

        self.up2 = ConvTranspose(input_channels=self.dc_block_6.output_channels,
                                 output_channels=32*4*2)
        self.dc_block_7 = DcBlock(corresponding_filters=32*4,
                                  input_channels=self.up2.output_channels,
                                  add_channels=128)

        self.up3 = ConvTranspose(input_channels=self.dc_block_7.output_channels,
                                 output_channels=32*2*2)
        self.dc_block_8 = DcBlock(corresponding_filters=32*2,
                                  input_channels=self.up3.output_channels,
                                  add_channels=64)

        self.up4 = ConvTranspose(input_channels=self.dc_block_8.output_channels,
                                 output_channels=32*2)
        self.dc_block_9 = DcBlock(corresponding_filters=32,
                                  input_channels=self.up4.output_channels,
                                  add_channels=32)

        self.conv_out = ConvBlock(input_channels=self.dc_block_9.output_channels,
                                  output_channels=1,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=autopad(k=(1, 1), p=None))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        dc_block_1 = self.dc_block_1(x)
        pool_1 = self.pool1(dc_block_1)
        dc_block_1 = self.res_path_1(dc_block_1)

        dc_block_2 = self.dc_block_2(pool_1)
        pool_2 = self.pool2(dc_block_2)
        dc_block_2 = self.res_path_2(dc_block_2)

        dc_block_3 = self.dc_block_3(pool_2)
        pool_3 = self.pool3(dc_block_3)
        dc_block_3 = self.res_path_3(dc_block_3)

        dc_block_4 = self.dc_block_4(pool_3)
        pool4 = self.pool4(dc_block_4)
        dc_block_4 = self.res_path_4(dc_block_4)

        dc_block_5 = self.dc_block_5(pool4)

        up1 = self.up1(dc_block_5, dc_block_4)
        dc_block_6 = self.dc_block_6(up1)

        up2 = self.up2(dc_block_6, dc_block_3)
        dc_block_7 = self.dc_block_7(up2)

        up3 = self.up3(dc_block_7, dc_block_2)
        dc_block_8 = self.dc_block_8(up3)

        up4 = self.up4(dc_block_8, dc_block_1)
        dc_block_9 = self.dc_block_9(up4)

        out = self.conv_out(dc_block_9)
        out = F.sigmoid(out)

        return out
