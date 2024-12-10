"""
Builds ResNet18 from scratch using PyTorch.
This does not build generalized blocks for all ResNets, just for ResNet18.
Paper => Deep Residual Learning for Image Recognition.
Link => https://arxiv.org/pdf/1512.03385v1.pdf
"""

import torch.nn as nn
import torch

from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion)
        # self.bn2 = nn.LayerNorm(shapes[out_channels * self.expansion])

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int = 256,
        block: Type[BasicBlock] = BasicBlock,
    ) -> None:
        super(ResNet, self).__init__()

        layers = [2, 2, 2, 2, 2]
        self.expansion = 1

        self.in_channels = 64

        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv1d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, self.in_channels, layers[0], stride=3)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=3)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=3)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=3)

        dropout_rate = 0.3

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.attn = nn.MultiheadAttention(512*3, 8)
        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),
        )

    def _make_layer(
        self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, self.expansion, downsample)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, expansion=self.expansion)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1, 408)

        # smooth the data
        # augment when training
        with torch.no_grad():
            x = self.gaussian_smoothing(x, augment=self.training)

        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # print(x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x, _ = self.attn(x, x, x)
        x = self.fc(x)
        return x

    def train_loss(self, output, target):
        # prediction_loss = nn.functional.mse_loss(output, target)
        # use huber loss
        prediction_loss = nn.functional.smooth_l1_loss(output, target)
        return prediction_loss

    def test_loss(self, output, target):
        x = output
        return nn.functional.pairwise_distance(x, target).mean()


    def gaussian_smoothing(self, data, augment):
        if augment:
            leave_out = 8
            select_start_pos = torch.randint(0, leave_out, (data.shape[0],))
            place_start_pos = torch.randint(0, leave_out, (data.shape[0],))
            num_units = 408 - leave_out
            new_data = torch.zeros_like(data)
            for i in range(data.shape[0]):
                new_data[i, :, place_start_pos[i]:place_start_pos[i]+num_units] = data[i, :, select_start_pos[i]:select_start_pos[i]+num_units]
            data = new_data

        return data

        # data_shape = data.shape
        # data = data.view(-1, 1, data_shape[-1])            

        # kernel_size = 3
        # sigma = 1.0

        # # Create Gaussian kernel
        # x = torch.arange(kernel_size, dtype=torch.float32, device=data.device) - kernel_size // 2
        # gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        # gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # # Reshape to match the dimensions of the data
        # gaussian_kernel = gaussian_kernel.view(1, 1, -1)

        # # Apply Gaussian smoothing (conv1d) along the last dimension
        # smoothed = nn.functional.conv1d(
        #     data,
        #     gaussian_kernel,
        #     padding=kernel_size // 2,
        #     groups=1  # Single-channel convolution for each sequence
        # )
        
        # # Reshape back to original shape
        # return smoothed.view(data_shape)


if __name__ == "__main__":
    tensor = torch.rand([3, 256, 408])
    model = ResNet()

    output = model(tensor)
    model.train()
    print(output.shape)
