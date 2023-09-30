#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2im.layers import get_normalization_2d
from sg2im.layers import get_activation
from sg2im.utils import timeit, lineno, get_gpu_memory


"""
Cascaded refinement network architecture, as described in:

Qifeng Chen and Vladlen Koltun,
"Photographic Image Synthesis with Cascaded Refinement Networks",
ICCV 2017
"""

# CRM,单个级联细化模块
class RefinementModule(nn.Module):
  def __init__(self, layout_dim, input_dim, output_dim,
               normalization='instance', activation='leakyrelu'):
    super(RefinementModule, self).__init__()
    
    layers = []
    layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                            kernel_size=3, padding=1))
    # lsc add
    # print("111layout_dim.shape:",layout_dim)
    # print("2222input_dim.shape:", input_dim)
    # 初始 111layout_dim.shape: 160
    # 初始 2222input_dim.shape: 1
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    self.net = nn.Sequential(*layers)

  def forward(self, layout, feats):
    _, _, HH, WW = layout.size()
    # print("layout.size():",layout.size())
    # 初始 torch.Size([32, 160, 64, 64])
    _, _, H, W = feats.size()
    # print("feats.size():",feats.size())
    # 初始 torch.Size([32, 1, 4, 4])
    assert HH >= H
    if HH > H:
      # factor每次不一样
      factor = round(HH // H)
      # print("factor:",factor)
      assert HH % factor == 0
      assert WW % factor == 0 and WW // factor == W
      layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
      # print("layout.shape:",layout.shape)
      # layout.shape: torch.Size([32, 160, 4, 4])
    net_input = torch.cat([layout, feats], dim=1)
    # net_input.shape torch.Size([32, 161, 4, 4])
    # print("net_input.shape",net_input.shape)
    out = self.net(net_input)
    # print("out:",out.shape)
    # out: torch.Size([32, 1024, 4, 4])
    return out

# 级联细化网络CRN
class RefinementNetwork(nn.Module):
  def __init__(self, dims, normalization='instance', activation='leakyrelu'):
    super(RefinementNetwork, self).__init__()
    # (layout_dim + layout_noise_dim,1024，512，256，128，64)
    layout_dim = dims[0]
    self.refinement_modules = nn.ModuleList()
    for i in range(1, len(dims)):
      # 第一层dim为layout_dim（160）+layout_noise_dim(1)，第二层dim为1024，后面依次
      input_dim = 1 if i == 1 else dims[i - 1]
      output_dim = dims[i]
      mod = RefinementModule(layout_dim, input_dim, output_dim,
                             normalization=normalization, activation=activation)
      self.refinement_modules.append(mod)
    output_conv_layers = [
      nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
      get_activation(activation),
      nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
    ]
    nn.init.kaiming_normal_(output_conv_layers[0].weight)
    nn.init.kaiming_normal_(output_conv_layers[2].weight)
    self.output_conv = nn.Sequential(*output_conv_layers)

  def forward(self, layout):
    """
    Output will have same size as layout
    """
    # H, W = self.output_size
    N, _, H, W = layout.size()
    self.layout = layout
    # print("layout:",layout.shape)
    # layout: torch.Size([32, 160, 64, 64])

    # Figure out size of input
    input_H, input_W = H, W
    # input_H，input_W //2, this operation repeat 5 times (because the len of refinement_modules is 5)
    for _ in range(len(self.refinement_modules)):
      input_H //= 2
      input_W //= 2


    assert input_H != 0
    assert input_W != 0

    feats = torch.zeros(N, 1, input_H, input_W).to(layout)
    print("feats.shape:",feats.shape)
    # feats.shape: torch.Size([32, 1, 2, 2])
    for mod in self.refinement_modules:
      feats = F.upsample(feats, scale_factor=2, mode='nearest')
      # 1111feats.shape: torch.Size([32, 1, 4, 4])
      # print("1111feats.shape:",feats.shape)
      # 经过一次CRM
      feats = mod(layout, feats)
      # 2222feats.shape: torch.Size([32, 1024, 4, 4])
      # print("2222feats.shape:",feats.shape)

    out = self.output_conv(feats)
    # 经过5次CRM，最后，out.shape: torch.Size([32, 3, 64, 64])
    print("out.shape:",out.shape)
    return out

