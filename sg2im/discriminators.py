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

from sg2im.bilinear import crop_bbox_batch
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0):
    super(PatchDiscriminator, self).__init__()
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x)


class AcDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               padding='same', pooling='avg'):
    super(AcDiscriminator, self).__init__()
    self.vocab = vocab

    cnn_kwargs = {
      # arch: 卷积神经网络的架构。
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    # 返回cnn结构和输出channel
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name'])

    # 伪造图像分类器，将通道数变为1，用于生成real_scores: 真实/伪造图像的得分
    self.real_classifier = nn.Linear(1024, 1)
    # 对象类别分类器
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y):
    if x.dim() == 3:
      # 例如，如果 x 是一个形状为 (3, 64, 64) 的张量，那么 x[:, None] 操作将返回一个形状为 (3, 1, 64, 64) 的张量，
      # 其中第二个维度被扩展为 1。总之，x[:, None] 的操作是在张量 x 的第二个维度位置上增加一个新的维度，用于模拟批量处理的形式。
      x = x[:, None]
    vecs = self.cnn(x)
    # 真实/伪造图像的得分
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    # 对象类别的损失 ac_loss
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(AcCropDiscriminator, self).__init__()
    self.vocab = vocab
    self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img):
    crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
    real_scores, ac_loss = self.discriminator(crops, objs)
    return real_scores, ac_loss
