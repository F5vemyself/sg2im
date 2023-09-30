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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np

import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.layers import build_mlp

# lsc add
import torchvision



class Sg2ImModel(nn.Module):
  def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
               gconv_dim=128, gconv_hidden_dim=512,
               gconv_pooling='avg', gconv_num_layers=5,
               refinement_dims=(1024, 512, 256, 128, 64),
               normalization='batch', activation='leakyrelu-0.2',
               mask_size=None, mlp_normalization='none', layout_noise_dim=0,
               **kwargs):
    super(Sg2ImModel, self).__init__()

    # We used to have some additional arguments: 
    # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
    if len(kwargs) > 0:
      print('WARNING: Model got unexpected kwargs ', kwargs)

    self.vocab = vocab
    self.image_size = image_size
    self.layout_noise_dim = layout_noise_dim
    # vocab['object_idx_to_name'] 存储了对象索引到对象名称的映射。
    # 这是一个字典！！！其中键（key）是对象的索引，值（value）是对应的对象名称。
    num_objs = len(vocab['object_idx_to_name'])
    num_preds = len(vocab['pred_idx_to_name'])
    # num_objs + 1 是为了包含这个额外的未知对象或填充对象。
    self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    # 与对象嵌入不同，关系嵌入没有特殊的未知关系或填充关系的概念。
    self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

    if gconv_num_layers == 0:
      self.gconv = nn.Linear(embedding_dim, gconv_dim)
    elif gconv_num_layers > 0:
      gconv_kwargs = {
        # 输入特征的维度，即对象嵌入的维度。
        'input_dim': embedding_dim,
        # 输出特征的维度，即图卷积操作后得到的对象特征的维度。
        'output_dim': gconv_dim,
        # 隐层特征的维度，用于定义图卷积的隐藏层。
        'hidden_dim': gconv_hidden_dim,
        # 图卷积操作中的汇聚方式，用于将邻居节点的特征进行汇聚。
        'pooling': gconv_pooling,
        # MLP（多层感知机）层中是否进行归一化操作。
        'mlp_normalization': mlp_normalization,
      }
      self.gconv = GraphTripleConv(**gconv_kwargs)

    self.gconv_net = None
    if gconv_num_layers > 1:
      gconv_kwargs = {
        'input_dim': gconv_dim,
        'hidden_dim': gconv_hidden_dim,
        'pooling': gconv_pooling,
        # 创建剩余的图卷积层，所以要-1
        'num_layers': gconv_num_layers - 1,
        'mlp_normalization': mlp_normalization,
      }
      self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

    # box_net_dim表示边界框预测网络的输出维度，即边界框的维度。
    box_net_dim = 4
    # box_net_layers是一个列表，其中包含边界框预测网络的层配置，包括输入dim，hidden dim，输出dim
    box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
    self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

    self.mask_net = None
    if mask_size is not None and mask_size > 0:
      # mask_net用于预测对象的掩码，它需要将对象的特征向量映射到一个与掩码相关的输出空间。掩码是一个二进制图像，
      # 需要通过一系列的变换和操作生成。因此，mask_net的结构可能与box_net不同，
      # 可能包括卷积层、上采样层、激活函数等，以便有效地生成对象的掩码。
      # 这里gconv_dim是输入dim；mask_net是layer结构。
      self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

    # 2 * embedding_dim表示将两个对象的嵌入拼接在一起，而+8表示将两个对象的边界框信息（4维）拼接在一起。
    rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
    # rel_aux_net在该模型中用于处理关系的辅助信息，并输出关系的分数，以帮助模型更好地理解和预测图像中对象之间的关系。
    self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

    refinement_kwargs = {
      # layout_dim + layout_noise_dim，表示输入的维度，通过将它们相加并用逗号表示，创建一个包含单个元素的元组。
      # refinement_dims是一个元组，包含了网络模型中每个精细化模块的输出维度。
      # 通过将这两个元组连接在一起，可以得到一个元组，表示整个网络模型中每个层次的维度。
      #  # 第一层dim为layout_dim（160）+layout_noise_dim(1)=161，第二层dim为1024，后面依次
      # (layout_dim + layout_noise_dim,1024，512，256，128，64)
      'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
      # 'normalization'：一个字符串，表示归一化的类型，如'instance'、'batch'等。
      'normalization': normalization,
      # 'activation'：一个字符串，表示激活函数的类型，如'relu'、'leakyrelu'等。
      'activation': activation,
    }
    self.refinement_net = RefinementNetwork(**refinement_kwargs)

  # 该网络模型通过上采样、卷积和批归一化等操作逐渐增加掩码图像的尺寸，并引入非线性变换，最终生成具有指定尺寸的对象掩码。
  def _build_mask_net(self, num_objs, dim, mask_size):
    # num_objs表示对象的数量，该参数在此方法中并没有直接使用，只是作为输入参数的一部分。
    # dim表示图卷积后的特征维度，即输入到掩码网络的通道数。
    # mask_size表示生成的掩码的大小,输出的掩码图像大小。
    output_dim = 1
    layers, cur_size = [], 1
    # 接下来，使用一个循环来构建网络层。每次迭代，将nn.Upsample层（通过双线性插值进行上采样）添加到图层列表中，
    # 将nn.BatchNorm2d层（二维批量归一化）添加到图层列表中，然后将nn.Conv2d层（二维卷积）添加到图层列表中，
    # 使用ReLU激活函数进行非线性变换。然后，将当前大小cur_size乘以2，以便进行下一次迭代。
    # 这个循环会一直进行，直到cur_size大于等于mask_size。
    while cur_size < mask_size:
      layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
      layers.append(nn.BatchNorm2d(dim))
      layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
      layers.append(nn.ReLU())
      cur_size *= 2
    if cur_size != mask_size:
      raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)

  def forward(self, objs, triples, obj_to_img=None,
              boxes_gt=None, masks_gt=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """

    # 得到对象数量和三元组数量
    O, T = objs.size(0), triples.size(0)

    # 将triples在维度1上分成3块，返回一个元组，每个元素都是形状为(T, 1)的张量。
    # 这对应于三元组的三个部分（subject，predicate，object）。
    s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    #print(s,p,o)    # s对应[[0],[2],[0]],p对应[[1],[1],[3]]
    # 对分块后的张量进行循环，使用squeeze(1)操作将形状为(T, 1)的张量转换为形状为(T,)的张量。
    # 现在，s，p，o都具有形状(T,)。
    s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    #print(s,p,o)    # s对应[0,2,0],p对应[1,1,3]
    # subject和object按维度1堆叠，返回一个形状为(T, 2)的张量。这里的edges表示三元组中的边（subject，object）。
    edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)
    # print(edges)
    # tensor([[ 0,  1],
    #         [ 2,  1],
    #         [ 0,  3]])

    if obj_to_img is None:
      # 在 obj_to_img 未提供时，创建一个默认的全零张量，用于表示所有对象属于同一个图像。
      # obj_to_img将用于将对象映射到图像中
      obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

    # 对象向量
    obj_vecs = self.obj_embeddings(objs)
    obj_vecs_orig = obj_vecs
    # 谓词向量
    pred_vecs = self.pred_embeddings(p)

    # 将对象向量、谓词向量通过图卷积提取特征。
    # 如果self.gconv是nn.Linear类型，表示使用线性变换进行图卷积操作。
    if isinstance(self.gconv, nn.Linear):
      obj_vecs = self.gconv(obj_vecs)
    else:
    # 相当于过一层图卷积
      # 此处的obj_vecs,pred_vecs,edges对应gconv的forward方法里的参数
      obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    # 相当于再经过几层图卷积
    if self.gconv_net is not None:
      obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
      # lsc add
      # print("obj_vecs.shape:",obj_vecs.shape)   #[对象个数, 128]
      # print("pred_vecs.shape:", pred_vecs.shape)  #[谓词个数, 128]

    # 经过图卷积之后的obj_vecs经过box_net得到边界框预测
    boxes_pred = self.box_net(obj_vecs)
    # [对象个数，4]
    # print("boxes_pred.shape:",boxes_pred.shape)

    masks_pred = None
    if self.mask_net is not None:
      # 如果self.mask_net不为None，则将对象嵌入表示obj_vecs进行调整形状，其中O是对象的数量，-1表示自动推断维度大小。
      # mask_scores，它表示每个对象的掩码分数。
      mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
      # print('mask_scores',mask_scores)   #[对象个数,1,16,16]
      # 最后，通过对mask_scores进行适当的操作，包括去除多余的维度和应用sigmoid函数，得到预测的掩码masks_pred。
      masks_pred = mask_scores.squeeze(1).sigmoid()
      # print("masks_pred.shape",masks_pred.shape) #[对象个数,16,16]


    # 辅助网络，识别谓词关系，[谓词个数，4]
    # 前面有boxes_pred = self.box_net(obj_vecs)
    s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
    # print("s_boxes.shape:", s_boxes.shape)
    # print("o_boxes.shape:", o_boxes.shape)
    s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
    # s_vecs.shape,o_vecs.shape:[谓词个数，4]
    # print("s_vecs.shape:", s_vecs.shape)
    # print("o_vecs.shape:", o_vecs.shape)
    # 使用torch.cat函数将s_boxes、o_boxes、s_vecs和o_vecs按照指定的维度（dim=1）进行拼接，
    # 形成一个包含主语边界框、客体边界框、主语嵌入和客体嵌入的输入张量rel_aux_input。
    rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
    # 将rel_aux_input作为输入传递给self.rel_aux_net，该网络将对输入进行处理并输出关系的分数rel_scores。
    # 这个分数什么作用？
    # print("rel_aux_input.shape:", rel_aux_input.shape)
    # rel_aux_input.shape: torch.Size([354, 264])
    rel_scores = self.rel_aux_net(rel_aux_input)

    # 这段代码的主要目的似乎是根据不同的条件（是否存在掩码数据）来处理图像中的对象，
    # 并将它们转换为布局信息。这些布局信息可能包括对象在图像中的位置、大小和其他属性。
    H, W = self.image_size
    print("H,W:",H,W)
    # H,W: 64 64


    layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
    # lsc add
    # print("layout_boxes.shape",layout_boxes.shape)
    # layout_boxes.shape torch.Size([204, 4])

    if masks_pred is None:
      # obj_to_img[o] 表示第o个对象所属的图像索引，取值范围在 [0, N)，其中 N 是图像的数量。
      # If obj_to_img[i] = j then vecs[i] belongs to image j.
      layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
      # lsc add
      # print("layout_without_masks_pres:",layout.shape)
    else:
      layout_masks = masks_pred if masks_gt is None else masks_gt
      layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                               obj_to_img, H, W)

      # lsc add 归一化
      # min_value = layout.min()
      # max_value = layout.max()
      # normalized_layout = (layout - min_value) / (max_value - min_value)
      # print("min_value:",min_value)
      # print("max_value:", max_value)
      # print('layyyyout:',normalized_layout)

      # lsc add
      # 将数据类型转换为uint8
      # layout = layout.byte()
      # 创建PIL图像对象
      # image = Image.fromarray(layout[0, 0].cpu().numpy())
      # 保存图像
      # image.save('layout_with_masks_pres.png')

      # lsc add
      from torchvision.utils import save_image
      save_image(layout.data, 'layout_masks.png')
      print("layout_with_maskes_pres:", layout.shape)
      # layout_with_maskes_pres: torch.Size([32, 128, 64, 64])

    if self.layout_noise_dim > 0:
      N, C, H, W = layout.size()
      noise_shape = (N, self.layout_noise_dim, H, W)
      print("noise_shape:", noise_shape)
      # noise_shape: (32, 32, 64, 64)
      layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                 device=layout.device)
      print("layout_noise.shape:",layout_noise.shape)
      # layout_noise.shape: torch.Size([32, 32, 64, 64])
      layout = torch.cat([layout, layout_noise], dim=1)
      print("layout.shape:", layout.shape)
      # layout.shape: torch.Size([32, 160, 64, 64])

      # lsc add
      # torchvision.utils.save_image(layout, 'layout_masks.png')
    img = self.refinement_net(layout)
    return img, boxes_pred, masks_pred, rel_scores

  def encode_scene_graphs(self, scene_graphs):
    """
    Encode one or more scene graphs using this model's vocabulary. Inputs to
    this method are scene graphs represented as dictionaries like the following:

    {
      "objects": ["cat", "dog", "sky"],
      "relationships": [
        [0, "next to", 1],
        [0, "beneath", 2],
        [2, "above", 1],
      ]
    }

    This scene graph has three relationshps: cat next to dog, cat beneath sky,
    and sky above dog.

    Inputs:
    - scene_graphs: A dictionary giving a single scene graph, or a list of
      dictionaries giving a sequence of scene graphs.

    Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
    same semantics as self.forward. The returned LongTensors will be on the
    same device as the model parameters.
    """

    # instance用于检查一个对象是否属于指定的类型。
    if isinstance(scene_graphs, dict):
      # We just got a single scene graph, so promote it to a list
      scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    # enumerate 是一个内置函数，用于在迭代过程中同时获得元素的索引和值。
    # 它接受一个可迭代对象作为输入，并返回一个生成器，该生成器生成包含索引和对应值的元组。
    for i, sg in enumerate(scene_graphs):
      # Insert dummy __image__ object and __in_image__ relationships
      sg['objects'].append('__image__')
      image_idx = len(sg['objects']) - 1
      for j in range(image_idx):
        sg['relationships'].append([j, '__in_image__', image_idx])

      for obj in sg['objects']:
        # # 是一个字典操作，用于获取给定对象名obj在词汇表中的索引。
        obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
        if obj_idx is None:
          raise ValueError('Object "%s" not in vocab' % obj)
        objs.append(obj_idx)
        # 标明图像是第几个
        obj_to_img.append(i)
        # sg的格式
        #     {
        #       "objects": ["cat", "dog", "sky"],
        #       "relationships": [
        #         [0, "next to", 1],
        #         [0, "beneath", 2],
        #         [2, "above", 1],
        #       ]
        #     }
      for s, p, o in sg['relationships']:
        pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
        if pred_idx is None:
          raise ValueError('Relationship "%s" not in vocab' % p)
        triples.append([s + obj_offset, pred_idx, o + obj_offset])
      # 每个场景图中的对象索引是相对于整个输入中的对象序列的。
      # 因此，为了确保每个对象在整个输入中具有唯一的索引，需要对每个场景图中的对象索引进行偏移。
      obj_offset += len(sg['objects'])
    device = next(self.parameters()).device
    objs = torch.tensor(objs, dtype=torch.int64, device=device)
    triples = torch.tensor(triples, dtype=torch.int64, device=device)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
    return objs, triples, obj_to_img

  def forward_json(self, scene_graphs):
    """ Convenience method that combines encode_scene_graphs and forward. """
    # encode_scene_graphs方法将场景图数据转换为模型的输入张量形式，获得objs、triples和obj_to_img这些输入张量。
    objs, triples, obj_to_img = self.encode_scene_graphs(scene_graphs)
    # print("objs",objs)
    # print("triples",triples)
    # print("obj_to_img",obj_to_img)
    # 然后，调用 forward 方法，将上述获得的输入张量传递给模型进行前向传播，得到模型的输出结果。
    return self.forward(objs, triples, obj_to_img)

