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

# GraphTripleConv 类是一个场景图卷积的单层，它接受对象向量、谓词向量和边信息作为输入，并输出更新后的对象向量和谓词向量。
# 在该类的构造函数中，可以指定输入维度、输出维度、隐藏维度、池化方式以及 MLP（多层感知机）的标准化方式等参数。

# 在这个上下文中，边信息是指描述图中对象之间关系的信息。在场景图中，边表示两个对象之间的关系，
# 例如一个谓词（predicate）可以连接两个对象（subject和object），表示它们之间的关系或动作。

# 在代码中，edges 参数是一个 LongTensor，它的形状为 (T, 2)，其中 T 表示谓词的数量，2 表示每个谓词对应的两个对象的索引。
# edges[k] = [i, j] 表示存在一个三元组 [obj_vecs[i], pred_vecs[k], obj_vecs[j]]，
# 其中 obj_vecs 是对象向量，pred_vecs 是谓词向量。

# GraphTripleConvNet 是一个由多个 GraphTripleConv 层组成的序列。它接收对象向量、谓词向量和边信息作为输入，
# 并通过多个 GraphTripleConv 层依次对输入进行场景图卷积操作。

import torch
import torch.nn as nn
from sg2im.layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none'):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    
    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)
    
    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim
    
    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()
    
    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]
    
    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]


    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    # .view(1, -1)：这种形式的 .view() 操作将张量重新塑造为一个行数为 1，列数自动推断的二维张量。
    # .view(-1, 1)：这种形式的 .view() 操作将张量重新塑造为一个列数为 1，行数自动推断的二维张量。
    # expand()函数括号中的输入参数为指定经过维度尺寸扩展后的张量的size。
    # expand（）函数只能将size=1的维度扩展到更大的尺寸，如果扩展其他size（）的维度会报错。
    # a = torch.tensor([1, 2, 3])
    # c = a.expand(2, 3)
    # # 输出信息：
    # tensor([1, 2, 3])
    # tensor([[1, 2, 3],
    #         [1, 2, 3]]
    # expand_as（）函数与expand（）函数类似，功能都是用来扩展张量中某维数据的尺寸，
    # 区别是它括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。

    # 对出现在多个三元组中的对象向量进行求和操作。
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    # scatter_add（dim,index,src），dim即维度，是对于self而言的，即在self的哪一dim进行操作
    # index是索引，即要在self的哪一index进行操作,src是待操作的源数字，比较好理解
    # index的维度等于src的维度，相当于将src的每一个数字都加到self的对应index上；

    # 使用 scatter_add 函数将 new_s_vecs（主语对象向量）和 new_o_vecs（宾语对象向量）按照索引
    # s_idx_exp 和 o_idx_exp 散布到 pooled_obj_vecs 上，并执行累加操作。
    # 这个操作的目的是将每个对象向量累加（汇总）到 pooled_obj_vecs 中，以便后续计算。
    # 这样，如果一个对象在多个三元组中出现，它的向量将根据不同的三元组被累加到 pooled_obj_vecs 中，
    # 以捕获其在不同上下文中的信息。
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      # obj_counts计算每个对象在三元组中出现的次数
      obj_counts = torch.zeros(O, dtype=dtype, device=device)
      ones = torch.ones(T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)
  
      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      # 将新的对象向量除以它们在三元组中出现的次数
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (O, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none'):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim': input_dim,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
    }
    for _ in range(self.num_layers):
      self.gconvs.append(GraphTripleConv(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs


