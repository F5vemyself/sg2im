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

import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('datasets/vg')
COCO_DIR = os.path.expanduser('../datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight 
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)


# 用于计算总损失并将其添加到损失字典中
def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  # 将当前的损失值乘以权重，得到加权后的损失值。
  curr_loss = curr_loss * weight
  # 通过调用 curr_loss.item()，可以获取这个张量中的数值，并将其转换为 Python 中的标量（即普通的数字类型，如整数或浮点数）。
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss


# 用于检查输入参数args是否满足特定条件。函数的主要目的是确保在细化网络（refinement network）中的层数不超过限制。
def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


# 根据输入的参数和词汇表（vocab），它返回一个模型实例和模型的关键字参数（kwargs）。
def build_model(args, vocab):
  if args.checkpoint_start_from is not None:
    checkpoint = torch.load(args.checkpoint_start_from)
    # checkpoint['model_kwargs']是在保存检查点时存储的模型的关键字参数。
    kwargs = checkpoint['model_kwargs']
    model = Sg2ImModel(**kwargs)

    # 下面这段代码：通过这个过程，就是对checkpoint_start_from的模型参数进行处理，使其后面可以用。

    # raw_state_dict 包含了模型的状态字典，其中包含了模型的所有参数和持久化缓冲区。
    # raw_state_dict通常具有以下常见的键:
    # 模型的权重参数（如卷积层、线性层的权重矩阵）；
    # 模型的偏置参数（如卷积层、线性层的偏置向量）
    # 用于特定模块的缓冲区（如Batch Normalization的均值和方差）
    raw_state_dict = checkpoint['model_state']
    state_dict = {}
    for k, v in raw_state_dict.items():
      # 如果键（key）以 'module.' 开头，说明它是由 torch.nn.DataParallel 模块包装的多GPU模型的参数。
      if k.startswith('module.'):
        # 将键名中的 'module.' 部分去除，得到原始的键名。
        k = k[7:]
      state_dict[k] = v
    model.load_state_dict(state_dict)
  else:
    # 如果重头训练，就直接赋予kwargs
    kwargs = {
      'vocab': vocab,
      'image_size': args.image_size,
      'embedding_dim': args.embedding_dim,
      'gconv_dim': args.gconv_dim,
      'gconv_hidden_dim': args.gconv_hidden_dim,
      'gconv_num_layers': args.gconv_num_layers,
      'mlp_normalization': args.mlp_normalization,
      'refinement_dims': args.refinement_network_dims,
      'normalization': args.normalization,
      'activation': args.activation,
      'mask_size': args.mask_size,
      'layout_noise_dim': args.layout_noise_dim,
    }
    model = Sg2ImModel(**kwargs)
  return model, kwargs


def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  # args.discriminator_loss_weight 是一个参数，用于表示鉴别器损失在总体损失中的权重。default=0.01
  d_weight = args.discriminator_loss_weight
  # default=1.0
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs


  # 架构类型 arch 通常用于指定底层鉴别器模型的结构，如卷积神经网络 (CNN)、残差网络 (ResNet)、
  # 全连接网络 (Fully Connected Network) 等。具体选择何种架构取决于模型设计和任务需求。
  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
  }
  discriminator = PatchDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


# 首先，根据训练集相关参数构建训练集 train_dset，并统计训练集中的图像数量和对象总数。
# 然后，根据验证集相关参数构建验证集 val_dset。
# 最后，确保训练集和验证集使用相同的词汇表，并返回词汇表和训练集、验证集对象。
def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    'include_relationships': args.include_relationships,
  }
  train_dset = CocoSceneGraphDataset(**dset_kwargs)
  # 统计数据集中包含的对象总数。
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  dset_kwargs['max_samples'] = args.num_val_samples
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  # 确保训练集和验证集使用相同的词汇表
  assert train_dset.vocab == val_dset.vocab
  # json.dumps(train_dset.vocab) 将训练集的词汇表转换为 JSON 格式的字符串。
  # json.loads() 函数将该 JSON 字符串再次解析为 Python 对象，从而创建了词汇表的副本。
  vocab = json.loads(json.dumps(train_dset.vocab))

  return vocab, train_dset, val_dset


def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = VgSceneGraphDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = VgSceneGraphDataset(**dset_kwargs)
  
  return vocab, train_dset, val_dset

# vocab是一个词汇表对象，用于存储模型训练过程中出现的所有单词或词汇。
# 在图像生成任务中，词汇表通常包含了与图像内容相关的单词，如物体类别、属性、关系等。
# collate_fn 函数的具体实现根据数据集和模型的要求而定。
# 它的作用是确保数据加载器返回的每个批量都符合模型的输入要求，并且能够高效地进行并行计算。
def build_loaders(args):
  if args.dataset == 'vg':
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn
  elif args.dataset == 'coco':
    vocab, train_dset, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader

# 这段代码是一个评估模型性能的过程，根据给定的数据集（loader），使用训练好的模型进行预测，并计算模型的损失和其他评估指标。
def check_model(args, t, loader, model):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  # 样本数
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      # 通过 triples[:, 1] 可以获取所有三元组中的谓词，将其存储在 predicates 变量中。
      predicates = triples[:, 1] 

      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

      skip_pixel_loss = False
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores)

      total_iou += jaccard(boxes_pred, boxes)
      # 计算预测便边框的总数
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      # 已经验证的样本总数
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break

    samples = {}
    samples['gt_img'] = imgs

    # 已知边界框、已知分割掩码时模型的预测结果
    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
    samples['gt_box_gt_mask'] = model_out[0]
    # 已知边界框时，掩码的预测结果
    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
    samples['gt_box_pred_mask'] = model_out[0]
    # 没有任何先验条件时，模型的预测结果
    model_out = model(objs, triples, obj_to_img)
    samples['pred_box_pred_mask'] = model_out[0]

    # imagenet_deprocess_batch 函数对每个值进行后处理，将其还原为原始图像的格式。
    for k, v in samples.items():
      samples[k] = imagenet_deprocess_batch(v)

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    # 计算目标检测的avg_iou
    avg_iou = total_iou / total_boxes

    # 将 masks 和 masks_pred 的值复制到可以存储的变量 masks_to_store 和 masks_pred_to_store 中
    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(), 
    'masks_gt': masks_to_store,
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
    'masks_pred': masks_pred_to_store
  }
  out = [mean_losses, samples, batch_data, avg_iou]

  return tuple(out)


def calculate_model_losses(args, skip_pixel_loss, model, img, img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates, predicate_scores):
  total_loss = torch.zeros(1).to(img)
  losses = {}
  # losses：一个字典，包含了每个损失的具体数值。

  l1_pixel_weight = args.l1_pixel_loss_weight
  if skip_pixel_loss:
    l1_pixel_weight = 0
  # （1）图像重建损失
  l1_pixel_loss = F.l1_loss(img_pred, img)
  # add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)
  # （2）边界框回归损失
  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)
  # （3）计算谓词预测损失（predicate prediction loss）。
  if args.predicate_pred_loss_weight > 0:
    loss_predicate = F.cross_entropy(predicate_scores, predicates)
    total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                          args.predicate_pred_loss_weight)

  # （4）计算mask掩码损失
  if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          args.mask_loss_weight)
  # 返回总的损失，和每个损失的具体值
  return total_loss, losses


def main(args):
  print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  # 创建loaders
  vocab, train_loader, val_loader = build_loaders(args)
  # print("vocab：",vocab)
  # vocab由以下四个字典组成：object_name_to_idx、pred_name_to_idx、object_idx_to_name、pred_idx_to_name
  # print("train_loader:")
  # print("val_loader:",val_loader)
  # 创建模型
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  # modal_kwargs:'image_size': (64, 64), 'embedding_dim': 128, 'gconv_dim': 128, 'gconv_hidden_dim': 512,
  # 'gconv_num_layers': 5, 'mlp_normalization': 'none', 'refinement_dims': (1024, 512, 256, 128, 64),
  # 'normalization': 'batch', 'activation': 'leakyrelu-0.2', 'mask_size': 16, 'layout_noise_dim': 32
  # print(model)

  # 创建优化器
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  # obj鉴别器、img鉴别器
  obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
  img_discriminator, d_img_kwargs = build_img_discriminator(args, vocab)
  # 生成器和鉴别器的损失
  gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

  # 如果有obj鉴别器，训练梯度更新，优化器更新。
  if obj_discriminator is not None:
    # 设置对象鉴别器的数据类型为 float_dtype，为了与其他模型组件的数据类型保持一致。
    obj_discriminator.type(float_dtype)
    # 将对象鉴别器设置为训练模式，这意味着在进行反向传播和参数更新时会计算梯度
    obj_discriminator.train()
    # print(obj_discriminator)
    # 使用Adam优化器，基于对象鉴别器的参数和学习率 args.learning_rate 初始化优化器 optimizer_d_obj。
    optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                       lr=args.learning_rate)
  # 如果有obj鉴别器，训练梯度更新，优化器更新。
  if img_discriminator is not None:
    img_discriminator.type(float_dtype)
    img_discriminator.train()
    # print(img_discriminator)
    optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(),
                                       lr=args.learning_rate)

  # 从之前保存的检查点中恢复模型和优化器的状态，继续进行训练，避免重新训练或丢失之前的训练进度。
  restore_path = None
  if args.restore_from_checkpoint:
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
  if restore_path is not None and os.path.isfile(restore_path):
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    if obj_discriminator is not None:
      obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
      optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

    if img_discriminator is not None:
      img_discriminator.load_state_dict(checkpoint['d_img_state'])
      optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

    t = checkpoint['counters']['t']
    if 0 <= args.eval_mode_after <= t:
      model.eval()
    else:
      model.train()
    epoch = checkpoint['counters']['epoch']

  #   重新训练模型，给出初始参数
  else:
    t, epoch = 0, 0
    checkpoint = {
      'args': args.__dict__,
      'vocab': vocab,
      'model_kwargs': model_kwargs,
      'd_obj_kwargs': d_obj_kwargs,
      'd_img_kwargs': d_img_kwargs,
      'losses_ts': [],
      'losses': defaultdict(list),
      'd_losses': defaultdict(list),
      'checkpoint_ts': [],
      'train_batch_data': [], 
      'train_samples': [],
      'train_iou': [],
      'val_batch_data': [], 
      'val_samples': [],
      'val_losses': defaultdict(list),
      'val_iou': [], 
      'norm_d': [], 
      'norm_g': [],
      'counters': {
        't': None,
        'epoch': None,
      },
      'model_state': None, 'model_best_state': None, 'optim_state': None,
      'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
      'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
      'best_t': [],
    }

  while True:
    # 训练结束，退出循环
    if t >= args.num_iterations:
      break
    # 训练伦次+1
    epoch += 1
    print('Starting epoch %d' % epoch)

    # 模型评估
    for batch in train_loader:
      # 达到验证的伦次要求，进行模型验证
      if t == args.eval_mode_after:
        print('switching to eval mode')
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
      # t表示iters
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      # 每个batch
      masks = None
      # batch doesn't contain segmentation mask
      if len(batch) == 6:
        # obj_to_img,表示每个对象属于哪个图像，triple_to_img表示每个三元组属于哪个图像
        # obj_to_img 是一个长为 O（对象数量）的列表，每个元素表示对应对象属于哪个图像
        # triple_to_img 是一个长为 T（三元组数量）的列表，每个元素表示对应三元组属于哪个图像。
        # imgs:(32,3,64,64)
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      # batch contains segmentation mask
      elif len(batch) == 7:
        # triples形如：tensor([  0,   5,   5],...


        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      else:
        assert False

      # lsc add 可以正常在matplotlib显示
      # img = torchvision.utils.make_grid(imgs)
      # img = img.to('cpu').numpy().transpose(1,2,0)
      # std = [0.5,0.5,0.5]
      # mean = [0.5,0.5,0.5]
      # img = img*mean+std
      # plt.imshow(img)
      from torchvision.utils import save_image
      save_image(imgs.data, 'imgs.png')

      predicates = triples[:, 1]

      # 记录前向传播的时间，并进行模型预测。
      with timeit('forward', args.timing):
        model_boxes = boxes
        model_masks = masks
        # 模型输出
        model_out = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=model_masks)
        # 依次赋给各预测值
        imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
      # 记录计算损失的时间，并计算损失
      with timeit('loss', args.timing):
        # Skip the pixel loss if using GT boxes
        # 如果model_boxes为None，跳过像素损失的计算。
        skip_pixel_loss = (model_boxes is None)
        # 此处的loss包括：（1）图像重建损失；（2）目标检测损失；（3）mask掩码损失；（4）谓词损失；
        total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores)
      # 计算obj鉴别器相关的损失，包括鉴别器:ac_loss和生成器:g_gan_obj_loss(scores_fake)，并加在total_loss上
      if obj_discriminator is not None:
        # 评估生成的对象（或图像）的真实性，并返回分数（scores_fake）以及一个关于对象鉴别器损失的值（ac_loss）。
        scores_fake, ac_loss = obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
        total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                              args.ac_loss_weight)
        weight = args.discriminator_loss_weight * args.d_obj_weight
        # 将对象鉴别器的生成损失gan_g_loss(scores_fake)
        total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                              'g_gan_obj_loss', weight)

      # 计算img鉴别器相关的损失，包括生成器:g_gan_obj_loss(scores_fake)，并加在total_loss上
      if img_discriminator is not None:
        scores_fake = img_discriminator(imgs_pred)
        weight = args.discriminator_loss_weight * args.d_img_weight
        total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                              'g_gan_img_loss', weight)

      losses['total_loss'] = total_loss.item()
      # 检查计算得到的总损失值是否为有限值。如果总损失值为 NaN，则打印警告信息并跳过反向传播（不执行梯度更新）。
      if not math.isfinite(losses['total_loss']):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      # 此时模型的全部损失值计算完毕，开始进行梯度清零，反响传播，参数优化
      optimizer.zero_grad()
      with timeit('backward', args.timing):
        total_loss.backward()
      optimizer.step()

      total_loss_d = None
      ac_loss_real = None
      ac_loss_fake = None
      d_losses = {}

      # 计算对象鉴别器的三种不同类型的损失：
      # 即GAN损失（用于区分真实和生成数据）、真实性损失（ac_loss_real,用于衡量对真实数据的判别能力）、生成性损失（ac_loss_fake,用于衡量对生成数据的判别能力）。
      # 通过计算生成的假数据（imgs_pred）和真实数据（imgs）的得分，可以计算出对象鉴别器的损失（ac_loss_real和ac_loss_fake）
      # 以及GAN损失（d_obj_gan_loss）。通过训练对象鉴别器，可以使其更好地区分生成的对象和真实的对象。
      if obj_discriminator is not None:
        d_obj_losses = LossManager()
        imgs_fake = imgs_pred.detach()
        scores_fake, ac_loss_fake = obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
        scores_real, ac_loss_real = obj_discriminator(imgs, objs, boxes, obj_to_img)

        # GAN损失（用于区分真实和生成数据）
        d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
        d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
        # 真实性损失
        d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
        # 生成性损失
        d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

        optimizer_d_obj.zero_grad()
        d_obj_losses.total_loss.backward()
        optimizer_d_obj.step()

      # 训练图像鉴别器，使其能够更好地区分生成的图像和真实的图像。
      if img_discriminator is not None:
        d_img_losses = LossManager()
        imgs_fake = imgs_pred.detach()
        scores_fake = img_discriminator(imgs_fake)
        scores_real = img_discriminator(imgs)

        # 它计算了图像鉴别器的GAN损失，并通过优化器的梯度更新来改善图像鉴别器的性能。
        d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
        d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
        
        optimizer_d_img.zero_grad()
        d_img_losses.total_loss.backward()
        optimizer_d_img.step()

      # 每个print_every迭代步骤后打印损失，并将损失值保存在checkpoint的losses字典中。
      if t % args.print_every == 0:
        print('t = %d / %d' % (t, args.num_iterations))
        # 首先，它遍历生成器的损失值（losses字典），打印每个损失的名称和对应的值，并将值添加到checkpoint的losses字典中。
        for name, val in losses.items():
          print(' G [%s]: %.4f' % (name, val))
          checkpoint['losses'][name].append(val)
        checkpoint['losses_ts'].append(t)

        # 然后，如果目标判别器存在，它遍历目标判别器的损失值（d_obj_losses字典），
        # 打印每个损失的名称和对应的值，并将值添加到checkpoint的d_losses字典中。
        if obj_discriminator is not None:
          for name, val in d_obj_losses.items():
            print(' D_obj [%s]: %.4f' % (name, val))
            checkpoint['d_losses'][name].append(val)

        # 最后，如果图像判别器存在，它遍历图像判别器的损失值（d_img_losses字典），
        # 打印每个损失的名称和对应的值，并将值添加到checkpoint的d_losses字典中。
        if img_discriminator is not None:
          for name, val in d_img_losses.items():
            print(' D_img [%s]: %.4f' % (name, val))
            checkpoint['d_losses'][name].append(val)

      # 模型参数文件的
      # if t % args.checkpoint_every == 0:
      if t % 100 == 0:
        print('checking on train')
        # check_model 函数，该函数会对训练集进行评估，并返回训练集上的结果，
        # 包括损失（t_losses）、样本数（t_samples）、批次数据（t_batch_data）以及目标检测的平均IoU（t_avg_iou）等信息。
        train_results = check_model(args, t, train_loader, model)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results

        # 将批次数据、样本数、迭代步骤、目标检测的平均IoU添加到checkpoint字典的train_batch_data键下，用于记录训练集上每个检查点的批次数据。
        checkpoint['train_batch_data'].append(t_batch_data)
        checkpoint['train_samples'].append(t_samples)
        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_iou'].append(t_avg_iou)

        print('checking on val')
        val_results = check_model(args, t, val_loader, model)
        val_losses, val_samples, val_batch_data, val_avg_iou = val_results
        checkpoint['val_samples'].append(val_samples)
        checkpoint['val_batch_data'].append(val_batch_data)
        checkpoint['val_iou'].append(val_avg_iou)

        # 打印训练集和验证集上的IoU值，以监视模型性能。
        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)

        for k, v in val_losses.items():
          checkpoint['val_losses'][k].append(v)
        #  将当前模型的状态字典（模型参数）保存到 checkpoint 字典的 model_state 键下。
        checkpoint['model_state'] = model.state_dict()

        # 将目标判别器的优化器状态字典保存到 checkpoint 字典的 d_obj_optim_state 键下
        if obj_discriminator is not None:
          checkpoint['d_obj_state'] = obj_discriminator.state_dict()
          checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

        # 将图像判别器的优化器状态字典保存到 checkpoint 字典的 d_img_optim_state 键下。
        if img_discriminator is not None:
          checkpoint['d_img_state'] = img_discriminator.state_dict()
          checkpoint['d_img_optim_state'] = optimizer_d_img.state_dict()

        # 将主模型的优化器状态字典保存到 checkpoint 字典的 optim_state 键下。
        checkpoint['optim_state'] = optimizer.state_dict()
        # 更新迭代计数器信息，包括当前迭代步骤 t 和当前的训练轮数 epoch。
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_with_model.pt' % args.checkpoint_name)
        print('Saving checkpoint to ', checkpoint_path)
        # 保存模型的参数
        torch.save(checkpoint, checkpoint_path)

        # Save another checkpoint without any model or optim state
        # 另外，将一个不包含模型或优化器状态的小型检查点保存为另一个文件，以节省存储空间。
        # 这个小型检查点只包含训练过程中的其他信息，而不包含模型参数。
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_no_model.pt' % args.checkpoint_name)
        key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                         'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                         'd_img_state', 'd_img_optim_state', 'd_img_best_state']
        small_checkpoint = {}
        for k, v in checkpoint.items():
          if k not in key_blacklist:
            small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

