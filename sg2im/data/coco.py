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

import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

from .utils import imagenet_preprocess, Resize


class CocoSceneGraphDataset(Dataset):
  def __init__(self, image_dir, instances_json, stuff_json=None,
               stuff_only=True, image_size=(64, 64), mask_size=16,
               normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
    """
    A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
    them to scene graphs on the fly.

    Inputs:
    - image_dir: Path to a directory where images are held
    - instances_json: Path to a JSON file giving COCO annotations
    - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
    - stuff_only: (optional, default True) If True then only iterate over
      images which appear in stuff_json; if False then iterate over all images
      in instances_json.
    - image_size: Size (H, W) at which to load images. Default (64, 64).
    - mask_size: Size M for object segmentation masks; default 16.
    - normalize_image: If True then normalize images by subtracting ImageNet
      mean pixel and dividing by ImageNet std pixel.
    - max_samples: If None use all images. Other wise only use images in the
      range [0, max_samples). Default None.
    - include_relationships: If True then include spatial relationships; if
      False then only include the trivial __in_image__ relationship.
    - min_object_size: Ignore objects whose bounding box takes up less than
      this fraction of the image.
    - min_objects_per_image: Ignore images which have fewer than this many
      object annotations.
    - max_objects_per_image: Ignore images which have more than this many
      object annotations.
    - include_other: If True, include COCO-Stuff annotations which have category
      "other". Default is False, because I found that these were really noisy
      and pretty much impossible for the system to model.
    - instance_whitelist: None means use all instance categories. Otherwise a
      list giving a whitelist of instance category names to use.
    - stuff_whitelist: None means use all stuff categories. Otherwise a list
      giving a whitelist of stuff category names to use.
    """
    super(Dataset, self).__init__()

    if stuff_only and stuff_json is None:
      print('WARNING: Got stuff_only=True but stuff_json=None.')
      print('Falling back to stuff_only=False.')

    self.image_dir = image_dir
    self.mask_size = mask_size
    self.max_samples = max_samples
    self.normalize_images = normalize_images
    self.include_relationships = include_relationships
    self.set_image_size(image_size)

    # 从分割实例的json文件中读取数据
    with open(instances_json, 'r') as f:
      instances_data = json.load(f)
      # 包括area、bbox、category_id,id,image_id、iscrowd,segmentation
    stuff_data = None
    if stuff_json is not None and stuff_json != '':
      with open(stuff_json, 'r') as f:
        stuff_data = json.load(f)  #[480.640]
        # 除了包括area、bbox、category_id, id, image_id、iscrowd, segmentation，还包括size[480,640]

    self.image_ids = []
    self.image_id_to_filename = {}
    self.image_id_to_size = {}

    # （1）在instances_data['images']里面拿到图像相关的信息
    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      self.image_ids.append(image_id)
      self.image_id_to_filename[image_id] = filename
      self.image_id_to_size[image_id] = (width, height)

    self.vocab = {
      'object_name_to_idx': {},
      'pred_name_to_idx': {},
    }
    object_idx_to_name = {}
    # {1："person",2:"bicycle",3:"car",}
    all_instance_categories = []

    # （2）在instances_data['categories']里面拿到每个对象类别的id、name
    for category_data in instances_data['categories']:
      category_id = category_data['id']
      category_name = category_data['name']
      all_instance_categories.append(category_name)
      object_idx_to_name[category_id] = category_name
      # 并把对象类别的idx和对象名字对上
      self.vocab['object_name_to_idx'][category_name] = category_id

    # 对于stuff里面的数据也是类似操作
    all_stuff_categories = []
    if stuff_data:
      for category_data in stuff_data['categories']:
        category_name = category_data['name']
        category_id = category_data['id']
        all_stuff_categories.append(category_name)
        object_idx_to_name[category_id] = category_name
        self.vocab['object_name_to_idx'][category_name] = category_id

    # 这段代码的目的是确定哪些类别应该包括在白名单中
    # 白名单中的类别通常是用户根据任务需求选择的，以便进行特定类别的分析或处理。
    if instance_whitelist is None:
      instance_whitelist = all_instance_categories
    if stuff_whitelist is None:
      stuff_whitelist = all_stuff_categories
    category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

    # Add object data from instances
    # （3）将从instances_data['annotations']中提取的对象数据添加到一个名为image_id_to_objects的字典中。
    self.image_id_to_objects = defaultdict(list)
    # object_data:{'segmentation': {'counts': [97214, 1, 425, 4, 422, 6, 420, 9, 417, 12, 415, 13,
    # ..., 4, 2, 12, 397, 2, 19, 8, 421, 4, 6531], 'size': [427, 640]}, 'area': 3489,
    # 'iscrowd': 1, 'image_id': 95999, 'bbox': [227, 260, 397, 82],
    # 'category_id': 1, 'id': 900100095999}
    for object_data in instances_data['annotations']:
      image_id = object_data['image_id']
      _, _, w, h = object_data['bbox']
      W, H = self.image_id_to_size[image_id]
      # 边界框的面积占整个图像面积的比例
      box_area = (w * h) / (W * H)
      # 边界框的面积比例大于min_object_size，才保留边界框，这通常用于过滤掉面积较小的对象。
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data['category_id']]
      # 对象的类别名称在白名单，才保留对象类别，否则过滤特定的对象
      category_ok = object_name in category_whitelist
      other_ok = object_name != 'other' or include_other
      if box_ok and category_ok and other_ok:
        self.image_id_to_objects[image_id].append(object_data)
        # print("image_id_to_objects:",self.image_id_to_objects)
    #     558840:[{'segmentation':[[...]],'area':...,'iscrowed':0,'image_id':558840,
    #     'bbox':[x,x,x,x],''category_id:58,'id':156}]

    # Add object data from stuff
    if stuff_data:
      image_ids_with_stuff = set()
      for object_data in stuff_data['annotations']:
        image_id = object_data['image_id']
        image_ids_with_stuff.add(image_id)
        _, _, w, h = object_data['bbox']
        W, H = self.image_id_to_size[image_id]
        box_area = (w * h) / (W * H)
        box_ok = box_area > min_object_size
        object_name = object_idx_to_name[object_data['category_id']]
        category_ok = object_name in category_whitelist
        other_ok = object_name != 'other' or include_other
        if box_ok and category_ok and other_ok:
          self.image_id_to_objects[image_id].append(object_data)

      #  仅保留包含“stuff”（通常是非物体类别，例如背景、地面等）的图像数据，并将其他图像数据从数据集中删除。
      if stuff_only:
        new_image_ids = []
        for image_id in self.image_ids:
          if image_id in image_ids_with_stuff:
            new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        # 创建一个包含所有图像ID的集合all_image_ids
        all_image_ids = set(self.image_id_to_filename.keys())
        # 计算要从数据集中删除的图像ID集合image_ids_to_remove
        image_ids_to_remove = all_image_ids - image_ids_with_stuff
        # 遍历image_ids_to_remove，并从数据集的相关字典中删除这些图像ID对应的条目，包括图像文件名、图像尺寸和对象信息。
        for image_id in image_ids_to_remove:
          self.image_id_to_filename.pop(image_id, None)
          self.image_id_to_size.pop(image_id, None)
          self.image_id_to_objects.pop(image_id, None)

    # COCO category labels start at 1, so use 0 for __image__
    self.vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    # 根据已有的vocab['object_name_to_idx'],
    # name_to_idx,共183类：{'person': 1, 'bicycle': 2, 'car': 3, ...,'__image__': 0}
    name_to_idx = self.vocab['object_name_to_idx']
    # 使用assert语句来确保物体类别名称到索引的映射没有重复的值
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    # max_object_idx:183
    max_object_idx = max(name_to_idx.values())
    # 创建一个名为idx_to_name的列表，其长度为1 + max_object_idx，184。
    # 在这个列表中，索引从0到max_object_idx对应于不同的物体类别，
    # 而索引为0对应于特殊类别"NONE"，用于表示图像中没有检测到物体的情况。
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    # 使用for循环遍历name_to_idx字典中的每个物体类别名称和对应的索引，
    # 并将类别名称放置到idx_to_name列表的相应索引位置，注意idx_to_name为列表，而不是字典了。
    for name, idx in self.vocab['object_name_to_idx'].items():
      idx_to_name[idx] = name

    # ['__image__', 'person', 'bicycle',..., 'other']
    self.vocab['object_idx_to_name'] = idx_to_name

    # Prune images that have too few or too many objects
    # 从数据集中删除包含过少或过多物体的图像
    new_image_ids = []
    total_objs = 0
    # 计算之后total_objs:877152
    for image_id in self.image_ids:
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
      # 3<= num_objs<=8
      if min_objects_per_image <= num_objs <= max_objects_per_image:
        new_image_ids.append(image_id)
    # image_ids:475546
    self.image_ids = new_image_ids

    #
    self.vocab['pred_idx_to_name'] = [
      '__in_image__',
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding',
    ]
    self.vocab['pred_name_to_idx'] = {}
    for idx, name in enumerate(self.vocab['pred_idx_to_name']):
      self.vocab['pred_name_to_idx'][name] = idx

  # 将图像大小变换为60*60(image_size)，并转换为tensor形式
  def set_image_size(self, image_size):
    print('called set_image_size', image_size)
    transform = [Resize(image_size), T.ToTensor()]
    # 检查是否需要对图像进行标准化
    if self.normalize_images:
      # 在transform列表中添加了一个名为imagenet_preprocess的函数，用于对图像进行标准化处理。
      # 标准化是将图像像素值缩放到特定范围以进行更好的训练。
      transform.append(imagenet_preprocess())
    # 将上述定义的图像变换操作组合成一个transforms.Compose对象，并将其赋值给类属性self.transform
    self.transform = T.Compose(transform)
    self.image_size = image_size

  # 计算数据集的总对象数量
  def total_objects(self):
    total_objs = 0
    for i, image_id in enumerate(self.image_ids):
      if self.max_samples and i >= self.max_samples:
        break
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
    return total_objs

  def __len__(self):
    if self.max_samples is None:
      return len(self.image_ids)
    return min(len(self.image_ids), self.max_samples)


  # 获取数据集中的一个样本
  def __getitem__(self, index):
    """
    Get the pixels of an image, and a random synthetic scene graph for that
    image constructed on-the-fly from its COCO object annotations. We assume
    that the image will have height H, width W, C channels; there will be O
    object annotations, each of which will have both a bounding box and a
    segmentation mask of shape (M, M). There will be T triples in the scene
    graph.

    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (O,)
    - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system
    - masks: LongTensor of shape (O, M, M) giving segmentation masks for
      objects, where 0 is background and 1 is object.
    - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    # 获取数据集中的一个样本，得到其图像id，图像名称，图像路径，图像宽高，并将其转换为rgb格式
    image_id = self.image_ids[index]
    filename = self.image_id_to_filename[image_id]
    image_path = os.path.join(self.image_dir, filename)
    with open(image_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size
    objs, boxes, masks = [], [], []
    for object_data in self.image_id_to_objects[image_id]:
      objs.append(object_data['category_id'])

      # [0.0, 301.0, 640.0, 59.0]
      x, y, w, h = object_data['bbox']
      # 将坐标调整至0-1之间，# tensor([0.0000, 0.0000, 1.0000, 0.8778])
      x0 = x / WW
      y0 = y / HH
      x1 = (x + w) / WW
      y1 = (y + h) / HH
      boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

      # This will give a numpy array of shape (HH, WW)
      # 得到16x16的矩阵，每个位置为1或者0
      # WW，HH :480,640
      mask = seg_to_mask(object_data['segmentation'], WW, HH)

      # Crop the mask according to the bounding box, being careful to
      # ensure that we don't crop a zero-area region
      # mx0,my0=(156,0) mx1,my1=(480,323)
      mx0, mx1 = int(round(x)), int(round(x + w))
      my0, my1 = int(round(y)), int(round(y + h))
      mx1 = max(mx0 + 1, mx1)
      my1 = max(my0 + 1, my1)
      # mx0, mx1, my0, my1分别是裁剪后的掩码的左上角和右下角的像素坐标。
      mask = mask[my0:my1, mx0:mx1]
      # mask维度（323，324）
      # 将裁剪后的掩码调整大小为指定的self.mask_size，并将像素值缩放为0到255之间。
      # mask维度（16，16）并且每个像素点的取值位于0～255之间
      mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                      mode='constant')
      #mask维度依然为（16，16），但每个值为1或0
      mask = torch.from_numpy((mask > 128).astype(np.int64))
      masks.append(mask)

    # Add dummy __image__ object
    # 把__image__作为虚拟的对象。
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    # 虚拟对象的边界框为[0,0,1,1]
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))
    # 虚拟对象的掩码为全为1，（16，16）
    masks.append(torch.ones(self.mask_size, self.mask_size).long())

    # 这种操作通常用于将多个张量合并成一个批次（batch）的张量，以便进行批处理的操作，例如在神经网络训练中同时处理多个样本。
    # 在这个场景中，它用于将一张图像中的多个物体的信息存储在一个张量中，以便后续的处理。
    # （1）将所有对象设置为LongTensor
    objs = torch.LongTensor(objs)
    # （2）将boxes列表中的多个张量按照 dim=0 的维度进行堆叠，也就是将它们垂直堆叠在一起，生成一个新的张量。
    # 如果有8个对象，则：（8，4）
    # tensor([[0.0017, 0.0989, 0.5006, 0.9865],
    #         [0.3455, 0.0488, 0.9994, 0.9858],
    #         [0.0000, 0.0000, 0.5625, 0.0854],
    #         [0.0562, 0.6271, 0.5078, 1.0000],
    #         [0.3047, 0.4354, 0.9500, 0.8479],
    #         [0.0000, 0.0000, 1.0000, 0.7542],
    #         [0.3156, 0.1896, 0.5406, 0.5125],
    #         [0.0000, 0.0000, 1.0000, 1.0000]])
    boxes = torch.stack(boxes, dim=0)
    # （3）将多个mask的信息保存在一起
    masks = torch.stack(masks, dim=0)

    # box具体格式是 [x0, y0, x1, y1]，
    # boxes[:, 2] 表示从 boxes 张量中选择所有行的第三列，即右下角 x 坐标 x1。
    # 因此，boxes[:, 2] - boxes[:, 0] 计算了每个物体边界框的宽度（x1 减去 x0），
    # 而 boxes[:, 3] - boxes[:, 1] 计算了每个物体边界框的高度（y1 减去 y0）。
    # 通过将这两个张量相乘，得到的 box_areas 张量包含了每个物体边界框的面积。
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute centers of all objects
    # 计算所有对象的中心坐标
    obj_centers = []
    _, MH, MW = masks.size()
    for i, obj_idx in enumerate(objs):
      x0, y0, x1, y1 = boxes[i]
      mask = (masks[i] == 1)
      #x0： 开始值，x1：结束值，MW：等分多少份
      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      # 检查掩码中是否有非零元素。如果掩码中没有非零元素，说明对象不存在于图像中
      # （可能是由于掩码被裁剪或其他原因），则将对象的中心坐标设置为边界框的中心坐标。
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      # 如果掩码中有非零元素，计算这些非零元素的均值，以获得对象的中心坐标。
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      obj_centers.append([mean_x, mean_y])
    obj_centers = torch.FloatTensor(obj_centers)

    # Add triples
    triples = []
    # num_objs:8
    num_objs = objs.size(0)
    # 虚拟图像
    __image__ = self.vocab['object_name_to_idx']['__image__']
    # 去掉虚拟对象 ： tensor([0, 1, 2, 3, 4, 5, 6])
    real_objs = []
    if num_objs > 1:
      # nonzero() 方法是用于获取张量中非零元素的索引的方法，返回一个包含非零元素索引的张量。
      # 在这段代码中，它被用于找到objs张量中非零值的索引。
      # squeeze(1) 则是用于移除张量中维度为 1 的维度，它将张量的形状从 (N, 1) 转换为 (N,)，其中 N 表示张量的长度。
      real_objs = (objs != __image__).nonzero().squeeze(1)
    for cur in real_objs:
      # choices列表包含了除了当前物体cur之外的所有真实物体的索引
      # [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4), tensor(5)]
      choices = [obj for obj in real_objs if obj != cur]
      # 当没有其他对象时，或在当前图像中不包含关系时，跳出循环，不再构建关系
      if len(choices) == 0 or not self.include_relationships:
        break
      # 从choices中随机选择一个对象
      other = random.choice(choices)
      # 如果随机数大于0.5,cur作为主语，other作为宾语
      if random.random() > 0.5:
        s, o = cur, other
      else:
        s, o = other, cur

      # Check for inside / surrounding
      # 它获取了两个物体的边界框坐标 sx0, sy0, sx1, sy1 和 ox0, oy0, ox1, oy1，
      sx0, sy0, sx1, sy1 = boxes[s]
      ox0, oy0, ox1, oy1 = boxes[o]
      # 以及它们的中心点坐标差d和极坐标角度theta。
      d = obj_centers[s] - obj_centers[o]
      theta = math.atan2(d[1], d[0])

      # 如果一个物体完全包围另一个物体，则关系谓词为 "surrounding"。
      if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
      #  如果一个物体完全被另一个物体包围，则关系谓词为 "inside"。
      elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
      # 如果一个物体位于另一个物体的左侧，则关系谓词为 "left of"。
      # [135度，正无穷]，[副无穷，-135度]
      elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
        p = 'left of'
      # 上方
      # （-135度，-45度）
      elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = 'above'
      #   （-45度，45度）
      elif -math.pi / 4 <= theta < math.pi / 4:
        p = 'right of'
      #  （45度，135度）
      elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = 'below'
      # p为关系对应的索引
      p = self.vocab['pred_name_to_idx'][p]
      triples.append([s, p, o])
      # [[tensor(4), 1, tensor(0)], [tensor(1), 1, tensor(2)], [tensor(2), 4, tensor(3)],
      #  [tensor(3), 1, tensor(0)], [tensor(0), 2, tensor(4)], [tensor(0), 3, tensor(5)],
      #  [tensor(6), 3, tensor(3)]]

    # Add __in_image__ triples
    # 对象数量
    O = objs.size(0)
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    # 为每个对象添加与整个图像之间的关系！！！
    # 它将在场景图的每个对象上运行，但会排除最后一个对象。
    # 循环从 0 开始，一直到 O - 1，因为最后一个对象是 __image__，表示整个图像。
    for i in range(O - 1):
      # 它为每个对象添加一个三元组，这个三元组表示对象与图像之间的关系。
      triples.append([i, in_image, O - 1])

    # [[tensor(2), 2, tensor(0)], [tensor(0), 5, tensor(1)], [tensor(4), 1, tensor(2)], [tensor(4), 2, tensor(3)], [tensor(4), 3, tensor(0)], [tensor(4), 1, tensor(5)],
    # 关系名称为0，每个对象被整个图像完全包围
    # [0, 0, 6], [1, 0, 6], [2, 0, 6], [3, 0, 6], [4, 0, 6], [5, 0, 6]]
    triples = torch.LongTensor(triples)
    # print("image:",image)
    return image, objs, boxes, masks, triples

# seg是分割信息，可以是多种格式，包括Python列表或COCO数据集中的分割信息对象。
# width和height表示图像的宽度和高度，默认为1.0。
def seg_to_mask(seg, width=1.0, height=1.0):
  """
  Tiny utility for decoding segmentation masks using the pycocotools API.
  """
  # 首先，函数检查seg的类型，如果是列表（list），
  # 则假设seg是由多个分割信息组成的列表，并将其转换为COCO分割格式（Run Length Encoding，RLE）。
  if type(seg) == list:
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
  # 如果seg的counts字段是列表类型，也将其转换为COCO分割格式（RLE）。
  elif type(seg['counts']) == list:
    rle = mask_utils.frPyObjects(seg, height, width)
  # 如果seg本身就是COCO分割格式（RLE），则直接使用它。
  else:
    rle = seg
  # 最后调用mask_utils.decode(rle)来解码COCO分割格式的信息，生成掩码。
  return mask_utils.decode(rle)

# 在使用CocoSceneGraphDataset创建DataLoader时的数据整理函数
# 它的作用是将一个批次（batch）的数据整理成模型可以处理的格式。这个函数返回一个包含以下内容的元组：
def coco_collate_fn(batch):
  """
  Collate function to be used when wrapping CocoSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving object categories
  - boxes: FloatTensor of shape (O, 4)
  - masks: FloatTensor of shape (O, M, M)
  - triples: LongTensor of shape (T, 3) giving triples
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - triple_to_img: LongTensor of shape (T,) mapping triples to images
  """
  # all_imgs (N, C, H, W)
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  for i, (img, objs, boxes, masks, triples) in enumerate(batch):
    all_imgs.append(img[None])
    if objs.dim() == 0 or triples.dim() == 0:
      continue
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    all_masks.append(masks)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out

