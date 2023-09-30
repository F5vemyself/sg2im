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

import argparse, os
import torch


"""
Checkpoints saved by train.py contain not only model parameters but also
optimizer states, losses, a history of generated images, and other statistics.
This information is very useful for development and debugging models, but makes
the saved checkpoints very large. This utility script strips away all extra
information from saved checkpoints, keeping only the saved models.
"""
# 这段脚本是一个实用工具脚本，用于从由train.py脚本生成的保存的检查点中删除额外信息。在训练模型时，
# 检查点通常不仅包含模型参数，还包含优化器状态、损失、生成图像的历史记录以及其他统计信息。
# 这些信息对于开发和调试模型非常有用，但会使保存的检查点变得非常大。这个实用脚本会剥离掉所有额外信息，只保留保存的模型。

parser = argparse.ArgumentParser()
parser.add_argument('--input_checkpoint', default=None)
parser.add_argument('--output_checkpoint', default=None)
parser.add_argument('--input_dir', default=None)
parser.add_argument('--output_dir', default=None)
parser.add_argument('--keep_discriminators', type=int, default=1)


def main(args):
  if args.input_checkpoint is not None:
    handle_checkpoint(args, args.input_checkpoint, args.output_checkpoint)
  if args.input_dir is not None:
    handle_dir(args, args.input_dir, args.output_dir)


def handle_dir(args, input_dir, output_dir):
  for fn in os.listdir(input_dir):
    if not fn.endswith('.pt'):
      continue
    input_path = os.path.join(input_dir, fn)
    output_path = os.path.join(output_dir, fn)
    handle_checkpoint(args, input_path, output_path)


def handle_checkpoint(args, input_path, output_path):
  input_checkpoint = torch.load(input_path)
  keep = ['args', 'model_state', 'model_kwargs']
  if args.keep_discriminators == 1:
    keep += ['d_img_state', 'd_img_kwargs', 'd_obj_state', 'd_obj_kwargs']
  output_checkpoint = {}
  for k, v in input_checkpoint.items():
    if k in keep:
      output_checkpoint[k] = v
  torch.save(output_checkpoint, output_path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

