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
This utility script removes deprecated kwargs in checkpoints.
"""

# 这个实用脚本用于从检查点中删除已弃用的kwargs（关键字参数）。
# 随着代码的更新和演变，某些参数可能被弃用或更改了名称。
# 使用这个脚本可以清除检查点中的已弃用参数，以确保与当前代码版本兼容。
parser = argparse.ArgumentParser()
parser.add_argument('--input_checkpoint', default=None)
parser.add_argument('--input_dir', default=None)


DEPRECATED_KWARGS = {
  'model_kwargs': [
    'vec_noise_dim', 'gconv_mode', 'box_anchor', 'decouple_obj_predictions',
  ],
}


def main(args):
  got_checkpoint = (args.input_checkpoint is not None)
  got_dir = (args.input_dir is not None)
  assert got_checkpoint != got_dir, "Must give exactly one of checkpoint or dir"
  if got_checkpoint:
    handle_checkpoint(args.input_checkpoint)
  elif got_dir:
    handle_dir(args.input_dir)



def handle_dir(dir_path):
  for fn in os.listdir(dir_path):
    if not fn.endswith('.pt'):
      continue
    checkpoint_path = os.path.join(dir_path, fn)
    handle_checkpoint(checkpoint_path)


def handle_checkpoint(checkpoint_path):
  print('Stripping old args from checkpoint "%s"' % checkpoint_path)
  checkpoint = torch.load(checkpoint_path)
  for group, deprecated in DEPRECATED_KWARGS.items():
    assert group in checkpoint
    for k in deprecated:
      if k in checkpoint[group]:
        print('Removing key "%s" from "%s"' % (k, group))
        del checkpoint[group][k]
  torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

