"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
import torch
from PIL import Image
import tqdm

from dataset.dataset_configs import STICKS
from experiment import init_model_from_dir
from tools.model_io import download_model
from tools.utils import get_net_input

from tools.vis_utils import show_projections
from visuals.rotating_shape_video import rotating_3d_video

INPUT_2D_DATA_ROOT = './data/2D_data/'
OTUPUT_DATA_ROOT = 'data/3D_data'


def run_demo():

    output = []
    

    model_dir = './data/exps/c3dpo/cat'

    model, _ = init_model_from_dir(model_dir)
    model.eval()

    print('pose estimating ...')
    for batch in tqdm.tqdm(get_test_h36m_sample()):
        net_input = get_net_input(batch)
        preds = model(**net_input)
        kp_pred_3d = preds['shape_image_coord'][0]
        
        output.append(kp_pred_3d.to('cpu').detach().tolist().copy())

    f = open(os.path.join(OTUPUT_DATA_ROOT, 'yarou_cat.json'), 'w')
    json.dump(output, f)


def get_test_h36m_sample():

    data_2d_names = os.listdir(INPUT_2D_DATA_ROOT)
    print('names: ', data_2d_names)
    json_open_2d = open(os.path.join(INPUT_2D_DATA_ROOT, data_2d_names[0]), 'r')
    json_load_2d = json.load(json_open_2d)
    output_json = []
    print('file loading ...')
    for data in tqdm.tqdm(json_load_2d):
        kp_loc, kp_vis = [torch.FloatTensor(a) for a in data.values()]
        d = {'kp_loc': kp_loc[None], 'kp_vis': kp_vis[None]}
        output_json.append(d)

    return output_json


if __name__ == '__main__':
    run_demo()
