import os
import json
import tqdm

import numpy as np

# 2D　のjsonファイルを読み込み、前処理をして、2D_dataにjsonファイルを生成する。

# PRE_2D_DATA_ROOT = '/mnt/c/Users/atori/dev/cat/mmpose/cat_pose_estimation/json_output/kandou_neko_2020_1-2021_11'
PRE_2D_DATA_ROOT = '/home/atori/dev/c3dpo/c3dpo_nrsfm/data/predata/kagoneko_2022_1-2021_11'
OTUPUT_DATA_ROOT = './data/2D_data'

# 信頼度の低いデータをスキップする。
def is_low_confidence(pose: list) -> bool:
    """
    Remove low confidence data
    """
    
    if pose['bbox']['score'] < 0.8:
        return True

        # if pose['keypoints']['Nose']['score'] < 0.8:
        #     return True

        # for joint in pose['keypoints']:
        #     if pose['keypoints'][joint]['score'] < 0.1:
        #         return True

    return False


# bboxで切り抜いて，左下を原点にし，幅×高さ=10000に変換する。
def _convert_bbox_to_relative_coordinate(left: float, top: float, right: float, bottom: float) -> list:
    """
    Convert bbox to relative coordinate
    """
    pre_w = right - left  # 幅
    pre_h = bottom - top  # 高さ
    ratio = pre_h / pre_w  # 高さと幅の比
    r_ratio = np.sqrt(ratio)  # 高さと幅の比のルート
    width = 100 / r_ratio  # 調整後の幅
    height = width * r_ratio  # 調整後の高さ
    scale = width / pre_w  # 調整後と調整前の比
    slide_x = left
    slide_y = top
    res = [width, height, scale, slide_x, slide_y]
    return res


def json_to_list(frame: list) -> list:
    """
    Json to list
    """
    res = []
    for pose in frame:
        # 信頼度の低いデータをスキップ
        if is_low_confidence(pose):
            continue

        loc_x = []
        loc_y = []
        vis = []
        w, h, s, x, y = _convert_bbox_to_relative_coordinate(
            pose['bbox']['left'], pose['bbox']['top'], pose['bbox']['right'], pose['bbox']['bottom'])

        for joint in pose['keypoints']:
            k_x = (pose['keypoints'][joint]['x'] - x) * s  # x座標を変換
            loc_x.append(k_x)
            k_y = h - (pose['keypoints'][joint]['y'] - y) * s  # y座標を変換
            loc_y.append(k_y)
            vis.append(1 if pose['keypoints'][joint]['score'] > 0.3 else 0) # スコアが0.3より大きければ見える関節として扱う。

        res.append({'kp_loc': [loc_x, loc_y], 'kp_vis': vis})

    return res


def main():
    data_2d_names = os.listdir(PRE_2D_DATA_ROOT)
    processed_data_names = os.listdir(OTUPUT_DATA_ROOT)
    data_2d_names = [
        name for name in data_2d_names if name not in processed_data_names]
    if len(data_2d_names) == 0:
        print('No data to process')
        return

    output_json = []
    file_num = 1

    for file_name in tqdm.tqdm(data_2d_names):
        json_open_2d = open(os.path.join(PRE_2D_DATA_ROOT, file_name), 'r', encoding='utf-8')
        
        try:        
            json_load_2d = json.load(json_open_2d)
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue


        for frame in json_load_2d['data']:
            # 空のデータを削除スキップ
            if frame == []:
                continue

            data = json_to_list(frame)
            if len(data) == 0:
                continue
            output_json += data
        
        if len(output_json) > 300000:
            print('\nsaving ... ', f'kago_{file_num}_cat_train.json')
            f = open(os.path.join(OTUPUT_DATA_ROOT, f'kago_{file_num}_cat_train.json'), 'w')
            json.dump(output_json, f)
            output_json = []
            file_num += 1

    f = open(os.path.join(OTUPUT_DATA_ROOT, f'kago_{file_num}_cat_train.json'), 'w')
    json.dump(output_json, f)


if __name__ == '__main__':
    main()