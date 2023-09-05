import os
from tqdm import tqdm
import torch
import numpy as np

"""
data format:

dict:
    dict[id]: list
            list shape:[bboxes]
            bbox: list[2], 2D position of the central point of bbox

"""

excluded_type = [6, 11, 13]
img_sizes = [
    [1920, 1080],
    [1920, 1080],
    [1172, 880],
    [1545, 1080],
    [1654, 1080],
    [1920, 734],
    [1920, 1080],
    [1920, 734]
]
def getdata(mot_root, scene_id):
    scene_name = 'MOT20-0' + str(scene_id)
    data_path = os.path.join(mot_root, 'train', scene_name, 'gt', 'gt.txt')
    file = open(data_path, 'r')
    lines = file.readlines()
    # print(lines)
    img_size = img_sizes[scene_id]
    data = dict()
    for line in tqdm(lines):
        item = list(map(int, line.split(',')[:-1]))
        # print(item)
        if item[7] in excluded_type:
            continue
        id = item[1]
        if id not in data.keys():
            data[id] = []
        data[id].append(np.array( [ (item[2] + item[4] / 2) / img_size[0], (item[3] + item[5] / 2) / img_size[1] ], dtype='float32'))
    return data

if __name__ == '__main__':
    mot_root = 'D:\DL_Workspace\pedestrians_prediction\data\MOT20'
    scene_id = 5
    scene_name = 'MOT20-0' + str(scene_id)
    img_size = img_sizes[scene_id]
    data_path = os.path.join(mot_root, 'train', scene_name, 'gt', 'gt.txt')
    file = open(data_path, 'r')
    lines = file.readlines()
    # print(lines)
    print(f'loading annotation from {scene_name}:')
    data = dict()
    for line in tqdm(lines):
        item = list(map(int, line.split(',')[:-1]))
        # print(item)
        if item[7] in excluded_type:
            continue
        id = item[1]
        if id not in data.keys():
            data[id] = []
        data[id].append( [(item[2] + item[4] / 2) / img_size[0], (item[3] + item[5] / 2)] / img_size[1] )
    for key, value in data.items():
        print(f'key:{key}, len:{len(value)}')
