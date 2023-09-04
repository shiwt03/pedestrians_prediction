import os
from tqdm import tqdm

"""
data format:

dict:
    dict[id]: list
            list shape:[bboxes]
            bbox: list[2], 2D position of the central point of bbox

"""

mot_root = 'D:\DL_Workspace\pedestrians_prediction\data\MOT20'
scene_id = 3
excluded_type = [6, 11, 13]

if __name__ == '__main__':
    scene_name = 'MOT20-0' + str(scene_id)
    data_path = os.path.join(mot_root, 'train', scene_name, 'gt', 'gt.txt')
    file = open(data_path ,'r')
    lines = file.readlines()
    # print(lines)
    data = dict()
    for line in tqdm(lines):
        item = list(map(int, line.split(',')[:-1]))
        # print(item)
        if item[7] in excluded_type:
            continue
        id = item[1]
        if id not in data.keys():
            data[id] = []
        data[id].append([item[2] + item[4] / 2, item[3] + item[5] / 2])
