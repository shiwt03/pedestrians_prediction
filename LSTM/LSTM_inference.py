import math
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import numpy as np
import torch.nn as nn

import torch
import mmcv
import matplotlib.pyplot as plt

from mmtrack.apis import inference_mot, init_model
from tracking_inference import tracking_inference
from scipy.spatial import distance

from dtw import dtw
from sdtw import SoftDTW
from sdtw.distance import euclidean_distances, SquaredEuclidean
from loss.dilate_loss import dilate_loss

from tqdm import tqdm

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
scene_id = 4


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        # 初始化参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        out, (hn, cn) = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.fc(out)
        return out


def convert_data(track_results):
    """
    Args:
        track_results: list[array], shape:[length]
                      array_shape:[id, bbox]
    Returns:
        tensor shape:[length, batch, num_features]
    """

    data = dict()
    start_frame = dict()
    for frame_id, result in enumerate(track_results):
        for bbox in result:
            obj_id = bbox[0]
            if obj_id not in start_frame.keys():
                start_frame[obj_id] = frame_id
                data[obj_id] = []
            data[obj_id].append(np.array(
                [(bbox[1] + bbox[3]) / 2 / img_sizes[scene_id][0], (bbox[2] + bbox[4]) / 2 / img_sizes[scene_id][1]],
                dtype='float32'))
    return data, start_frame


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('weight', help='path_to_LSTM_weight')
    parser.add_argument('--input', default='data/MOT20/test/MOT20-0' + str(scene_id) + '/img1',
                        help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', default=24, help='FPS of the output video')
    parser.add_argument('--loadresult', action='store_true')
    args = parser.parse_args()

    if args.loadresult:
        track_results = np.load('tracking_result.npy', allow_pickle=True)
    else:
        track_results = tracking_inference(args)
        np.save('tracking_result', track_results)

    # build LSTM model
    if os.path.isfile(args.weight):
        LSTM_model = torch.load(args.weight)
    else:
        print("=> no checkpoint found at '{}'".format(args.weight))
        return

    sequence_data, start_frame_dic = convert_data(track_results)
    # print(start_frame_dic)
    input_seq_len = 100
    pred_seq_len = 100
    seq_interval = 5
    x_test = []
    y_test = []
    start_frames = []

    for key, value in sequence_data.items():
        # print(f'key:{key}, len:{len(value)}')
        if len(value) < input_seq_len + pred_seq_len:
            continue
        stt = 0
        x_test.append(np.array(value[stt:stt + input_seq_len:seq_interval]))
        y_test.append(np.array(value[stt + input_seq_len:stt + input_seq_len + pred_seq_len:seq_interval]))
        start_frames.append(start_frame_dic[key])

    x_test = torch.from_numpy(np.array(x_test)).cuda()
    y_test = np.array(y_test)

    # print(x_test.shape)

    t_iters = tqdm(range(pred_seq_len // seq_interval))
    predictions = []
    x = x_test

    # batch seqence_len 2
    # batch 2
    for iter in t_iters:
        # print(x.shape)
        y = LSTM_model(x)
        # print(y.shape)
        predictions.append(y.cpu().detach().numpy())
        x = torch.cat([x[:, 1:, :], y.unsqueeze(1)], dim=1)
        # print(x.shape)

    prediction = np.array(predictions).transpose(1, 0, 2)

    # print(prediction.shape)
    # prediction = LSTM_model(x_test)

    # print(prediction)
    # print(y_test)

    # prediction = prediction.cpu().detach().numpy()
    # y_test = y_test.cpu().detach().numpy()

    # for ele in range(len(prediction)):
    #     pre_x = []
    #     pre_y = []
    #     real_x = []
    #     real_y = []
    #     for frame in range(pred_seq_len // seq_interval):
    #         pre_x.append(prediction[ele][frame][0])
    #         pre_y.append(prediction[ele][frame][1])
    #         real_x.append(y_test[ele][frame][0])
    #         real_y.append(y_test[ele][frame][1])
    #     plt.plot(pre_x, pre_y)
    #     plt.plot(real_x, real_y)
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    #     plt.savefig(f'prediction_figs/{str(ele)}.png')
    #     plt.clf()

    img_names = os.listdir(args.input)

    avg_dis = []
    max_dis = []
    dtw_dis = []
    softdtw_dis = []
    dilate_dis = []
    t_prediction = tqdm(range(len(prediction)))
    for ele in t_prediction:
        show_frame_id = start_frames[ele] + input_seq_len
        # print(show_frame_id)
        img = cv2.imread(os.path.join(args.input, img_names[show_frame_id]))
        distances = []
        pred_dots = []
        label_dots = []

        for frame in range(input_seq_len // seq_interval - 1):
            cv2.line(img, (
                int(x_test[ele][frame][0] * img_sizes[scene_id][0]),
                int(x_test[ele][frame][1] * img_sizes[scene_id][1])),
                     (int(x_test[ele][frame + 1][0] * img_sizes[scene_id][0]),
                      int(x_test[ele][frame + 1][1] * img_sizes[scene_id][1])),
                     (170, 255, 127), 3)
        for frame in range(pred_seq_len // seq_interval - 1):
            cv2.line(img, (
                int(prediction[ele][frame][0] * img_sizes[scene_id][0]),
                int(prediction[ele][frame][1] * img_sizes[scene_id][1])),
                     (int(prediction[ele][frame + 1][0] * img_sizes[scene_id][0]),
                      int(prediction[ele][frame + 1][1] * img_sizes[scene_id][1])),
                     (180, 105, 255), 3)
            cv2.line(img,
                     (int(y_test[ele][frame][0] * img_sizes[scene_id][0]),
                      int(y_test[ele][frame][1] * img_sizes[scene_id][1])),
                     (int(y_test[ele][frame + 1][0] * img_sizes[scene_id][0]),
                      int(y_test[ele][frame + 1][1] * img_sizes[scene_id][1])),
                     (0, 255, 255), 3)

            pred_dots.append([int(prediction[ele][frame][0] * img_sizes[scene_id][0]),
                              int(prediction[ele][frame][1] * img_sizes[scene_id][1])])
            label_dots.append([int(y_test[ele][frame][0] * img_sizes[scene_id][0]),
                              int(y_test[ele][frame][1] * img_sizes[scene_id][1])])
            distances.append(distance.euclidean(
                (pred_dots[-1][0], pred_dots[-1][1]),
                (label_dots[-1][0], label_dots[-1][1])
            ))



        pred_dots.append(np.array([int(prediction[-1][frame][0] * img_sizes[scene_id][0]),
                          int(prediction[-1][frame][1] * img_sizes[scene_id][1])]))
        label_dots.append(np.array([int(y_test[-1][frame][0] * img_sizes[scene_id][0]),
                           int(y_test[-1][frame][1] * img_sizes[scene_id][1])]))

        distances.append(distance.euclidean(
            (pred_dots[-1][0], pred_dots[-1][1]),
            (label_dots[-1][0], label_dots[-1][1])
        ))
        dtw_value, mat, _, _ = dtw(np.array(pred_dots), np.array(label_dots), dist=lambda x, y:distance.euclidean(x, y))
        dtw_dis.append(dtw_value / (pred_seq_len // seq_interval))

        D = euclidean_distances(np.array(pred_dots), np.array(label_dots))
        sdtw = SoftDTW(D, gamma=1.0)
        softdtw_dis.append(sdtw.compute() / (pred_seq_len // seq_interval))

        # print(np.expand_dims(np.array(label_dots), axis=0).shape)
        dilate_pred = torch.from_numpy(np.expand_dims(np.array(pred_dots), axis=0))
        dilate_label = torch.from_numpy(np.expand_dims(np.array(label_dots), axis=0))
        dilate = dilate_loss(dilate_pred, dilate_label)[0]
        dilate_dis.append(dilate / (pred_seq_len // seq_interval))

        distances = np.array(distances)


        avg_dis.append(distances.mean())
        cv2.imwrite(f'prediction_figs/{str(ele)}.png', img)

    print(np.array(avg_dis).mean())
    print(np.array(dtw_dis).mean())
    print(np.array(softdtw_dis).mean())
    print(np.array(dilate_dis).mean())


if __name__ == '__main__':
    main()
