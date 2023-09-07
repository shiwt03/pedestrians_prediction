import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import numpy as np
import torch.nn as nn

import torch
import mmcv

from mmtrack.apis import inference_mot, init_model


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

        out = out[:, 50:, :]

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
            data[obj_id].append(np.array([(bbox[1] + bbox[3]) / 2, (bbox[2] + bbox[4]) / 2], dtype='float32'))
    return data, start_frame


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('weight', help='path_to_LSTM_weight')
    parser.add_argument('--input', help='input video file or folder')
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
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show


    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True


    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    track_model = init_model(args.config, args.checkpoint, device=args.device)

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images

    track_results = []

    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(track_model, img, frame_id=i)

        track_results.append(result['track_bboxes'][0])
        # print(result['track_bboxes'][0].shape)
        # result format: dict, keys: 'track_bboxes'
        #                      values: list, list[0]:array ([id, bbox])
        #                       bbox:[id, left_up_x, left_up_y, right_down_x, right_down_y, confidence_level]
        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        track_model.show_result(
            img,
            result,
            score_thr=args.score_thr,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


    #build LSTM model
    LSTM_model = LSTM(2, 20, 2, 2).cuda()
    if os.path.isfile(args.weight):
        LSTM_model = torch.load(args.weight)
        # state_dict = LSTM_model.state_dict()
        # # print(torch.load(args.weight).keys())
        # checkpoint = torch.load(args.weight)['state_dict']
        # print(checkpoint.keys())
        # # print(checkpoint['state_dict'].keys())
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        # print(pretrained_dict.keys())
        # state_dict.update(pretrained_dict)
        # LSTM_model.load_state_dict(state_dict)
    else:
        print("=> no checkpoint found at '{}'".format(args.weight))
        return

    sequence_data, start_frame = convert_data(track_results)
    input_seq_len = 100
    pred_seq_len = 50
    x_test = []
    y_test = []
    for key, value in sequence_data.items():
        # print(f'key:{key}, len:{len(value)}')
        if len(value) < input_seq_len + pred_seq_len:
            continue
        stt = 0
        x_test.append(np.array(value[stt:stt + input_seq_len]))
        y_test.append(np.array(value[stt + input_seq_len:stt + input_seq_len + pred_seq_len]))

    x_test = torch.from_numpy(np.array(x_test)).cuda()
    y_test = torch.from_numpy(np.array(y_test)).cuda()

    # print(sequence_data)
    print(x_test.shape)

    prediction = LSTM_model(x_test)
    print(prediction)
    print(y_test)


if __name__ == '__main__':
    main()
