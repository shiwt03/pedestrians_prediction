# 基于 LSTM 的行人轨迹预测

### 依赖

pytorch 1.12.1 + cu11.6

mmcv-full 1.6.1

mmtrack 0.14.0


### 数据集准备
#### MOT20数据集
下载链接 <https://motchallenge.net/>
```
pedestrians_prediction
├── data
|   ├── MOT20
|   |   ├── train
|   |   ├── test
```

### 训练

```
python LSTM/train.py
```

### 推理及可视化
```
python LSTM/LSTM_inference.py path/to/configfile.py path/to/LSTM_checkpoint.pth --output mot.mp4 --checkpoint path\to\tracking_model_checkpoint.pth
```
另外可选参数``--loadresult``读取存储的跟踪结果