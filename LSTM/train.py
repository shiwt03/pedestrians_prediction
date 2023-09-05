import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from get_data import getdata
from loss.dilate_loss import dilate_loss

mot_root = 'D:\DL_Workspace\pedestrians_prediction\data\MOT20'
scene_id = 3


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


if __name__ == '__main__':
    model = LSTM(2, 20, 2, 2)
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_data = getdata(mot_root, scene_id)
    input_seq_len = 100
    pred_seq_len = 50
    x_train = []
    y_train = []
    for key, value in train_data.items():
        if len(value) < input_seq_len + pred_seq_len:
            continue
        x_train.append(np.array(value[:input_seq_len]))
        y_train.append(np.array(value[input_seq_len:(input_seq_len + pred_seq_len)]))

    x_train = torch.from_numpy(np.array(x_train)).cuda()
    y_train = torch.from_numpy(np.array(y_train)).cuda()

    # print(x_train.shape)
    # print(y_train.shape)

    # x_train = torch.randn(100, 50, 10)
    # y_train = torch.randn(100, 5)

    epochs = 20000

    t_epoch = tqdm(range(epochs))
    for epoch in t_epoch:
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        # loss = dilate_loss(y_pred[:, :, 0], y_train[:, :, 0]) + dilate_loss(y_pred[:, :, 1], y_train[:, :, 1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        t_epoch.set_postfix(loss = loss.item())