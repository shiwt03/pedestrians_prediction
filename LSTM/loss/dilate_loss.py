import torch
from . import soft_dtw
from . import path_soft_dtw
from sdtw.distance import euclidean_distances


def dilate_loss(outputs, targets, alpha=0.01, gamma=0.01, device='cpu'):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :], outputs[k, :, :])
        # print(Dk.shape)
        D[k:k + 1, :, :] = Dk
    loss_shape = softdtw_batch(D, gamma)

    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
