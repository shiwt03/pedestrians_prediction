import numpy as np
from matplotlib import pyplot as plt

prediction = np.load('../prediction.npy')
y_test = np.load('../ytest.npy')

for ele in range(len(prediction)):
    pre_x = []
    pre_y = []
    real_x = []
    real_y = []
    for frame in range(len(prediction[ele])):
        pre_x.append(prediction[ele][frame][0])
        pre_y.append(prediction[ele][frame][1])
        real_x.append(y_test[ele][frame][0])
        real_y.append(y_test[ele][frame][1])
    plt.plot(pre_x, pre_y)
    plt.plot(real_x, real_y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    plt.clf()
