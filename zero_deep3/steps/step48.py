if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
import dezero.datasets
from dezero import optimizers
import matplotlib.pyplot as plt

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)
loss_list = []

for epoch in range(max_epoch):
    #データセットのインデックスをシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0
    
    for i in range(max_iter):
        #ミニバッチの生成
        batch_index = index[i*batch_size : (i+1)*batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]
        
        #勾配の算出、パラメータの更新
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(batch_t)
        
    avg_loss = sum_loss / data_size
    loss_list.append(avg_loss)
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
