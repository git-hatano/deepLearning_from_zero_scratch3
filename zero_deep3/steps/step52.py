if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
from dezero import optimizers
import dezero.datasets
from dezero import Variable, DataLoader, no_grad
import matplotlib.pylab as plt
import dezero.cuda


max_epoch = 5
batch_size= 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True) 
train_loader = DataLoader(train_set, batch_size)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

#GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    #train
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    print(f'epoch: {epoch+1}')
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), 
                                                        sum_acc / len(train_set)))    
