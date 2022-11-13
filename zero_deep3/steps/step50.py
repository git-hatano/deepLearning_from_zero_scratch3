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
import matplotlib.pyplot as plt


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
test_set = dezero.datasets.Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

train_loss, train_acc, test_loss, test_acc = [], [], [], []

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
    train_loss.append(sum_loss / len(train_set))
    train_acc.append(sum_acc / len(train_set))
    
    #test
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():#勾配不要モード
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), 
                                                    sum_acc / len(test_set)))
    test_loss.append(sum_loss / len(test_set))
    test_acc.append(sum_acc / len(test_set))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

axes[0].plot(train_loss, label="train")
axes[0].plot(test_loss, label="test")
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend()

axes[1].plot(train_acc, label="train")
axes[1].plot(test_acc, label="test")
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('acc')
axes[1].legend()

fig.savefig("step50.png")
