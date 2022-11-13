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

def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x

max_epoch = 5
batch_size= 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True, transform=f) 
test_set = dezero.datasets.MNIST(train=False, transform=f)

# x, t = train_set[0]
# plt.imshow(x.reshape(28, 28), cmap='gray')
# plt.axis('off')
# plt.show()
# print(f'lable: {t}')

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

# 隠れ層*2+Adam: PCが落ちる
# model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
# optimizer = optimizers.Adam().setup(model)

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


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

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

fig.savefig("step51.png")
