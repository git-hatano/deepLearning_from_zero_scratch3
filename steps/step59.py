if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP, Model
from dezero import optimizers
import dezero.datasets
from dezero import Variable, DataLoader, no_grad
import matplotlib.pyplot as plt


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

# seq_data = [np.random.randn(1,1) for _ in range(1000)]
# xs = seq_data[:-1]
# ts = seq_data[1:]
# model = SimpleRNN(10, 1)
# loss, cnt = 0, 0
# for x, t in zip(xs, ts):
#     y = model(x)
#     loss += F.mean_square_error(y, t)
#     cnt += 1
#     if cnt == 2:
#         model.cleargrads()
#         loss.backward()
#         break


# train_set = dezero.datasets.SinCurve(train=True)
# print(len(train_set))
# print(train_set[0])
# print(train_set[1])
# print(train_set[2])
# print(train_set[3])

# xs = [example[0] for example in train_set] 
# ts = [example[1] for example in train_set] 
# plt.plot(np.arange(len(xs)), xs, label='xs') 
# plt.plot(np.arange(len(ts)), ts, label='ts')
# plt.legend()
# plt.show()


max_epoch = 100
hidden_size= 100
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = optimizers.Adam().setup(model)

#train
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0
    
    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_square_error(y, t)
        count += 1
    
    if count % bptt_length == 0 or count == seqlen:
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    
    avg_loss = float(loss.data) / count
    print(f'| epoch {epoch+1} | loss {avg_loss}')

#predict
xs = np.cos(np.linspace(0, 4*np.pi, 1000))
model.reset_state()
pred_list = []

with no_grad():
    for x in xs:
        x = np.array(x).reshape(1,1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)') 
plt.plot(np.arange(len(xs)), pred_list, label='predict') 
plt.xlabel('x') 
plt.ylabel('y') 
plt.legend() 
plt.show()
