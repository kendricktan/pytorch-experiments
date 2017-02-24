import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        x = self.fc1(x).clamp(min=0)
        x = self.fc2(x)
        return x

# Model
model = Net()
loss_fn = nn.MSELoss(size_average=False)

# High learning rate might cause overflow
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Data
# MSE needs to be normalized between 0-1
x = Variable(torch.from_numpy(np.linspace(0,1,100).astype(np.float32)))
y = Variable(torch.from_numpy(np.linspace(0,1,100).astype(np.float32)), requires_grad=False)

for j in range(100):
    optimizer.zero_grad()
    output = model(x)

    # Calculate loss via backprop
    loss = loss_fn(output, y)
    loss.backward()

    # Update params
    optimizer.step()

    print('Epoch: {}, Loss: {}'.format(j, loss.data[0]))

