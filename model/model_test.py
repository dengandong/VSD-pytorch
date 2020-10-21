import torch
from model.ConvGRU import ConvGRU

y = torch.rand(2, 120)

fc = torch.nn.Linear(120, 4*4*32*8)
y = fc(y)
y = y.view(2, 32*8, 4, 4).contiguous()
print(y.size())

gru1 = ConvGRU(8*32, hidden_sizes=[8*32, 8*32], n_layers=2, kernel_sizes=[3, 3])
y1 = gru1(y)
print(len(y1))
for i in range(len(y1)):
    print(y1[i].size())

y2 = gru1(y, y1)
print(len(y2))
for i in range(len(y2)):
    print(y2[i].size())
