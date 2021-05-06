import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
from torchvision.datasets import FashionMNIST
from export import save, load

train = FashionMNIST("data", train=True, download=True)
test = FashionMNIST("data", train=False, download=True)

train_data = train.data.reshape(-1, 784) / 127.5 - 1
train_labels = train.targets

test_data = test.data.reshape(-1, 784) / 127.5 - 1
test_labels = test.targets


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.from_numpy(load("weights/w1.data").T))
        self.b1 = torch.nn.Parameter(torch.from_numpy(load("weights/b1.data")[0]))
        self.w2 = torch.nn.Parameter(torch.from_numpy(load("weights/w2.data").T))
        self.b2 = torch.nn.Parameter(torch.from_numpy(load("weights/b2.data")[0]))
        # self.w1 = torch.nn.Parameter(torch.randn(784, 256) / np.sqrt(784) * 2)
        # self.b1 = torch.nn.Parameter(torch.zeros(256))
        # self.w2 = torch.nn.Parameter(torch.randn(256, 10) / np.sqrt(256) * 2)
        # self.b2 = torch.nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x = F.relu(x @ self.w1 + self.b1)
        return x @ self.w2 + self.b2

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

model = Model().cuda()

# print(F.cross_entropy(model(test_data), test_labels).item())

bs = 1250
epochs = 10
lr = 0.01

start = time.time()
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(test_data) // bs):
        start_i = i * bs
        end_i = start_i + bs
        xb = test_data[i * bs : (i + 1) * bs]
        yb = test_labels[i * bs : (i + 1) * bs]
        pred = model(xb.cuda())
        loss = F.cross_entropy(pred, yb.cuda())

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
            model.zero_grad()
        total_loss += loss.detach().item()
    print("Epoch: ", epoch, "loss: ", total_loss / (len(test_data) // bs))
print(time.time() - start)

print(accuracy(model(test_data.cuda()), test_labels.cuda()))
# save(model.w1.detach().cpu().numpy().T, "weights/w1.data")
# save(model.b1.detach().cpu().numpy()[None, :], "weights/b1.data")
# save(model.w2.detach().cpu().numpy().T, "weights/w2.data")
# save(model.b2.detach().cpu().numpy()[None, :], "weights/b2.data")
