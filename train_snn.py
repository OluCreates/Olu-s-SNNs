import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate

#parameters
batch_size = 128
learningRate = 1e-3
epochs = 5 #cycles
steps = 25  

#load and process data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#SNN Class
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
      self.fc1 = nn.Linear(784, 256) #28x28 pixels     
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())   
        self.fc2 = nn.Linear(256, 10)  #10 outputs
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
mem1 = self.lif1.init_leaky()
mem2 = self.lif2.init_leaky()
spk_out = []
for t in range(num_steps):
    cur1 = self.fc1(x) #calc current to pass into first batch of Leaky neurons
    spk1, mem1 = self.lif1(cur1, mem1) #keep track and add our membrane poten
    cur2 = self.fc2(spk1)
    spk2, mem2 = self.lif2(cur2, mem2)
    spk_out.append(spk2)   #save output spike
    return torch.stack(spk_out, dim=0)
