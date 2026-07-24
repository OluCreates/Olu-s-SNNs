import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np

#parameters
batch_size = 128
learningRate = 1e-3
num_epochs = 5 #cycles
num_steps = 5 #timesteps per image
hidden_size = 64 #neurons in hidden layer
seed = 42
signed = True #signed {-1,0,+1} vs binary {0,1}
learn_dynamics = True #beta and threshold trainable

#seed control so runs are reproducible
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

#load and process data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#binary spike - fires +1 when membrane crosses threshold
#forward is a step function so gradient is 0 everywhere, backward uses fast sigmoid
#surrogate centered at threshold so gradient is strongest where firing happens
class BinarySpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem, threshold) #save for backward pass
        return (mem > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        return grad_output / (1 + torch.abs(mem - threshold)) ** 2, None

#signed spike - fires +1 above +threshold, -1 below -threshold, else 0
class SignedSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mem, threshold):
        ctx.save_for_backward(mem, threshold)
        pos = (mem > threshold).float()
        neg = (mem < -threshold).float()
        return pos - neg

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        #two surrogate peaks, one at each firing boundary
        grad_pos = grad_output / (1 + torch.abs(mem - threshold)) ** 2
        grad_neg = grad_output / (1 + torch.abs(mem + threshold)) ** 2
        return grad_pos + grad_neg, None

#LIF neuron layer - my own implementation
class LIF(nn.Module):

    def __init__(self, beta=0.9, threshold=1.0, signed=signed, learn=learn_dynamics):
        super().__init__()
        self.signed = signed
        if learn:
            self.beta = nn.Parameter(torch.tensor(beta)) #trainable decay
            self.threshold = nn.Parameter(torch.tensor(threshold)) #trainable threshold
        else:
            self.register_buffer('beta', torch.tensor(beta)) #on device but frozen
            self.register_buffer('threshold', torch.tensor(threshold))

    def forward(self, cur, mem):
        beta = self.beta.clamp(0.0, 1.0) #keep decay under 1 or membrane grows and training blows up
        mem = beta * mem + cur #leaky integrate

        if self.signed:
            spk = SignedSpike.apply(mem, self.threshold)
        else:
            spk = BinarySpike.apply(mem, self.threshold)

        mem = mem - spk * self.threshold #soft reset, works for +1 and -1 both
        return spk, mem

#SNN Class
class SNN(nn.Module):

    #constructor - define layers
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size) #28x28 pixels flattened
        self.lif1 = LIF()
        self.fc2 = nn.Linear(hidden_size, 10) #10 outputs, one per digit
        self.lif2 = LIF()

    #forward pass - runs every time we call model(imgs)
    def forward(self, x):
        mem1 = torch.zeros(1, device=x.device) #broadcasts to batch shape
        mem2 = torch.zeros(1, device=x.device)
        spk_out = [] #store output spikes across all timesteps

        #timestep loop - each image is processed num_steps times
        for t in range(num_steps):
            cur1 = self.fc1(x) #weighted sum, input current for lif1
            spk1, mem1 = self.lif1(cur1, mem1) #update membrane potential, get spikes
            cur2 = self.fc2(spk1) #weighted sum on spikes, input current for lif2
            spk2, mem2 = self.lif2(cur2, mem2) #update membrane potential, get output spikes
            spk_out.append(spk2) #save this timesteps output spikes

        return torch.stack(spk_out, dim=0) #shape (num_steps, batch, 10)

#instantiate model, optimizer, loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_fn = nn.CrossEntropyLoss()

#efficiency metrics for this config
num_params = sum(p.numel() for p in model.parameters())
synaptic_ops = (784 * hidden_size + hidden_size * 10) * num_steps

print(f"Using device: {device}")
print(f"neurons: {hidden_size} | timesteps: {num_steps} | spikes: {'signed' if signed else 'binary'}")
print(f"params: {num_params:,} | synaptic ops: {synaptic_ops:,} per inference\n")

#training loop
for epoch in range(num_epochs):

    model.train() #set model to training mode
    for imgs, labels in train_loader:
        imgs = imgs.view(imgs.size(0), -1) #flatten 28x28 to 784
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad() #clear gradients from previous batch
        spk_out = model(imgs) #forward pass, runs timestep loop internally
        spk_sum = spk_out.sum(dim=0) #sum spikes across timesteps, shape (batch, 10)
        loss = loss_fn(spk_sum, labels) #compute loss against real labels
        loss.backward() #backprop, compute gradients
        optimizer.step() #update weights using gradients

    #test accuracy after each epoch
    model.eval() #set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): #disable gradient tracking, saves memory
        for imgs, labels in test_loader:
            imgs = imgs.view(imgs.size(0), -1) #flatten 28x28 to 784
            imgs = imgs.to(device)
            labels = labels.to(device)
            spk_out = model(imgs) #forward pass
            spk_sum = spk_out.sum(dim=0) #sum spikes across timesteps
            preds = spk_sum.argmax(dim=1) #index of neuron that spiked most = prediction
            correct += (preds == labels).sum().item() #count correct predictions
            total += labels.size(0) #count total images seen

    print(f"Epoch {epoch+1}/{num_epochs} | Test Accuracy: {100*correct/total:.2f}%")

#save model
torch.save(model.state_dict(), "snn_mnist_best.pth")
print("Model saved.")
