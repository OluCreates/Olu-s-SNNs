import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate

#parameters
batch_size = 128
learningRate = 1e-3
num_epochs = 5 #cycles
num_steps = 25 #timesteps per image

#load and process data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#SNN Class
class SNN(nn.Module):

    #constructor - define layers
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256) #28x28 pixels flattened, 256 hidden neurons
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid()) #LIF neurons, beta=leak rate
        self.fc2 = nn.Linear(256, 10) #10 outputs, one per digit
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    #forward pass - runs every time we call model(imgs)
    def forward(self, x):
        mem1 = self.lif1.init_leaky() #init membrane potential for hidden layer
        mem2 = self.lif2.init_leaky() #init membrane potential for output layer
        spk_out = [] #store output spikes across all timesteps

        #timestep loop - each image is processed num_steps times
        for t in range(num_steps):
            cur1 = self.fc1(x) #weighted sum, produces input current for lif1
            spk1, mem1 = self.lif1(cur1, mem1) #update membrane potential, get spikes
            cur2 = self.fc2(spk1) #weighted sum on spikes, produces input current for lif2
            spk2, mem2 = self.lif2(cur2, mem2) #update membrane potential, get output spikes
            spk_out.append(spk2) #save this timesteps output spikes

        return torch.stack(spk_out, dim=0) #stack into one tensor of shape (num_steps, batch, 10)

#instantiate model, optimizer, loss function
model = SNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #Adam optimizer for weight updates
loss_fn = nn.CrossEntropyLoss() #measures how wrong predictions are

#training loop
for epoch in range(num_epochs):

    model.train() #set model to training mode
    for imgs, labels in train_loader:
        imgs = imgs.view(imgs.size(0), -1) #flatten 28x28 to 784
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
            spk_out = model(imgs) #forward pass
            spk_sum = spk_out.sum(dim=0) #sum spikes across timesteps
            preds = spk_sum.argmax(dim=1) #pick neuron that spiked most = prediction
            correct += (preds == labels).sum().item() #count correct predictions
            total += labels.size(0) #count total images seen

    print(f"Epoch {epoch+1}/{num_epochs} | Test Accuracy: {100*correct/total:.2f}%")

#save model
torch.save(model.state_dict(), "snn_mnist.pth")
print("Model saved.")
