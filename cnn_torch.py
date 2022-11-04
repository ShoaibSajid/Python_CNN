
from matplotlib.pyplot import *
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import torch
import os
import mnist
from torch.cuda import amp

os.environ["CUDA_VISIBLE_DEVICES"]=""
device = "cpu"
print(f"Using {device} device")


# Define Model
class simple_network(nn.Module):
    def __init__(self):
        super(simple_network, self).__init__()
        self.conv0   = nn.Conv2d    (   1,  8,   kernel_size=(3, 3),     bias=False  , padding=1)   
        self.max0    = nn.MaxPool2d (            kernel_size=(2,2)                   )   
        self.conv1   = nn.Conv2d    (   8,  16,  kernel_size=(3, 3),     bias=False , padding=1 )   
        self.max1    = nn.MaxPool2d (            kernel_size=(2,2)                   )   
        self.flat    = nn.Flatten   (                                                )   
        self.linear0 = nn.Linear    (            784,  784                           )
        self.linear1 = nn.Linear    (            784,  10                            )
        self.soft    = nn.Softmax   (            dim=1                               )
        

    def forward(self,x):
        x=x.to(torch.device("cpu"))
        x_conv0      = self.conv0     (   x              )  
        x_max0       = self.max0      (   x_conv0        )
        x_conv1      = self.conv1     (   x_max0         )  
        x_max1       = self.max1      (   x_conv1        )
        x_flat       = self.flat      (   x_max1         )
        x_linear0    = self.linear0   (   x_flat         )
        x_linear1    = self.linear1   (   x_linear0      )
        x_prob       = self.soft      (   x_linear1      )

        debug=True
        # TODO: Fix Learning Rate and Gradient Descent Method, check batch size
        if debug:
            im=x[0][0]
            print(f"Input Image: {im[-4]}\n")
            
            print(f"conv0 filters: {model.state_dict()['conv0.weight'][0][0]}\n")
            
            print(f"x_conv0 : {x_conv0[0,0,:,:][-1]}\n")

            print(f"MaxPool0: {x_max0[0,0][-1]}\n")
            
            print(f"conv1 filters: {model.state_dict()['conv1.weight'][0][0]}\n")
            
            print(f"x_conv1 : {x_conv1[0,0,:,:][-1]}\n")

            print(f"MaxPool1: {x_max1[0,0][-1]}\n")
            
            print(f"FC0 Weight: {model.state_dict()['linear0.weight'][0][:10]}\n")
            
            print(f"FC0 Output: {x_linear0[0,:10]}\n")
            
            print(f"FC1 Weight: {model.state_dict()['linear1.weight'][0][:10]}\n")
            
            print(f"FC1 Output: {x_linear1[0,:10]}\n")
            
            print(f"SoftMax Output: {x_prob}\n")
            
        return x_prob
model = simple_network()
model.double()
print(model)



# Hyper Parameters
n_epochs = 3
# batch_size_train = 64
# batch_size_test = 1000
learning_rate = 0.005
momentum = 0
log_interval = 1
torch.backends.cudnn.enabled = False



# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)


# Scaler
scaler = amp.GradScaler()


# Loss Function
calculate_loss = nn.CrossEntropyLoss()
calculate_loss_nlll = nn.NLLLoss()



# Define Dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)



# Counter
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_labels) for i in range(n_epochs + 1)]



# Function for copying weights from basic model 
def copy_weights():
    # TODO: To copy filter weights
    name = "weights/debug.pkl"
    print(f"Loading weights from {name}")
    weight_file = open(name, "rb")
    weights = pickle.load(weight_file)
    print(f"\nLoading weights from {name} file")
    
    fconv0   = weights["conv0"]  
    fconv1   = weights["conv1"]  
    flinear0 = weights["fc0_weights"]
    blinear0 = weights["fc0_biases" ]
    flinear1 = weights["fc1_weights"]
    blinear1 = weights["fc1_biases" ]

    for i in range(fconv0.shape[0]):
        model.state_dict()['conv0.weight'][i][0]     = torch.from_numpy(fconv0[i]).double()
        print(f"Weights for Conv0 Filter {i} are {model.state_dict()['conv0.weight'][i][0][0]}   ")

    for i in range(fconv1.shape[0]):
        for j in range(fconv1.shape[3]):
            model.state_dict()['conv1.weight'][i,j,:,:]     = torch.from_numpy(fconv1[i,:,:,j]).double()
        print(f"Weights for Conv1 Filter {i} are {model.state_dict()['conv1.weight'][i][0][0]}   ")

    for i in range(flinear0.shape[1]):
        model.state_dict()['linear0.weight'][i]      = torch.from_numpy(flinear0[:,i]).double()
        if i<10: print(f"Weights for Linear Filter {i} are {model.state_dict()['linear0.weight'][i][:5]} ")
        
    for i in range(blinear0.shape[0]):
        model.state_dict()['linear0.bias'][i]        = torch.tensor(blinear0[i]).double()
        if i<10: print(f"Weights for Linear Bias   {i} are {model.state_dict()['linear0.bias'][i]}   ")
    
    for i in range(flinear1.shape[1]):
        model.state_dict()['linear1.weight'][i]      = torch.from_numpy(flinear1[:,i]).double()
        if i<10: print(f"Weights for Linear Filter {i} are {model.state_dict()['linear1.weight'][i][:5]} ")
        
    for i in range(blinear1.shape[0]):
        model.state_dict()['linear1.bias'][i]        = torch.tensor(blinear1[i]).double()
        if i<10: print(f"Weights for Linear Bias   {i} are {model.state_dict()['linear1.bias'][i]}   ")
    
        
            
            
# Train
def train(epoch):
    model.train()
    model.double()
    copy_weights()
    for batch_idx, (image, target) in enumerate(zip(train_images,train_labels)):
        optimizer.zero_grad()
        
        # Prepare Image
        img_numpy  = image[:,:,0]
        img_tensor = torch.tensor(img_numpy).double()
        img_for_model = img_tensor.unsqueeze(0).unsqueeze(0).double()

        # Prepare label
        label_vector = torch.zeros(1,10)
        label_vector[:,target]=1

        # Model Output
        output = model(img_for_model.double())
        
        # Calculate Loss
        loss = calculate_loss(label_vector, output)
        # loss_nlll = calculate_loss_nlll(label_vector, output)
        
        # Calculate gradients
        # scaler.scale(loss).backward()
        loss.backward()
        
        # Update weights
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        
        grad_bank = {}
        for idx, param in enumerate(model.parameters()):
            # print(f"Parameter: {param.name}")
            # print(f"Parameter: {param.names}")
            # print(f"Parameter: {param.grad.data}") 
            grad_bank[f"layer_{idx}"] = param.grad.data
        
        print(grad_bank['layer_5'])
            
            
        # if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image), len(train_images),100. * batch_idx / len(train_images), loss.item()),end='\r')
        # train_losses.append(loss.item())
        # train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        # torch.save(network.state_dict(), '/results/model.pth')
        # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
        
    
for epoch in range(1, n_epochs + 1):
    train(epoch)    
    print('test')








