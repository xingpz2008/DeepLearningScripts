#This script is to use Convlutional Neutral Network in Deep Learning
#In CNN, there are pooling, conv and Linear layers
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.utils.data

#The convlutional function is provided as follows:
#torch.nn.Conv2d<Module>(in_channels<int>,out_channels<int>,kernel_size=kernel_size<int>,*args)
#padding is also provided,using it by adding padding=padding<int> in arg list
#stride can be configured by adding arg that stride=stride<int>
#Note that, here, in the intermedium stages, we do not need the shape of matrix, just need#channelNums

#If we are too poor to afford a server with a GPU, we should set
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Usually, we will have a input matrix, then we have to reshape it using
#torch.Tensor(input).view(batch_size,channelNum,Width,Height)

#Pooling: We present MaxPooling here, which do not change the channelNum
#Usage: torch.nn.MaxPool2d(kernel_size<int>,stride=kernel_size(def),*args)

#Before we construct the Network, we have to calculate the inDim and outDim respectively
#The last full-connected layer cares the inDims most. For ez building,
#we can define our network before the FC Layer, then use size() as the inDims of FC.

#Now let's define out network
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(320,10)
        self.activation=torch.nn.ReLU()

    def forward(self,x):
        batch_size=x.size(0)
        x=self.activation(self.pooling(self.conv1(x)))
        x=self.activation(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        #Here we separate x by batch, that n batch, each batch have 320 samples
        x=self.fc(x)
        return x

model=CNN()
model.to(device)
#This sentence will send the model to GPU if one is available
#Accordingly, we shoud send input and label to device as 
#inputs,target=inputs.to(device),target.to(device) in training and test func

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081))])
train_dataset=torchvision.datasets.MNIST(root='./datasets', train=True,transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./datasets',train=False,transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=False,num_workers=0)

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
criterion=torch.nn.CrossEntropyLoss()

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target=data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx%300==299:
            print('[%d %5d] loss: %.3f' % (epoch+1,batch_idx+1,running_loss/2000))
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,targets=data
            inputs,targets=inputs.to(device),targets.to(device)
            outputs=model(inputs)
            _,pred=torch.max(outputs.data,dim=1)
            total+=targets.size(0)
            correct+=(pred==targets).sum().item()
    print('Accuracy on test set:%d %%' %(100*correct/total))

if __name__=='__main__':
    for epoch in range(100):
        train(epoch)
        if epoch %10==0:
            test()

