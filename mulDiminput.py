#This script is used for classification when input dim is not sole

#Let's load lib first
import torch
import numpy as np

#In the scenario we have to process the dataset, so we have to use numpy lib

#In multi-dim-input task, we need to load the data to our x matrix
#In previous script, we use y=wx+b
#Now x is mul-dim input, so w should be W, a matrix as well
#i.e. y_pred_i=sigmoid([x1,x2,...,xn][w1,...,wn]^T+b)=sigmoid(z_i) empirically, where x1 to xn is the feature of i-th input x 
#Note that, b is identical for every x_i

#For mini batch, we choose N samples at one time(batch), 
#[z_1]   [x1_1, x1_2, ..., x1_n]   [w1]   [b]
#[z_2] = [x2_1, x2_2, ..., x2_n] * [w2] + [b]
#[...]   [...   ...   ...  ... ]   [..]   [.]
#[z_n]   [xN_1, xN_2, ..., xN_n]   [wn]   [b]
# where n is numAttr, N is numBatch

#0.Global Attr, Hyper-params, Data Loader
inputDimNum=8
mediumDimNum=[6,4]
outputDimNum=1

xy=np.loadtxt('./datasets/diabetes.csv.gz',delimiter=',',dtype=np.float32)
#np.loadtxt(rootDir,delimiter='<symbol>',dtype=<type>), the 1st arg indicates the filepath, the second arg defines delimiter in .csv file
#Nvidia device supports float32 more than double
x=torch.from_numpy(xy[:,:-1])
y=torch.from_numpy(xy[:,[-1]])
#In xy[:,:-1], : stands for all data, :-1 eliminates the last column
#In xy[:,[-1]], [-1] re-format the last column into a matrix. If use -1, y will be a vector (incorrect format)

#1.Lets define the class
class multiDim(torch.nn.Module):
    def __init__(self):
        super(multiDim,self).__init__()
        #Usually we would use: self.linear=torch.nn.Linear(inputDimNum,outputDimNum)
        #Here, torch.nn.Linear(inputDim,outputDim), out input dim= n, output dim=1
        #However, we are not satisfied when always using single layer network
        #This time, lets try to define 3 linear layers with 8->6->4->1 dims and connect them 
        self.linear1=torch.nn.Linear(inputDimNum,mediumDimNum[0])
        self.linear2=torch.nn.Linear(mediumDimNum[0],mediumDimNum[1])
        self.linear3=torch.nn.Linear(mediumDimNum[1],outputDimNum)
        self.sigmoid=torch.nn.Sigmoid() #Here we treat the Sigmoid as a layer, not a simple function, so we use <class>torch.nn.Sigmoid call its __init__()

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
        #Here, we still use func.sigmoid, we can also use torch.nn.Sigmoid(*args)
        #Accordingly, we change forward func into a 3-layer one
        #Tip: For a sequential network, we can use the same variant in input and output (i.e. x=f(x)) to avoid typo.

model=multiDim()

#2.Define the loss func
criterion=torch.nn.BCELoss(size_average=True)

#3.Optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#4.Training
for epoch in range(100):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

#print('Layer1: w=', model.linear1.weight.item())
#print('Layer1: b=', model.linear1.bias.item())
#print('Layer2: w=', model.linear2.weight.item())
#print('Layer2: b=', model.linear2.bias.item())
#print('Layer3: w=', model.linear3.weight.item())
#print('Layer3: b=', model.linear3.bias.item())

#5.Input real data
#ignored here
