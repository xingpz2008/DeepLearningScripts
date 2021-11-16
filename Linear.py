#This script is used to realize the Linear Func using Pytorch
import torch


x=torch.Tensor([[1.0],[2.0],[3.0]])
y=torch.Tensor([[2.0],[4.0],[6.0]])
#This region is to load the data as (y,x), where y is the label and x is the input.
#Stored as the Tensor format @ torch.Tensor(```), creating the vector(tensor?)

#The next step is to construct the Calculation Graph
#Instead of considering the detailed Linear function, f'(x) and Loss Function

#In this example, we use y=w*x+b as Linear func, delta(y'-y)^2 as Loss func
#In real deployment, we need to consider the num of Dimension of Input tensor and output tensor
#E.g. 
#
#      z=w * x + b
#     3*1   4*1
#      --->w=3*4
#For all, loss is the exact value, not a vector. However, for every input and lable, there will be a loss, so we aggregate the loss value
#usually sum up. 
#Attention: This is the mini batch fashion, as input consists of many single values.

#Now we try to construct the class in the fucking object oriented programming language.
class Linear(torch.nn.Module):

    #Inherit from Module, we can use or edit the superMethods in Module
    #First, initiate func
    def __init__(self):
        super(Linear,self).__init__()
        self.linear=torch.nn.Linear(1,1) #This defines a Linear Func inside the network
        #torch.nn.Linear(in_feature_size,out_feature_size,bias=True)
        #In this example, input is 1 value, output is 1 value.

    def forward(self,x):        #In Python Module, it must be this name and args
        y_pred=self.linear(x)   #This defines that the forword algorithm in the network involves a Linear Unit, then return the result.
        return y_pred
    #forword func defines the calculation process
    #Module automatically construct the backword() function, or override.
    #i.e. the Module can make diviation function to functions, so to construct the backward()/ Module knows what is the backward() of nn.Linear

    #Attention: self.linear can be called because nn.Linear has method __call__(*args), where *args means non-explicit format variants.
    #__call__() defines what to do when calling <className>()


#Now we can instantiate the class
model=Linear()

#The following part describes how to construct the Loss Func
#Here we choose MESLoss, as l=(y'-y)^2
criterion=torch.nn.MSELoss(size_average=False)

#torch.nn.MSELoss(size_average=True, reduce=True), arg reduce indicates if the dims of output will be reduced.

#The following part describes how too construct optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#torch.optim.SGD(params,lr=<obj obj>,momentum=0,dampening=0,weight_decay=0,nesterov=False)
#model.parameters is callable, will include all parameters in the model.
#Optimizer now know which parameters should be updated in the iteration.

#Now lets start training
for epoch in range(100):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(epoch, loss.item())

    optimizer.zero_grad()   #The grad should be set to zero, or it will be accumulated.
    loss.backward()         #Calculate the new grad
    optimizer.step()        #Update the params with lr and grad

#Now lets output the training results
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

#We can also test the model with a new input
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)
    
