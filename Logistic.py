#This script is to realize the logistic regression model in classification problem, 
#First import lib
import torch
#==============================
#import torchvision

#The torchvision lib provides Dataset as 
#torchvision.datasets.<datasetsName>(root='<rootDir>',train=<Boolean>,download=<Boolean>,*args)
#DataSets: MNIST for Algebra Number Classification, CIFAR10 for Object Classification
#In fact, we do not need to use this in this script

#==============================

#In this example, we use 1/(1+exp(-x)) after ax+b. (BCE Func)
#i.e. x->ax+b->f(ax+b)
#The above function reflect a result into [0,1] as the prob value
#i.e. in the new Logistic model, we have a linear part and a regression func part to obtain the result
#BCE function is provided in 
#torch.nn.functional.sigmoid(*args)
#We can use "import as" here
import torch.nn.functional as func

#Accordingly, the loss function will be changed to
#loss=-(ylogy'+(1-y)log(1-y')), where y'=ypred, loss for binary classification
#This loss function defines the diff between distributions (y and ypred), like cross-entropy
#This loss function is smaller-better, as when the distribution is identical, the f_loss=-1
#Attention: for every input, there will be a BCE loss. The mini-batch loss is defined as
#total_loss=avg(sum(BCE[i]))

#Now lets try to write the code as the Linear one
#What should we do when coding?
#1.define a class and class method
#2.call to realize the loss function 
#3.call to realize the optimizer
#4.itertion of training
#5.input real data

#1. define class and class methods
class Logistic(torch.nn.Module):
    def __init__(self):
        super(Logistic,self).__init__()
        self.linear=torch.nn.Linear(1,1)    #Here we still need a Linear method as we do have one before f()
        #__init__() is end here.
        #Why it is identical to the Linear.py?
        #Because BCE func do not have params, so we do not define it in __init__()

    def forward(self,x):
        y_pred=func.sigmoid(self.linear(x))
        return y_pred


model=Logistic()

#2.define loss func
criterion=torch.nn.BCELoss(size_average=False)
#torch.nn.BCELoss(..) is the loss func of BCE Func

#3.define the optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#4.Training
#Preparing the data
x=torch.Tensor([[1.0],[2.0],[3.0]])
y=torch.Tensor([[0],[0],[1]])

#Iteration
for epoch in range(100):
    y_pred=model(x)
    loss=criterion(y_pred,y)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

#5. Input real data
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)

