#This script is about DataLoader

#As usual, import libs
import torch
import numpy as np
#Here are several concepts
#Batch-Size: Defines how many samples used when calling model(x)
#Iteration: Defines how many times calling model(), ending at using all sample
#Epoch: Defines how many times calling backward() and step()

#To enable DataLoader, extra libs have to be imported
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#The Dataset is an abstract classes, cannot be instantiate directly, need to be inherited
#DataLOader can be instantiated directly

#Let's try to define a Dataset
class DiabDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        #In our example, the dataset size is N*(8+1)
        #i.e. xy.shape=(N,9)
        self.len=xy.shape[0]
        #i.e. xy.shape[0]=N, which is the number of samples
        self.x=torch.from_numpy(xy[:,:-1])
        self.y=torch.from_numpy(xy[:,[-1]])
        #We have to clarify self.<attrName> as the attribute of the class
        #Empirically, for a small dataset, we can read it into memory
        #For a large dataset, we can maintain a file list(index) in initialization phase and call __getitem__() when we have to read the file

    def __getitem__(self,index):
        return self.x[index], self.y[index]
        #To fetch data from it
        #Method __getitem__() will be called when using expression <datasetName>[indexNum]
        #Such func is called 'magic function'
        #The returned value is a cell -> (x,y)

    def __len__(self):
        return self.len
        #To return length of dataset
        #Method __len__() will be called when calling len(<datasetName>)

#Now instantiate it:
dataset=DiabDataset('./datasets/diabetes.csv.gz')

#Then, we have to instantiate a DataLoader, which is used to load file
loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
#The arg shuffle defines if we have to choose random sample to make a batch (normally True for training, False for test)
#The arg num_workers defines the thread number used for loading.

#Now let's just use the network we constructed before.
inputDimNum=8
mediumDimNum=[6,4]
outputDimNum=1

class multiDim(torch.nn.Module):
    def __init__(self):
        super(multiDim,self).__init__()
        self.linear1=torch.nn.Linear(inputDimNum,mediumDimNum[0])
        self.linear2=torch.nn.Linear(mediumDimNum[0],mediumDimNum[1])
        self.linear3=torch.nn.Linear(mediumDimNum[1],outputDimNum)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

model=multiDim()
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


#Next we design our new training part
if __name__=='__main__':
    #Here we use if sentence to avoid spawn/folk API runtime error
    #So we build the training part into a module
    for epoch in range(1000):
        for i,data in enumerate(loader,0):
        #This enumerate(loader<DataLoader>,beginNum<int>) send (numberSeq<int>,data<dataType>)->(i,date) from the first to last in the loader
            inputs,labels=data
            #The following is responsible for extract the data into input and label
            #The above sentence is tricky that it can automatically spit data into 2 parts
            #The execution is as follows:
            #enumerate calls dataset[i], dataset[i] calls __getitem__() and return the val as 
            #(self.x[i],self.y[i])->(inputs,labels)
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            print(epoch, i, loss.item())
            #In the following section, inputs is a list of 32(batch_size) samples, and the same to labels.
            #y_pred is a list of 32 pridiction results
            #loss is a single number by aggregating 32 loss vals

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#Note:
#For datasets implemented in torchvision, all datasets are subclasses from torch.utils.data.Dataset
#i.e. They do not have to inherit from superClass, and can be passed to DataLoader(dataset<Dataset>,...) directly
#e.g. img=torchvision.datasets.ImageFolder(*args)
#     loader=torch.utils.data.DataLoader(img,*args)
