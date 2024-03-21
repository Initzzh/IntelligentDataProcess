import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def get_data(path = 'train_data_small.csv'):
    data =[]
    dataframe = pd.read_csv(path, header=0)
    m = len(dataframe)
    data_input = []
    data_target = []
    data_input = np.zeros((1,40000))
    data_target = np.zeros((1,3))
    for i in range(m):
        point_a = dataframe['a'][i]
        point_b = dataframe['b'][i]
        point_c = dataframe['c'][i]
        # print(point_c)

        df = pd.read_csv('./data/'+dataframe['path'][i],header=None)
        df['time'] = df[0].str.split()
        df['data']=df['time'].map(lambda x:eval(x[1]))

        # x = np.linspace(0,20000,20000)
        x = np.arange(0,20000,1)
        # x = x.reshape((20000,1))
        y = df['data'].to_numpy()
        input = np.concatenate((x,y),axis=0).reshape((1,40000))
        
        data_input = np.concatenate((data_input,input),axis=0)
        special_point = np.array([point_a,point_b,point_c]).reshape((1,3))
        data_target = np.concatenate((data_target,special_point),axis=0)
       
    data_input = np.delete(data_input,0,axis=0)
    data_target = np.delete(data_target,0,axis=0)
    return (data_input,data_target)


def get_data2(path = 'train_data_small.csv'):
    data =[]
    dataframe = pd.read_csv(path, header=0)
    m = len(dataframe)
    data_input = []
    data_target = []
    data_input = np.zeros((1,2,20000))
    data_target = np.zeros((1,3))
    for i in range(m):
        point_a = dataframe['a'][i]
        point_b = dataframe['b'][i]
        point_c = dataframe['c'][i]
        # print(point_c)

        df = pd.read_csv('./data/'+dataframe['path'][i],header=None)
        df['time'] = df[0].str.split()
        df['data']=df['time'].map(lambda x:eval(x[1]))

        # x = np.linspace(0,20000,20000)
        x = np.arange(0,20000,1).reshape((1,1,20000))
        # x = x.reshape((20000,1))
        y = df['data'].to_numpy().reshape((1,1,20000))
        input = np.concatenate((x,y),axis=1)
        
        data_input = np.concatenate((data_input,input),axis=0)
        special_point = np.array([point_a,point_b,point_c]).reshape((1,3))
        data_target = np.concatenate((data_target,special_point),axis=0)

    data_input = np.delete(data_input,0,axis=0)
    data_target = np.delete(data_target,0,axis=0)
    return (data_input,data_target)






# 将数据加载到PyTorch Dataset中
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        input =  self.data[0][idx]
        target = self.data[1][idx]
        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input,target

# 定义模型

class CnvModel(nn.Module):
    def __init__(self):
        super(CnvModel,self).__init__()
        input= 2
        features = 32

        layers = []

        layers.append(nn.Conv1d(in_channels=2,out_channels=features,kernel_size=9,stride=1,padding=4))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(features))
        layers.append(nn.MaxPool1d(kernel_size=10,stride=2,padding=4))
        size = 10000
        for _ in range(2):
            layers.append(nn.Conv1d(in_channels=features,out_channels=features,kernel_size=9,stride=1,padding=4))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.MaxPool1d(kernel_size=10,stride=2,padding=4))
            size = size//2
        
        self.seq = nn.Sequential(*layers)

        self.fc = nn.Linear(features*size,3)

        # size = 20000

        # self.conv1 = nn.Conv1d(in_channels=2,out_channels=features,kernel_size=9,stride=1,padding=4)
        # self.relu1 = nn.ReLU()
        # self.batchnoraml1 =  nn.BatchNorm1d(features)
        # self.maxpool1 = nn.MaxPool1d(kernel_size=10,stride=2,padding=4)
        # size = size//2

        # self.conv2 = nn.Conv1d(in_channels=features,out_channels=features,kernel_size=9,stride=1,padding=4)
        # self.relu2 = nn.ReLU()
        # self.batchnoraml2  = nn.BatchNorm1d(features)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=10,stride=2,padding=4) # padding = 4
        # size = size//2

        # self.conv3 = nn.Conv1d(in_channels=features,out_channels=features,kernel_size=9,stride=1,padding=4)
        # self.relu3 = nn.ReLU()
        # self.batchnoraml3  = nn.BatchNorm1d(features)
        # self.maxpool3 = nn.MaxPool1d(kernel_size=10,stride=2,padding=4)
        # size = size//2

        # # size = 10000
        # self.fc = nn.Linear(features*size,3)

        # layers = []
        # layers.append(nn.Conv1d(in_channels=2,out_channels=features,kernel_size=9,stride=4))
        # layers.append(nn.ReLU())
        # layers.append(nn.MaxPool1d(kernel_size=3,stride=2))

        # for _ in range(1):
        #     layers.append(nn.Conv1d(in_channels=features,out_channels=features,kernel_size=9,stride=1,padding=4))
        #     layers.append(nn.BatchNorm1d(features))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.MaxPool1d(kernel_size=3,stride=2))
        # size = 2000
        # layers.append(nn.Linear(features*size,3))

        # self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x  = self.seq(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.batchnoraml1(x)
        # x = self.maxpool1(x)

        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.batchnoraml2(x)
        # x = self.maxpool2(x)

        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.batchnoraml3(x)
        # x = self.maxpool2(x)

        # x = torch.flatten(x,1)
        # x = self.fc(x)

        # return x



# 定义模型

class LiModel(nn.Module):
    def __init__(self):
        super(LiModel, self).__init__()
        layers = []
        # layers.append(nn.Linear(input_dim,input_dim/10))
        # layers.append(nn.BatchNorm1d(input_dim))
        # layers.append(nn.ReLU(inplace=True))
        input_dim=40000
        out_dim=3
        for _ in range(2):
            layers.append(nn.Linear(input_dim, int(input_dim/10)))
            input_dim = int(input_dim/10)
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
        
            # layers.append(nn.Dropout(p=0.4))
        
        # for _ in range(3):
        #     layers.append(nn.Linear(input_dim,input_dim))
        #     # input_dim /= 10
        #     layers.append(nn.BatchNorm1d(input_dim))
        #     layers.append(nn.ReLU(inplace=True))
        #     layers.append(nn.Dropout(p=0.5))
        
        layers.append(nn.Linear(input_dim,out_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(40000, 2000)
        self.batchnormal1 = nn.BatchNorm1d(4000)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2000, 100)
        self.batchnormal2 = nn.BatchNorm1d(400)
        self.dropout2 = nn.Dropout(p=0.5)



        self.fc3 = nn.Linear(100, 3)
    

    def forward(self,x):
        x = self.fc1(x)
        x = self.batchnormal1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        x = self.batchnormal2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    # 



import math

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm1d') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)


def train():

    path = 'train_data.csv'
    data = get_data2(path)
    dataset = MyDataset(data)
    print("加载数据集")
    # input,target = dataset[0]

    train_size = 90
    test_size = len(dataset)-train_size

    # 设置随机种子
    torch.manual_seed(0)

    train_dataset, test_dataset = random_split(dataset,[train_size, test_size])
    num_epochs = 40000
    learning_rate = 1e-3
    batch_size = 90
    test_batch_size = 1

    device = torch.device("cuda:0")

    train_dataloader = DataLoader(train_dataset,batch_size =batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size =test_batch_size,shuffle=True)
    # test_input = torch.randn(90,2,20000)
    # test_model = CnvModel()
    # t = test_model(test_input)
    model = CnvModel()
    model = model.to(device)
    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =learning_rate)


    for epoch in range(num_epochs):
        if (epoch == 4000):
            learning_rate *=0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        # if (epoch == 10000):
        #     learning_rate *=0.1
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = learning_rate
        if (epoch == 10000):
            learning_rate *=0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
        # # if(epoch==2000):
        # #     learning_rate *=0.2
        # #     for param_group in optimizer.param_groups:
        # #         param_group["lr"] = learning_rate
        
        # if(epoch==3000):
        #     learning_rate *=0.1
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = learning_rate

            # optimizer.param_groups['lr'] = learning_rate
        #train
        train_distance_loss =0
        for i, (input,target) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            # print("predict:",output)
            loss = criterion(output,target)
            
            loss.backward()
            # 计算3个点之间的距离
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            distance_loss = 0 
            for i in range(output.shape[0]):
                distance_loss += (abs(output[i][0]-target[i][0])+abs(output[i][1]-target[i][1])+abs(output[i][2]-target[i][2]))
            distance_loss /=output.shape[0]
            train_distance_loss += distance_loss
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # print("Epoch [%d/%d], Step[%d/%d], Loss: %4f" %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))
        train_distance_loss /= (train_size//batch_size)
        # print("Epoch [%d/%d], Loss: %4f, avg_distance_loss: %4f" %(epoch+1, num_epochs, loss.item(), train_distance_loss))
        if train_distance_loss < 10:
            torch.save(model.state_dict(),'models/model.pth')
            break

        # valid
        val_distance_loss = 0
        for i,(input, target) in enumerate(test_dataloader):
            model.eval()
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                output =  model(input)
                eval_loss = criterion(output,target)
            # eval_all_loss += eval_loss.item()
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            # 计算distance_loss:
            distance_loss = 0
            for i in range(output.shape[0]):
                distance_loss += (abs(output[i][0]-target[i][0])+abs(output[i][1]-target[i][1])+abs(output[i][2]-target[i][2]))
            distance_loss /=output.shape[0]
            val_distance_loss += distance_loss

        val_distance_loss /= (test_size//test_batch_size)
        print("Epoch [%d/%d], Loss: %4f, distance_loss: %4f, eval_Loss: %4f, eval_distance_loss: %4f" %(epoch+1, num_epochs, loss.item(), train_distance_loss,eval_loss.item(),val_distance_loss))
        # print("Epoch [%d/%d], eval_loss:%4f  val_distance_loss:%4f" % (epoch+1, num_epochs, eval_loss.item(),val_distance_loss))
        
        
    # test
    # torch.save(model.state_dict(),'model2.pth')
    # test
    test_distance_loss =0
    test_all_loss = 0
    # test_dataloader = DataLoader(test_dataset,batch_size =1,shuffle=True)
    for i,(input, target) in enumerate(test_dataloader):
        model.eval()
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output =  model(input)
            test_loss = criterion(output,target)
        test_all_loss += test_loss.item()
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # print(output.shape)
        # return
        print("predict_a:",output[:,0],"practial_a:",target[:,0],"predict_b:",output[:,1],"practial_b:",target[:,1],"predict_c:",output[:,2],"practial_c:",target[:,2])
        
        distance_loss = 0
        for i in range(output.shape[0]):
            distance_loss += (abs(output[i][0]-target[i][0])+abs(output[i][1]-target[i][1])+abs(output[i][2]-target[i][2]))
        distance_loss /=output.shape[0]

        test_distance_loss += distance_loss
        print("test_loss:%4f distance_loss:%4f" %(test_loss.item(),distance_loss))

    test_distance_loss /= len(test_dataset)
    test_all_loss /=len(test_dataset)
    print("eval_loss:%4f eval_distance_loss: %4f" % (test_all_loss,test_distance_loss))
    


# train()



# def predict():
#     criterion = nn.MSELoss()
#     # accuacy = 0
#     model = CnvModel()
#     model.to(device)
#     # model.load_state_dict("model.pth")
#     model.load_state_dict(torch.load("model.pth"))
#     test_loss = 0
#     test_dataloader = DataLoader(test_dataset,batch_size =test_batch_size,shuffle=True)
#     test_distance_loss =0
#     for i,(input, target) in enumerate(test_dataloader):
#         model.eval()
#         input = input.to(device)
#         target = target.to(device)
#         with torch.no_grad():
#             output =  model(input)
#             test_loss = criterion(output,target)
#         # test_loss = eval_loss.item()
#         # eval_all_loss += eval_loss.item()
#         output = output.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
#         # print(output.shape)
#         # return
#         print("predict_a:",output[:,0],"practial_a:",target[:,0],"predict_b:",output[:,1],"practial_b:",target[:,1],"predict_c:",output[:,2],"practial_c:",target[:,2])
        
#         distance_loss = 0
#         for i in range(output.shape[0]):
#             distance_loss += (abs(output[i][0]-target[i][0])+abs(output[i][1]-target[i][1])+abs(output[i][2]-target[i][2]))
#         distance_loss /=output.shape[0]

#         test_distance_loss += distance_loss
#         print("eval_loss:%4f distance_loss:%4f" %(test_loss.item(),distance_loss))
#     test_distance_loss /= distance_loss
#     test_loss /=len(test_dataset)
#     print("test_loss: %4f test_distance_loss: %4f" % (test_loss,test_distance_loss))


# 预测结果
def predict(model_path,input):
    """
    input: 1*2*20000
    """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = CnvModel()
    model.to(device)
    # if device=='cuda':
    #     model.load_state_dict(torch.load(model_path))
    # else:
    # 根据当前设配类型（cpu，cuda）加载模型
    model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage))
    input = input.to(device)
    
    output = model(input)
    # print(output.shape)
    # if 
    output = output.detach().cpu().numpy()
    # print(output.shape)
    point_a = output[:,0]
    point_b = output[:,1]
    point_c = output[:,2]
    
    print(point_a,point_b,point_c)
    return point_a,point_b,point_c



