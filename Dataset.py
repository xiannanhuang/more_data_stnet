import torch
from torch.utils.data import Dataset
import numpy as np
class STDataset(Dataset):
    def __init__(self, data, config):
        '''
        data:nparray (265,day_num,48,2)
        '''

        
        # 根据索引创建新的data数组
        self.data = data
        self.data = self.data.reshape(data.shape[0],-1,2)
        self.input_window = config['input_window']
        self.output_window = config['output_window']
    def __len__(self):
        return self.data.shape[1] - (self.input_window + self.output_window) + 1

    def __getitem__(self, index):
        x = self.data[:, index:index + self.input_window, :].transpose(1,0,2)
        y = self.data[:, index + self.input_window:index + self.input_window + self.output_window, :].reshape(-1,self.output_window,2).transpose(1,0,2)
        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32),index

class Auxility_dataset(Dataset):
    def __init__(self, dataset, classindex,config):
        super(Auxility_dataset, self).__init__()
        self.input_window = config['input_window']
        self.output_window = config['output_window']
        self.data = dataset
        self.classindex = torch.zeros(self.data.shape[1] - (self.input_window + self.output_window) + 1,dtype=torch.float32)
        self.classindex[-classindex:] = 1
    
    def __getitem__(self, index):
        x = self.data[:, index:index + self.input_window, :].transpose(1,0,2)
        return torch.tensor(x,dtype=torch.float32),self.classindex[index]
    
    def __len__(self):
        return self.data.shape[1] - (self.input_window + self.output_window) + 1
