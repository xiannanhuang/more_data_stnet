import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from Dataset import STDataset
import os
from Dataset import Auxility_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml,logging
from models.GWNET import gwnet
from models.MTGNN import MTGNN
from models import STGCN,AGCRN,DCRNN
import copy
from tools.data_tools import expand_adjacency_matrix,get_datasets
from torch import nn




def validate_model(val_loader,model,scaler,config,device,criterion):
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[:2]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)[...,:2]
            loss = criterion(outputs, targets)
            val_loss += loss.item()*scaler.std
    val_loss /= len(val_loader)

    return val_loss




def kd_loss_with_confid(student_out,y_hat,target,c,epoch,config):
    
    alpha = config['alpha2']*(100*config['hype_kd']-epoch)/(100*config['hype_kd']) #
    # alpha=1
    
    y_=(1-alpha)*y_hat+target*(alpha)
    if config.get('if_kd',True)==True:
        if config['if_c']:
            # c=abs(y_hat-target).mean().item()
            w=torch.pow(c,config['hype_c'])
            loss=(w*abs(y_-student_out)).mean()
        else:
            loss=(abs(y_-student_out)).mean()
    elif config.get('if_kd',True)==False:
        if config['if_c']==True:
            # c=abs(y_hat-target).mean().item()
            # w=torch.exp(-abs(y_hat-target)/(config['hype_c']*c))
            w=torch.pow(c,config['hype_c'])
            loss=(w*abs(target-student_out)).mean()
        else:
            loss=(abs(target-student_out)).mean()

    return loss





    
def ema(student_model, teacher_model, alpha):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data = alpha * teacher_param.data + (1.0 - alpha) * student_param.data




def test(model,test_loader,device,scaler):
    rmse = 0.0
    mae = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[:2]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)[...,:2]
            
            # Inverse transform the scaled predictions to the original scale
            scaled_outputs = outputs.cpu().numpy()
            unscaled_outputs = scaler.inverse_transform(scaled_outputs)
            
            # Inverse transform the scaled targets to the original scale
            scaled_targets = targets.cpu().numpy()
            unscaled_targets = scaler.inverse_transform(scaled_targets)
            
            # Compute RMSE and MAE for this batch
            batch_rmse = np.sqrt(mean_squared_error(unscaled_targets.reshape(-1), unscaled_outputs.reshape(-1)))
            batch_mae = mean_absolute_error(unscaled_targets.reshape(-1), unscaled_outputs.reshape(-1))
            
            # Update the total RMSE and MAE
            rmse += batch_rmse
            mae += batch_mae
            num_samples += 1

    # Calculate the average RMSE and MAE over all samples
    rmse /= num_samples
    mae /= num_samples
    return rmse,mae


def init_train(model,train_month,log_file_path,type):
    with open(r'models\config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    config['model'],config['train_months']=model,train_month
    if 'confid' in type:
        config['model_confidence']=True
    # Configure logging
    log_dir = config['log_dir']  # You can specify the directory for log files
    os.makedirs(log_dir, exist_ok=True)
    # log_file_path = os.path.join(log_dir, '{}_training_month_{}_self_distill_vanilla.log'.format(config['model'],config['train_months']))
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('{}_training_month_{}_self_{}.log'.format(config['model'],config['train_months'],type))
    logging.info('Training {} model for {} months'.format(config['model'],config['train_months']))

    # Check if a GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets(config['data_dir'], config)
   
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    adj_mx =  expand_adjacency_matrix(np.load(config['adj_mx_file']), config['num_nodes'])[valid_gird][:,valid_gird][:,0,:]
    config['num_nodes']=len(valid_gird[0])
    logging.info(f"Total valid_gird: {len(valid_gird[0])}")
    print(f"Total valid_gird: {len(valid_gird[0])}")
    if config['model']=='GWNET':
    # Initialize the model
        model = gwnet(config, adj_mx).to(device)
    elif config['model']=='MTGNN':
        model=MTGNN(config,adj_mx).to(device)
    elif config['model']=='DCRNN':
        model=DCRNN.DCRNN(config,adj_mx).to(device)
    elif config['model']=='AGCRN':
        model=AGCRN.AGCRN(config,adj_mx).to(device)
    elif config['model']=='STGCN':
        model=STGCN.STGCN(config,adj_mx).to(device)



    
    for name, param in model.named_parameters():
        logging.info(f"Parameter Name: {name}\t Shape: {param.shape}")
    
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_trainable_params}")
    if 'confid' in type:
        model_path = os.path.join(config['model_dir'], f'{config["model"]}+{config["dataset_name"]}_train_months_24_with_confid_' + 'final_model.pth')
    else:

        model_path = os.path.join(config['model_dir'], f'{config["model"]}+{config["dataset_name"]}_train_months_24' + 'final_model.pth')
    # model.load_state_dict(torch.load(model_path))
    subset_indices1= list(range(len(train_dataset)))[-config['warm_up_day_num']*24:]
    # 创建一个SubsetRandomSampler，用于从数据集中选择指定的索引子集
    sampler1 = SubsetRandomSampler(subset_indices1)
    traindataloader1=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=False,sampler=sampler1)
    subset_indices2 = list(range(len(train_dataset)))[:-config['warm_up_day_num']*24]
    # 创建一个SubsetRandomSampler，用于从数据集中选择指定的索引子集
    sampler2 = SubsetRandomSampler(subset_indices2)
    traindataloader2=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=False,sampler=sampler2)

    student_model=model
    teacher_model=copy.deepcopy(student_model)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=config['kd_lr'])
    if 'lr_scheduler' in config:
        lr_scheduler_config = config['lr_scheduler']
        if lr_scheduler_config['type'] == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_config['step_size'], gamma=lr_scheduler_config['gamma'])
        elif lr_scheduler_config['type'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_config['factor'], patience=lr_scheduler_config['patience'], verbose=True)   
    return student_model, teacher_model, optimizer, traindataloader1, traindataloader2, lr_scheduler,val_loader,test_loader,config,device,scaler,lr_scheduler_config,adj_mx