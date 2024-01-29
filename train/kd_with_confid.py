import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler
from datetime import datetime
from models.GWNET import gwnet
from models.MTGNN import MTGNN,MTGNN_c
from models import STGCN,AGCRN,DCRNN
from Dataset import STDataset
from tools import train_tools,data_tools
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import copy

def train_1_epoch_vanilla(dataloader,model,optimizer,config,device,criterion=torch.nn.MSELoss()):
  
    train_loss=0.
    model.train()
      
    start_time = datetime.now()
    for batch in dataloader:
        inputs, targets = batch[:2]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)[...,:2]

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for this epoch
    train_loss /= len(dataloader)
    end_time = datetime.now()
    epoch_duration = end_time - start_time
    return train_loss,epoch_duration

def self_distill_with_confid(classification,student_model,teacher_model,dataloader,optimizer,config,device,epoch):
    train_loss=0.
    student_model.train()
    
    start_time = datetime.now()
    for batch in dataloader:
        inputs, targets = batch[:2]
        
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = teacher_model(inputs)
            y_hat=outputs[...,:2]
            c=classification(inputs)[:,...,-1,-1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,config['num_nodes'],2)
            c=torch.sigmoid(c)
        student_out=student_model(inputs)[...,:2]
        loss=train_tools.kd_loss_with_confid(student_out,y_hat,targets,c,epoch,config)
        # loss=train_tools.kd_loss1(student_out,y_hat,targets,epoch,config)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average training loss for this epoch
    train_loss /= len(dataloader)
    end_time = datetime.now()
    epoch_duration = end_time - start_time
    return train_loss,epoch_duration


def train_kd_confid(model,train_month,log_file_path,if_kd,if_c,hype_kd,hype_c):
    student_model, teacher_model, optimizer, traindataloader1, traindataloader2, lr_scheduler,val_loader,test_loader,config,device,scaler,lr_scheduler_config,adj_mx=train_tools.init_train(model,train_month,log_file_path,type='kd')
    best_val_loss = float('inf')
    config['if_kd']=if_kd
    config['if_c']=if_c
    config['hype_kd'],config['hype_c']=hype_kd,hype_c
    criterion = torch.nn.L1Loss()
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    consecutive_no_decrease = 0
    max_consecutive_no_decrease = 10  # Adjust as needed
    classification=MTGNN_c(config,adj_mx).to(device)
    classification.load_state_dict( torch.load(os.path.join(config['model_dir'], f'classification_MTGNN+{config["dataset_name"]}_train_months_96' + 'final_model.pth')))
    # opt1=torch.optim.Adam(classification.parameters(),lr=0.005)
    for epoch in range(10):
        train_loss2,epoch_duration2=train_1_epoch_vanilla(traindataloader1,student_model,optimizer,config,device,criterion)
    teacher_model=copy.deepcopy(student_model)
    for epoch in range(config['num_epochs']):
        train_loss,epoch_duration=self_distill_with_confid(classification,student_model,teacher_model,traindataloader2,optimizer,config,device,epoch)
        train_loss2,epoch_duration2=train_1_epoch_vanilla(traindataloader1,student_model,optimizer,config,device,criterion)
        
        # train_loss2,epoch_duration2=Train_class.train_1_epoch_with_confid(traindataloader1,student_model,optimizer,config,device,criterion)
        val_loss=train_tools.validate_model(val_loader,student_model,scaler,config,device,criterion)
        # print(config['alpha2']*(100-epoch)/100 )
        train_tools.ema(student_model, teacher_model, config['alpha'])
        if 'lr_scheduler' in config and lr_scheduler_config['type'] == 'plateau':
            lr_scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_epoch = epoch
            consecutive_no_decrease = 0  # Reset the counter
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pth')
            torch.save(student_model.state_dict(), checkpoint_path)
        else:
            consecutive_no_decrease += 1
    
        if consecutive_no_decrease >= max_consecutive_no_decrease:
            if epoch>5:
                logging.info(f'Stopping training as validation loss did not improve for {max_consecutive_no_decrease} consecutive epochs.')
                print(f'best validation loss : {best_val_loss:.4f}')
                logging.info(f'best validation loss : {best_val_loss:.4f}')
                break
        
        logging.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}]: '
                    f'Train Loss: {train_loss:.4f}; '
                    f'Train Loss2: {train_loss2:.4f}; '
                    f'Validation Loss: {val_loss:.4f}; '
                    f'Epoch Time: {epoch_duration+epoch_duration2}; '
                    f'Learning Rate: {optimizer.param_groups[0]["lr"]}')


    model=student_model
    model.eval()




    best_model_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{best_model_epoch}.pth')
    model.load_state_dict(torch.load(best_model_path))
    rmse,mae=train_tools.test(model,test_loader,device,scaler)
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')


    logging.info(f'MAE: {mae:.4f}')
    logging.info(f'RMSE: {rmse:.4f}')



    # Optionally, you can save the trained model
    model_path = os.path.join(config['model_dir'], f'{config["model"]}+{config["dataset_name"]}_train_months_{config["train_months"]}_self_distill_confid_' + 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    return mae,rmse

    