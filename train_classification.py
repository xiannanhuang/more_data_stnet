import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from Dataset import STDataset,Auxility_dataset
import os
from Dataset import Auxility_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler,random_split
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

def train_classfication(model,train_month,log_file_path):
    with open(r'models\config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    config['model'],config['train_months']=model,train_month
    # Configure logging
    log_dir = config['log_dir']  # You can specify the directory for log files
    os.makedirs(log_dir, exist_ok=True)
    # log_file_path = os.path.join(log_dir, '{}_training_month_{}_self_distill_vanilla.log'.format(config['model'],config['train_months']))
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.info('{}_training_month_{}_self_{}.log'.format(config['model'],config['train_months'],type))
    logging.info('Training classiication {} model for {} months'.format(config['model'],config['train_months']))

    # Check if a GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets(config['data_dir'], config)
    train_dataset=Auxility_dataset(train_dataset.data,classindex=config['warm_up_day_num']*24,config=config)
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    # Create data loaders
    
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
    logging.info(f"Total valid_gird: {len(valid_gird[0])}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCELoss()
    consecutive_no_decrease = 0
    max_consecutive_no_decrease = 10  # Adjust as needed
    # Learning rate scheduler based on the configuration
    if 'lr_scheduler' in config:
        lr_scheduler_config = config['lr_scheduler']
        if lr_scheduler_config['type'] == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_config['step_size'], gamma=lr_scheduler_config['gamma'])
        elif lr_scheduler_config['type'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_config['factor'], patience=lr_scheduler_config['patience'], verbose=True)

    # Training loop
    best_val_loss = 100

    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader =DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        start_time = datetime.now()
        correct1,total1=0,0
        for batch in train_loader:
            inputs, targets = batch
            inputs=inputs+torch.randn_like(inputs)*0.5

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            inputs, targets = inputs.to(device), targets.to(device)
            outputs =nn.functional.sigmoid(model(inputs)[:,-1,-1,-1])

            # Calculate loss
            loss = criterion(outputs, targets)+abs(outputs-0.5).mean()

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.tensor(outputs>0.5,dtype=torch.int32)
            correct1 += (predicted == targets).sum().item()
            total1+=outputs.shape[0]

        # Calculate average training loss for this epoch
        train_loss /= len(train_loader)
        train_acc= correct1/total1

        # Validation
        model.eval()
        val_loss = 0.0
        correct2,total2=0,0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = nn.functional.sigmoid(model(inputs)[:,-1,-1,-1])
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                predicted = torch.tensor(outputs>0.5,dtype=torch.int32)
                correct2 += (predicted == targets).sum().item()
                total2+=outputs.shape[0]

        # Calculate average validation loss for this epoch
        val_loss /= len(val_loader)
        val_acc = correct2/total2

      

        # Learning rate scheduler step (if applicable)
        if 'lr_scheduler' in config and lr_scheduler_config['type'] == 'plateau':
            lr_scheduler.step(-val_acc)

        # Save the model if validation loss is improved
        if -val_acc < best_val_loss:
            best_val_loss = -val_acc
            best_model_epoch = epoch
            consecutive_no_decrease = 0  # Reset the counter
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        else:
            consecutive_no_decrease += 1
       
        if consecutive_no_decrease >= max_consecutive_no_decrease:
            logging.info(f'Stopping training as validation loss did not improve for {max_consecutive_no_decrease} consecutive epochs.')
            break
        # Calculate epoch duration
        end_time = datetime.now()
        epoch_duration = end_time - start_time

        # Log training progress
        # Log to the file using the logging module
        logging.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}]: '
                    f'Train Loss: {train_loss:.3f}; '
                    f'Validation Loss: {val_loss:.3f}; '
                    f'train_acc: {train_acc:.3f}; '
                    f'val_acc: {val_acc:.3f}; '
                    f'Epoch Time: {epoch_duration}; '
                    f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
    best_model_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{best_model_epoch}.pth')
    model.load_state_dict(torch.load(best_model_path))
    model_path = os.path.join(config['model_dir'], f'classification_{config["model"]}+{config["dataset_name"]}_train_months_{config["train_months"]}' + 'final_model.pth')
    torch.save(model.state_dict(), model_path)

# log_dir='./logs/nyc_bike'
# model,train_month='MTGNN',96
# os.makedirs(log_dir, exist_ok=True)
# train_classfication(model,train_month,os.path.join(log_dir, 'classfication__test_{}_training_month_{}_.log'.format(model,train_month)))
