import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from models.GWNET import gwnet
from models.MTGNN import MTGNN
from models import STGCN,AGCRN,DCRNN
from tools.data_tools import load_data, get_datasets, expand_adjacency_matrix
from Dataset import STDataset
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging


def train(model,train_month,log_file_path):
# Load project configuration from config.yaml
    with open(r'models\config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    config['model'],config['train_months']=model,train_month
    # Configure logging
    # log_dir = config['log_dir']  # You can specify the directory for log files
 

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('{}_training_month_{}_.log'.format(config['model'],config['train_months']))



    # Check if a GPU is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get datasets and scaler
    train_dataset, val_dataset, test_dataset, scaler,valid_gird = get_datasets(config['data_dir'], config)
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    adj_mx =  expand_adjacency_matrix(np.load(config['adj_mx_file']), config['num_nodes'])[valid_gird][:,valid_gird][:,0,:]
    config['num_nodes']=len(valid_gird[0])
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

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.L1Loss()
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
    best_val_loss = float('inf')
    training_logs = []
    for name, param in model.named_parameters():
        logging.info(f"Parameter Name: {name}\t Shape: {param.shape}")

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_trainable_params}")
    logging.info(f"Total valid_gird: {len(valid_gird[0])}")
    print(f"Total valid_gird: {len(valid_gird[0])}")
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0

        start_time = datetime.now()

        for batch in train_loader:
            inputs, targets = batch[:2]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*scaler.std

        # Calculate average training loss for this epoch
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[:2]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()*scaler.std

        val_loss /= len(val_loader)


        if 'lr_scheduler' in config and lr_scheduler_config['type'] == 'plateau':
            lr_scheduler.step(val_loss)

        # Save the model if validation loss is improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_epoch = epoch
            consecutive_no_decrease = 0  # Reset the counter
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        else:
            consecutive_no_decrease += 1
        if consecutive_no_decrease >= max_consecutive_no_decrease:
            if epoch>5:
                logging.info(f'Stopping training as validation loss did not improve for {max_consecutive_no_decrease} consecutive epochs.')
                print(f'best validation loss : {best_val_loss:.4f}')
                logging.info(f'best validation loss : {best_val_loss:.4f}')
                break
        # Calculate epoch duration
        end_time = datetime.now()
        epoch_duration = end_time - start_time

        # Log training progress
        training_log = {
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Epoch Time': epoch_duration.total_seconds(),
            'Learning Rate': optimizer.param_groups[0]['lr']
        }
        training_logs.append(training_log)

        # Log to the file using the logging module
        logging.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}]: '
                    f'Train Loss: {train_loss:.4f}; '
                    f'Validation Loss: {val_loss:.4f}; '
                    f'Epoch Time: {epoch_duration}; '
                    f'Learning Rate: {optimizer.param_groups[0]["lr"]}')

    # Testing the model
    model.eval()


    rmse = 0.0
    mae = 0.0
    num_samples = 0
    best_model_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{best_model_epoch}.pth')
    model.load_state_dict(torch.load(best_model_path))
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[:2]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
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



    # Calculate MAE and RMSE



    logging.info(f'MAE: {mae:.4f}')
    logging.info(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')



    # Optionally, you can save the trained model
    model_path = os.path.join(config['model_dir'], f'{config["model"]}+{config["dataset_name"]}_train_months_{config["train_months"]}' + 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    return mae,rmse

