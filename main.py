from train.train import train
from train.kd_with_confid import train_kd_confid
import os
import logging
import torch
log_dir='./logs/nycbike'
train_month=96
model='STGCN'
os.makedirs(log_dir, exist_ok=True)
kd,c=1,0.2
mae,rmse=train(model,train_month,log_file_path)
mae,rmse=train_kd_confid(model,train_month,log_file_path,True,True,kd,c)
