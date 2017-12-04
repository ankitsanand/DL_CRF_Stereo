from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import numpy as np


#import torch.optim as optim
#from torch.optim import lr_scheduler
#from lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import MiddleburyDataset
from dataset import KittyDataset
from stereocnn1 import StereoCNN
from compute_error import compute_error

parser = argparse.ArgumentParser(description='StereoCNN model')
parser.add_argument('-k', "--disparity", type=int, default=256)
parser.add_argument('-ul', "--unary-layers", type=int, default=7)
parser.add_argument('-data', "--dataset", type=str, default="Kitty")

parser.add_argument('-lr', "--learning-rate", type=float, default=1e-2)
parser.add_argument('-m', "--momentum", type=float, default=0.8)
parser.add_argument('-b', "--batch-size", type=int, default=1)
parser.add_argument('-n', "--num-epoch", type=int, default=1000)
parser.add_argument('-v', "--verbose", type=bool, default=True)
parser.add_argument('-ms', "--model-file", type=str, default="model1.pkl")
parser.add_argument('-ls', "--log-file", type=str, default="logs1.txt")
#parser.add_argument('-md',"--mode",type=str,default="val")
args = parser.parse_args()

# Global variables
k = args.disparity
unary_layers = args.unary_layers
dataset=args.dataset
learning_rate = args.learning_rate
momentum = args.momentum
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = 1
verbose=args.verbose


def print_grad(grad):
  print("Grad_Max")
  print(torch.max(grad))

# DATA_DIR = '/Users/ankitanand/Desktop/Stereo_CRF_CNN/Datasets/Kitty/data_scene_flow'
#DATA_DIR = '/Users/Shreyan/Downloads/Datasets/Kitty/data_scene_flow'
#DATA_DIR = '/home/ankit/Stereo_CNN_CRF/Datasets/'
DATA_DIR = '/scratch/cse/phd/csz138105/Datasets/'

model_save_path = os.path.join("/scratch/cse/phd/csz138105/experiments", args.model_file)
log_file = open(os.path.join("/scratch/cse/phd/csz138105/experiments", args.log_file), "w")


def main():
  global learning_rate
  if(dataset=="Middlebury"):
    train_set = MiddleburyDataset(DATA_DIR,'split_train')
    val_set = MiddleburyDataset(DATA_DIR,'split_val')
  else:
    train_set = KittyDataset(DATA_DIR,'split_train')
    val_set = KittyDataset(DATA_DIR,'split_val')
  
  train_data_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  
  val_data_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)
  if(True):
    model = torch.load(model_save_path)
  else:
    model = StereoCNN(unary_layers, k)
  loss_fn = nn.CrossEntropyLoss(size_average=True,ignore_index=-1)
  
  if torch.cuda.is_available():

    model = model.cuda()
  for epoch in range(num_epoch):
    if(epoch%5==0):
      learning_rate =learning_rate * 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.01)

    if(verbose):
      print("Epoch",epoch)
      train_error = 0
      val_error = 0
    torch.save(model, "/scratch/cse/phd/csz138105/experiments/kitty_val_model_fast_0.01_0.8.pkl")
    for mode in ['train','val']:
      if(verbose):
        print(mode)
      if(mode=='train'):
        data_loader=train_data_loader
        
      else:
        data_loader=val_data_loader
        
      for num, data in enumerate(data_loader):
        if(verbose):
          print(" Image",num)  
        left_img, right_img, labels = data
        # No clamping might be dangerous
        labels=labels.clamp_(-1,k)

        optimizer.zero_grad()
        if torch.cuda.is_available():
          left_img = left_img.cuda()
          right_img = right_img.cuda()
          labels=labels.cuda()
        if(mode == 'train'):
          left_img = Variable(left_img,requires_grad=True,volatile=False)
          right_img = Variable(right_img,requires_grad=True,volatile=False)
        else:
          left_img = Variable(left_img,requires_grad=False,volatile=True)
          right_img = Variable(right_img,requires_grad=False,volatile=True)
        labels = Variable(labels.type('torch.cuda.LongTensor'))
        left,right= model(left_img,right_img)
        b,d,r,c = left.size()
        if(mode == 'train'):
          corr=Variable(torch.zeros(b,k,r,c).cuda())
        else:
          corr=Variable(torch.zeros(b,k,r,c).cuda(),volatile=True)
        for i in range(k):
          corr[:,i,:,k:c] = (left[:,:,:,k:c]*right[:,:,:,k-i:c-i]).sum(1)
        y_pred_perm = corr.permute(0,2,3,1)
        y_pred_ctgs = y_pred_perm.contiguous()
        y_pred_flat= y_pred_ctgs.view(-1,k)
        y_vals, y_labels = torch.max(y_pred_ctgs, dim=3)
        loss = loss_fn(y_pred_flat, labels.view(-1))
        if(mode=='train'):
          loss.backward()
          optimizer.step()

          #scheduler.step()
          train_error = train_error + compute_error(epoch, i, log_file, loss.data[0], y_labels.data.cpu().numpy(), labels.data.cpu().numpy())
        else:
          val_error =val_error + compute_error(epoch, i, log_file, loss.data[0], y_labels.data.cpu().numpy(), labels.data.cpu().numpy())
        # error = 0

    log = "Epoch" +str(epoch) + "Train_error"+ str(train_error/160) + "Val_error" +str(val_error/40)
    if(verbose):
      print(log)
 
    log_file.write(log+"\n")
    log_file.flush()
      
    

if __name__ == "__main__":
  main()
