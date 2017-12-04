from torch.utils.data import Dataset
import collections
import os
#import scipy.misc as m
import numpy as np
from load_pfm import load_pfm
from PIL import Image

class KittyDataset(Dataset):
  """Kitty dataset"""
  def __init__(self, root, split='split_train', transform=None):
    self.root = root+"/Kitty/data_scene_flow"
    self.split = split
    self._transform = transform
    self.files = collections.defaultdict(list)
    #for split in ['split_train','split_val']:
    self.files[split] = os.listdir(self.root + '/' + split+'/myimage_2')


  def __len__(self):
    return len(self.files[self.split])

  def __getitem__(self, i):
    """
      Get the ith item from the dataset
      Return : left_img, right_img, target
    """
   
    img_name = self.files[self.split][i]
    left_img_path = self.root + '/' + self.split + '/myimage_2/' + img_name
    right_img_path= self.root + '/' + self.split + '/myimage_3/' + img_name 
    lbl_path = self.root + '/' + self.split + '/disp_noc_0/' + img_name

    #left_img = m.imread(left_img_path)
    left_img  = Image.open(left_img_path)
    left_img = np.array(left_img, dtype=np.float32)
    # left_img = np.pad(left_img, ((0,376-left_img.shape[0]), (0,1242-left_img.shape[1]), (0,0)), 'constant', constant_values=0)
    left_img = left_img.transpose(2,0,1)

    right_img = Image.open(right_img_path)
    right_img = np.array(right_img,dtype=np.float32)
    # right_img = np.pad(right_img, ((0,376-right_img.shape[0]), (0,1242-right_img.shape[1]), (0,0)), 'constant', constant_values=0)
    right_img = right_img.transpose(2,0,1)
    
    # Normalizing images
    left_img = (left_img - left_img.mean())/left_img.std()
    right_img = (right_img - right_img.mean())/right_img.std()

    lbl = Image.open(lbl_path)
    lbl = np.array(lbl, dtype=np.int64)
    # lbl = np.pad(lbl, ((0,376-lbl.shape[0]), (0,1242-lbl.shape[1])), 'constant', constant_values=0)
    
    
    lbl[lbl<=0]=-256 # Set the invalid Pixels

    
    lbl = lbl/256
    lbl= lbl.astype(int)
    

    if self._transform:
      left_img, right_img, lbl = self.transform(img, lbl)

    return left_img, right_img, lbl
    

class MiddleburyDataset(Dataset):
  """Middlebury dataset"""

  def __init__(self, root, split='splitTraining', Resolution='Q',transform=None):
      self.root = root + "Middlebury/Eval3/"
      self.split = split
      self._transform = transform
      self.files = collections.defaultdict(list)
      self.Resolution=Resolution
      for split in ['splitTraining', 'splitVal']:
          print(self.root + split + Resolution)
          #os.system("rm "+self.root + split + Resolution+"/.DS_Store ")
          self.files[split] = os.listdir(self.root + split + Resolution )



  def __len__(self):
      return len(self.files[self.split])

  def __getitem__(self, i):
      """
        Get the ith item from the dataset
        Return : left_img, right_img, target
      """
      img_name = self.files[self.split][0]
      left_img_path = self.root  + self.split + self.Resolution + '/' + img_name+ '/' + 'im0.png'
      right_img_path = self.root + self.split + self.Resolution + '/' + img_name+ '/' + 'im1_rectified.png'

      lbl_path = self.root + "GT-" + self.split + self.Resolution + '/' + img_name+ '/' +'disp0GT.pfm'
      left_img = Image.open(left_img_path)
      left_img = np.array(left_img, dtype=np.float32)
      left_img = left_img.transpose(2, 0, 1)

      right_img =  Image.open(right_img_path)
      right_img = np.array(right_img, dtype=np.float32)
      right_img = right_img.transpose(2, 0, 1)

      # Normalizing images
      left_img = (left_img - left_img.mean()) / left_img.std()
      right_img = (right_img - right_img.mean()) / right_img.std()

      lbl = load_pfm(lbl_path)
      lbl=lbl[0]
      lbl= lbl.astype(int)

     
      lbl[lbl==np.inf]=-1

      
      if self._transform:
          left_img, right_img, lbl = self.transform(img, lbl)

      return left_img, right_img, lbl
