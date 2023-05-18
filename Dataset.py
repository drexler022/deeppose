import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.transform

import torch

class LSP_Dataset(torch.utils.data.Dataset):
  
  def __init__(self, path="./lsp_dataset"): 
    self.path = path

    imgs_list = sorted(os.listdir(os.path.join(path, "images")))
    
    self.joint_data = scipy.io.loadmat(os.path.join(path, "joints.mat"))["joints"]
    
    self.dataset_size = self.joint_data.shape[2]
    
    assert len(imgs_list) == self.dataset_size

    self.max_h, self.max_w = 196, 196

    self.array_of_images = np.empty([self.dataset_size, self.max_h, self.max_w, 3], dtype=float)
    self.array_of_labels = np.empty([self.dataset_size, 3, 14], dtype=float) #N x (X,Y) x (14 joints)
    
    
    for file_idx, file_name in enumerate(imgs_list):
      img, labels = self.scale_and_pad( plt.imread(os.path.join(path, "images", file_name)), self.joint_data[:2,:,file_idx])
      
      self.array_of_images[file_idx] = img
      self.array_of_labels[file_idx, :2, :] = labels
      self.array_of_labels[file_idx, 2, :]  = self.joint_data[2, :, file_idx]
    
    print(f"Built Dataset: found {self.__len__()} image-target pairs")

  
  def scale_and_pad(self, img, labels):
    scale_factor = self.max_h/max(*img.shape)

    scaled_img = skimage.transform.rescale(img, scale=scale_factor, multichannel=True)

    img_h, img_w, _   = scaled_img.shape
    padded_scaled_img = np.zeros([self.max_h, self.max_w, 3])
    start_h, start_w  = int((self.max_h - img_h)/2), int((self.max_w - img_w)/2)

    padded_scaled_img[start_h:start_h + img_h, start_w:start_w + img_w, :] = scaled_img
    padded_scaled_labels = (labels*scale_factor + np.array([[start_w], [start_h]]))/self.max_h - 0.5
    return padded_scaled_img, padded_scaled_labels
  
  
  def __getitem__(self,idx):
    return self.array_of_images[idx], self.array_of_labels[idx]

  
  def __len__(self):
    return self.array_of_images.shape[0]
  

if __name__ == "__main__":
  dataset = LSP_Dataset()

