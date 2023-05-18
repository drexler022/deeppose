import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import MPII_Dataset

import warnings
warnings.filterwarnings('ignore')

import Dataset
import AlexNet
import ResNet
import VggNet

# for reproducible results
torch.manual_seed(0)
# torch.cuda.manual_seed(0) #called internally from torch.manual_seed()
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(epochs, model, train_dl, val_dl, optimizer, criterion, train_size, val_size, is_lsp=True):
  train_loss_lst, val_loss_lst, batch_epoch_loss_lst = [], [], []
  pck_lst_1 = []
  pck_lst_2 = []
  for e in range(epochs):
    file_name = "model" + str(e)
    save_path = "./Res/" + file_name + ".pth"
    
    train_loss, val_loss = 0, 0
    
    # Training
    model.train()
    for batch_idx,(batch_imgs, batch_labels) in enumerate(train_dl):
      optimizer.zero_grad()
      batch_imgs,batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      #print(batch_labels[0])
      batch_labels = batch_labels[:, :2, :].permute((0,2,1))
      #print(batch_labels[0])
      output = output.view(batch_labels.shape)
      
      loss = criterion(output,batch_labels.float())
      batch_epoch_loss_lst.append(loss.item())
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    
    train_loss_lst.append(train_loss/train_size)
    
    # Validation
    model.eval()
    pck_batch_1 = []
    pck_batch_2 = []
    for batch_idx,(batch_imgs,batch_labels) in enumerate(val_dl):
      sample = batch_imgs[0]
      batch_imgs, batch_labels = batch_imgs.float().to(device),batch_labels.to(device)
      output = model(batch_imgs)
      batch_labels = batch_labels[:, :2, :].permute((0,2,1))
      # Reshape the outputs of shape (batch_size x 28) -> (batch_size x 14 x 2)
      output = output.view(batch_labels.shape)
      
      # Visualization
      if batch_idx%10 == 0:
          plt.imshow(sample)
          for i in range(0,14):
              prediction = (output[0][i]+0.5)*196
              groundtruth = (batch_labels[0][i]+0.5)*196
              prediction = prediction.cpu().detach().numpy()
              groundtruth = groundtruth.cpu().detach().numpy()
              #print(prediction,groundtruth,'\n')
          
              plt.scatter(prediction[0],prediction[1],c='r')
              #plt.scatter(groundtruth[0],groundtruth[1],c='b')
          plt.show()
          
      # Evaluation PCK
      if is_lsp == True:
          hip = 2
          shoulder = 9
      if is_lsp == False:
          hip = 2
          shoulder = 11
      threshold1 = 1.0
      threshold2 = 0.5
      pck_img_1 = []
      pck_img_2 = []
      for j in range(0,len(batch_imgs)):
          correct1 = 0
          correct2 = 0
          # neck to hip distance
          right_hip = (output[j][hip]+0.5)*196
          left_shoulder = (output[j][shoulder]+0.5)*196
          right_hip = right_hip.cpu().detach().numpy()
          left_shoulder = left_shoulder.cpu().detach().numpy()
          norm = np.linalg.norm(left_shoulder-right_hip)
          
          for k in range(0,14):
              pre = (output[j][k]+0.5)*196
              gt = (batch_labels[j][k]+0.5)*196
              pre = pre.cpu().detach().numpy()
              gt = gt.cpu().detach().numpy()
              dist = np.linalg.norm(pre-gt)/norm
              #print(dist)
              if dist <= threshold1:
                  correct1 += 1
              if dist <= threshold2:
                  correct2 += 1
          #print(correct/14)        
          pck_img_1.append(correct1/14)
          pck_img_2.append(correct2/14)

      #pck_batch_1.append(sum(pck_img_1)/len(batch_imgs))
      
      loss = criterion(output, batch_labels.float())
      val_loss += loss.item()
      
    val_loss_lst.append(val_loss/val_size)
    pck_lst_1.append(sum(pck_img_1)/16)
    pck_lst_2.append(sum(pck_img_2)/16)

    print("[{}/{}]: Train loss={:2.4f}, Validation loss={:2.4f}, PCK1.0={:2.4f},PCK0.5={:2.4f}".format(e+1,epochs,train_loss_lst[-1],val_loss_lst[-1],pck_lst_1[-1],pck_lst_2[-1]))
    
    if train_loss_lst[-1]<=0.25:
      for param in optimizer.param_groups:
        param["lr"]=5e-4

    if train_loss_lst[-1]<=0.15:
      for param in optimizer.param_groups:
        param["lr"]=1e-4
    #torch.save(model,save_path)
  return train_loss_lst, val_loss_lst, batch_epoch_loss_lst

def main():
  lsp_dataset = Dataset.LSP_Dataset()
  dataset = torch.utils.data.ConcatDataset([lsp_dataset])
  
  #mpii_Dataset = MPII_Dataset.MPII_Dataset()
  #dataset = torch.utils.data.ConcatDataset([mpii_Dataset])

  batch_size = 16
  total = len(dataset)
  train_size, val_size = int(total*0.8), int(total*0.2)

  lengths = [train_size, val_size]
  train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, lengths)

  train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dl   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

  #model = ResNet.DeepPose(14).float().to(device)
  model = AlexNet.DeepPose().float().to(device)

  criterion = nn.MSELoss(reduction="sum")
  
  optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
  
  #optimizer = torch.optim.Adagrad(model.parameters(),lr=1e-3)

  train_loss_lst, val_loss_lst, batch_epoch_loss_lst = train( epochs=80, 
                                                              model=model, 
                                                              train_dl=train_dl, 
                                                              val_dl=val_dl, 
                                                              optimizer=optimizer, 
                                                              criterion=criterion, 
                                                              train_size=train_size, 
                                                              val_size=val_size,
                                                              is_lsp=True)

  plt.plot(train_loss_lst,color='r')
  plt.plot(val_loss_lst,color='g')
  plt.show()

if __name__ == "__main__":
  main()