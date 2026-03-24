import torch
from tqdm import tqdm
import numpy as np
from src.metric import dataset_by_indexes, dataset_gen

import torchvision.transforms.v2 as transforms

def get_transform():
    transform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0,5)),
        transforms.RandomAffine(degrees=0, translate=(0.021, 0.021), fill=0), # 5%
        transforms.Resize(size=(240,240), antialias=True)
        )


    return transform



def fit_by_name(cnn_model, train_names, num_epochs, optimizer, loss_function, batch_size, device):
  cnn_model.train()

  for epoch in range(num_epochs):
      np.random.shuffle(train_names)
      loop = tqdm(dataset_gen(train_names, batch_size, device), total=len(train_names)//batch_size)

      loss_val = 0.0
      train_running_correct = 0
      
      transform = get_transform()
  
      index = 0
      for i, (images, labels) in enumerate(loop):
          index = i
          with torch.no_grad():
            x = transform(images)

          x = x.requires_grad_()

          #print(x.shape)
          #forward 
          outputs = cnn_model(x)
          
          #print(outputs.shape)
          loss = loss_function(outputs, labels)

          preds = torch.round(outputs)

          train_running_correct += (preds == labels).sum().item() # и 0==0 и 1==1

          #backward 
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          loss_val += loss.item()

          # экономия памяти
          del images
          del x
          del labels
                                                                                                 # усреднить по батчу и по номеру 
          loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], accuracy= {round(train_running_correct/ (batch_size * (i+1)), 4)}, loss= {round(loss.item(), 4)}")
      print(f"\tEpoch {epoch+1}: mean_epoch_accuracy = {round(train_running_correct/ (batch_size * (index+1)), 4)}, mean_epoch_loss = {round(loss_val/(index+1), 4)}")
      print()

  cnn_model.eval()
  return cnn_model


def fit_by_indexes(cnn_model, dataset, train_indexes, num_epochs, optimizer, loss_function, batch_size, device):
  cnn_model.train()

  for epoch in range(num_epochs):
      np.random.shuffle(train_indexes)
      loop = tqdm(dataset_by_indexes(dataset, train_indexes, batch_size))

      loss_val = 0.0
      train_running_correct = 0
  
      index = 0
      for i, (images, labels) in enumerate(loop):
          index = i
          
          #print(images.shape, labels.shape)
          
          images = images.requires_grad_()

          #print(images.shape)
          #forward 
          outputs = cnn_model(images)
          
          #print(outputs.shape)
          loss = loss_function(outputs, labels)

          preds = torch.round(outputs)

          train_running_correct += (preds == labels).sum().item() # и 0==0 и 1==1

          #backward 
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          loss_val += loss.item()
                                                                                                 # усреднить по батчу и по номеру 
          loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], accuracy= {round(train_running_correct/ (batch_size * (i+1)), 4)}, loss= {round(loss.item(), 4)}")
      print(f"\tEpoch {epoch+1}: mean_epoch_accuracy = {round(train_running_correct/ (batch_size * (index+1)), 4)}, mean_epoch_loss = {round(loss_val/(index+1), 4)}")
      print()

  cnn_model.eval()
  return cnn_model