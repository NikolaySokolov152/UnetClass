import torch
from tqdm import tqdm

def fit(cnn_model, train_names, num_epochs, optimizer, loss_function):
  cnn_model.train()

  for epoch in range(num_epochs):
      np.random.shuffle(train_names)
      loop = tqdm(dataset_gen(train_names, batch_size), total=len(train_names)//batch_size)

      loss_val = 0.0
      train_running_correct = 0
  
      index = 0
      for i, (images, labels) in enumerate(loop):
          index = i
          
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

          # экономия памяти
          del images
          del labels
                                                                                                 # усреднить по батчу и по номеру 
          loop.set_description(f"Epoch [{epoch+1}/{num_epochs}], accuracy= {round(train_running_correct/ (batch_size * (i+1)), 4)}, loss= {round(loss.item(), 4)}")
      print(f"Epoch {epoch+1}: mean_epoch_accuracy = {round(train_running_correct/ (batch_size * (index+1)), 4)}, mean_epoch_loss = {round(loss_val/(index+1), 4)}")
      print()

  cnn_model.eval()
  return cnn_model