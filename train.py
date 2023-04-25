import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        image = self.transform(np.array(image).astype('uint8'))
        return image, label
    def __len__(self):
        return len(self.labels)

def train(x_train, y_train, x_valid, y_valid):

  model_transfer = models.googlenet(weights='GoogLeNet_Weights.IMAGENET1K_V1')

  clf = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01,
                    max_iter = 50)
  



  train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=.30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  x_train = x_train.reshape(len(x_train),3,32,32)
  x_valid = np.array([np.array(xi) for xi in x_valid]).reshape(len(x_valid),3,32,32)
  x_train = np.transpose(x_train, (0,2,3,1))
  x_valid = np.transpose(x_valid, (0,2,3,1))

  x_train_mlp = x_train.reshape(len(x_train), 3072)
  clf.fit(x_train_mlp ,y_train)
  
  

  #y_train = y_train.reshape(len(y_train),3,32,32)
  #y_valid = np.array([np.array(xi) for xi in y_valid]).reshape(len(y_valid),3,32,32)
  y_valid = np.array(y_valid)
  x_train = torch.Tensor(x_train)
  x_valid = torch.Tensor(x_valid)
  y_train = torch.Tensor(y_train)
  y_valid = torch.Tensor(y_valid)
  train_data = Dataset(x_train, y_train, train_transform)
  valid_data = Dataset(x_valid, y_valid, train_transform)

#   train_data = datasets.CIFAR100('./cifar100/', train=True,
#                                 download=True, transform=train_transform)

  # number of subprocesses to use for data loading
  num_workers = 0
  # how many samples per batch to load
  batch_size = 64
  # percentage of training set to use as validation
#   valid_size = 0.1

#   # # Dividing the training dataset further for validation set
#   num_train = len(train_data)
#   indices = list(range(num_train))
#   np.random.shuffle(indices)
#   split = int(np.floor(valid_size * num_train))
#   train_idx, valid_idx = indices[split:], indices[:split]

  # # define samplers for obtaining training and validation batches
#   train_sampler = SubsetRandomSampler(train_idx)
#   valid_sampler = SubsetRandomSampler(valid_idx)

#   train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#       sampler=train_sampler, num_workers=num_workers)
#   valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
#       sampler=valid_sampler, num_workers=num_workers)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
       num_workers=num_workers)
  valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
       num_workers=num_workers)

#   for param in model_transfer.parameters():
#       param.requires_grad = False
      
 # in_features = model_transfer.fc.in_features
 # model_transfer.fc = nn.Linear(in_features, 100)

  use_gpu = torch.cuda.is_available()
  if use_gpu:
      model_transfer = model_transfer.cuda()

  criterion = nn.CrossEntropyLoss().cuda()
  model_transfer_grad_paramaters = filter(lambda p: p.requires_grad, model_transfer.parameters())
  # optimizer = torch.optim.Adam(model_transfer_grad_paramaters, lr=0.001)
  optimizer = torch.optim.Adam(model_transfer.parameters(), lr=0.01)
  n_epochs = 50

  valid_loss_min = np.Inf # track change in validation loss
  #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)
  #scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.94)
  for epoch in range(1, n_epochs+1):

      # keep track of training and validation loss
      train_loss = 0.0
      valid_loss = 0.0
      correct_train = 0.0
      correct_valid = 0.0
      ###################
      # train the model #
      ###################
      model_transfer.train()
      for data, target in train_loader:
          target = target.type(torch.LongTensor)
          # move tensors to GPU if CUDA is available
          if use_gpu:
              data, target = data.cuda(), target.cuda()
          # clear the gradients of all optimized variables
          optimizer.zero_grad()
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model_transfer(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()
          # perform a single optimization step (parameter update)
          optimizer.step()
          # update training loss
          train_loss += loss.item()*data.size(0)

          _, preds_tensor = torch.max(output, 1)
          preds = np.squeeze(preds_tensor.numpy()) if not use_gpu else np.squeeze(preds_tensor.cpu().numpy())


          correct_train += ( preds_tensor == target ).float().sum()
      #scheduler.step()   
      after_lr = optimizer.param_groups[0]["lr"]
      print(after_lr)
      ######################    
      # validate the model #
      ######################
      model_transfer.eval()
      for data, target in valid_loader:
          target = target.type(torch.LongTensor)
          # move tensors to GPU if CUDA is available
          if use_gpu:
              data, target = data.cuda(), target.cuda()
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model_transfer(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # update average validation loss 
          valid_loss += loss.item()*data.size(0)
          _, preds_tensor = torch.max(output, 1)
          correct_valid += ( preds_tensor == target ).float().sum()
      
      # calculate average losses
      train_loss = train_loss/len(train_loader.sampler)
      valid_loss = valid_loss/len(valid_loader.sampler)
      acc_train = 100 * correct_train / len(train_loader.sampler)
      acc_valid = 100 * correct_valid / len(valid_loader.sampler)
      # print training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
          epoch, train_loss, valid_loss, acc_train, acc_valid))
      
      # save model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model_transfer.state_dict(), 'model_transfer_cifar.pt')
          valid_loss_min = valid_loss
  return model_transfer, clf


def evaluate_model(model, x_test, y_test, classes, clf):
  # number of subprocesses to use for data loading
  num_workers = 0
  # how many samples per batch to load
  batch_size = 64
   # define transformations for test
  # for test we dont need much of augmentations other than converting to tensors and normalizing the pictures
  use_gpu = torch.cuda.is_available()
  test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  
#   test_data = datasets.CIFAR100('./cifar100/', train=False,
#                              download=True, transform=test_transform)
  x_test = x_test.reshape(len(x_test),3,32,32)
  x_test = np.transpose(x_test, (0,2,3,1))
  x_test = np.float32(x_test)
  y_test = np.float32(y_test)
  x_test = torch.Tensor(x_test)
  y_test = torch.Tensor(y_test)
  test_data = Dataset(x_test, y_test, transform = test_transform)
  

  x_test_mlp = x_test.reshape(len(x_test),3072)
  ypred=clf.predict(x_test_mlp)
  print(accuracy_score(y_test,ypred),'%')

  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
  
  correct = 0
  for data, target in test_loader:
      target = target.type(torch.LongTensor)
      if use_gpu:
              data, target = data.cuda(), target.cuda()
      output = model(data)
      # convert output probabilities to predicted class
      _, preds_tensor = torch.max(output, 1)
      preds = np.squeeze(preds_tensor.numpy()) if not use_gpu else np.squeeze(preds_tensor.cpu().numpy())
      correct += ( preds_tensor == target ).float().sum()
  acc = correct * 100 / len(test_loader.sampler)

  # obtain one batch of test images
  dataiter = iter(test_loader)
  images, labels = next(dataiter)
  images.numpy()
  labels.numpy()
  labels = labels.int()
  images = images.int()
  model.to('cpu')
  # get sample outputs
  output = model(images)
  # convert output probabilities to predicted class
  _, preds_tensor = torch.max(output, 1)
  preds = np.squeeze(preds_tensor.numpy()) if not use_gpu else np.squeeze(preds_tensor.cpu().numpy())
  # fig = plt.figure(figsize=(25, 4))
  # for idx in np.arange(20):
  #   ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
  #   image = images.cpu()[idx]
  #   plt.imshow( (np.transpose(image, (2,1,0))))
  #   ax.set_title("{} ({})".format(classes[preds[idx]].decode('UTF-8'), classes[labels[idx]].decode('UTF-8')),
  #                color=("green" if preds[idx]==labels[idx].item() else "red"))
  # plt.show()
  return acc