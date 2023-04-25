import numpy as np
import random
import yaml
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch

def unpickle(file):
   import pickle
   with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
   return dict

def read(path):
   dict = unpickle(path)
   filenames = dict[b'filenames']
   labels = dict[b'fine_labels']
   data = dict[b'data']
   return filenames, labels, data

def read2(valid_size, batch_size, num_workers):
   # define transformations for train
  train_transform = transforms.Compose([
      transforms.RandomRotation(30),
      transforms.RandomHorizontalFlip(p=.30),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  train_data = datasets.CIFAR100('./cifar100/', train=True,
                                download=True, transform=train_transform)
    # Dividing the training dataset further for validation set
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size * num_train))
  train_idx, valid_idx = indices[split:], indices[:split]

  # define samplers for obtaining training and validation batches
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
      sampler=train_sampler, num_workers=num_workers)
  valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
      sampler=valid_sampler, num_workers=num_workers)

  test_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  test_data = datasets.CIFAR100('./cifar100/', train=False,
                             download=True, transform=test_transform)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
  
  return train_loader, valid_loader, test_loader

def read_labels(path):
   dict = unpickle(path)
   labels_names = dict[b'fine_label_names']
   return labels_names

def split_validation(train_filenames, train_labels, train_data):
   rand_idx = random.sample(range(0, len(train_data)), int(0.1 * len(train_data)) )
   valid_data = [train_data[idx] for idx in rand_idx]
   valid_filenames = [train_filenames[idx] for idx in rand_idx]
   valid_labels = [train_labels[idx] for idx in rand_idx]

   train_filenames = np.delete(train_filenames, rand_idx, 0)
   train_labels = np.delete(train_labels, rand_idx, 0)
   train_data = np.delete(train_data, rand_idx, 0)

   return train_filenames, train_labels, train_data, valid_filenames, valid_labels, valid_data

def get_paths():
   with open('paths.yaml', 'r') as file:
       prime_service = yaml.safe_load(file)
   return prime_service['train'], prime_service['test'], prime_service['meta']