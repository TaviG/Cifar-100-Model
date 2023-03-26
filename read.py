import numpy as np
import random
import yaml

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

def read_labels(path):
   dict = unpickle(path)
   labels_names = dict[b'fine_label_names']
   return labels_names

def split_validation(train_filenames, train_labels, train_data):
   rand_idx = random.sample(range(0, len(train_data)), int(0.2 * len(train_data)) )
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