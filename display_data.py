
import matplotlib.pyplot as plt
import numpy as np 
import torch

def data_distribution( data ):
    count = []
    for i in range(100):
        count.append(np.count_nonzero( data == i ))
    classes = np.arange(100)
    plt.bar(classes, count)
    # Add labels
    plt.title('Data Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
def data_distribution2( data_loader ):
    labels = []
    for _, target in data_loader:
        labels.append(target)
    labels = torch.cat(labels)
    result = labels.numpy()
    print(labels)
    data_distribution(result)
def show_images(img, labels, label_names):
    images = img.reshape(len(img),3,32,32).transpose(0,2,3,1)
    rows, columns = 5, 5
    imageId = np.random.randint(0, len(images), rows * columns)
    images = images[imageId]
    labels = [labels[i] for i in imageId]
    fig=plt.figure(figsize=(10, 10))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title("{}"
            .format(label_names[labels[i-1]].decode('UTF-8')))
    plt.show()

def show_images2( data_loader, label_names ):
         # obtain one batch of test images
        dataiter = iter(data_loader)
        images, labels = next(dataiter)
        images.numpy()
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            image = images.cpu()[idx]
            plt.imshow( (np.transpose(image, (1,2,0))))
            ax.set_title("{}".format(label_names[labels[idx]].decode('UTF-8')))
        plt.show()