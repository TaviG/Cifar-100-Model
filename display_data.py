
import matplotlib.pyplot as plt
import numpy as np

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