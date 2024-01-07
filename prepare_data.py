# import library

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import os




# set the super parameters
batch_size_train = 32
batch_size_test = 10


def prepare_data():
    """
    Prepare the MNIST dataset for training and testing.

    Returns:
    train_loader (DataLoader): DataLoader object for training data.
    test_loader (DataLoader): DataLoader object for testing data.
    """
    # 1.data preparation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    # 2.load train and test data
    trainset = datasets.MNIST(
        "./data/MNIST_data/", download=True, train=True, transform=transform
    )
    testset = datasets.MNIST(
        "./data/MNIST_data/", download=True, train=False, transform=transform
    )

    # 3.create data loader
    train_loader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader

def save_img(imgs, labels):
    # save image named ordered label in imgs,dont use plt
    # imgs: a batch of images
    # labels: a batch of labels
     for i, item in enumerate(zip(imgs, labels)):
        img, label = item
        print("original data:",img)
        print("shape of original data:",img.shape)
        print("shape of label :",label.shape)
        # disnormalization
        # img = img * 0.5 + 0.5
        print("transformed data:",img)
        img = img[0].numpy()
        array = (img.reshape((28, 28)) * 255).astype(np.uint8)

        img = Image.fromarray(array, 'L')

        label = label.cpu().numpy()
        img_path = './imgs/' + str(label) + '/' + str(i) + '.jpg'
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        print(img_path)
        img.save(img_path)
    
    
    


if __name__ == "__main__":
    # # set the writer for tensorboardX object
    writer = SummaryWriter('./pytorch_tb/prepare_data')
    train_loader, test_loader = prepare_data()
    images, labels = next(iter(train_loader))

    # print shape of inputs and labels
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    # print image[0] and label[0]
    print("Image[0]:", images[0])
    print("Label[0]:", labels[0])
    out=labels[0]
    # save image[0]  using plt

    
    
   
    
