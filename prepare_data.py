# import library

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt



# set the super parameters
batch_size_train = 32
batch_size_test = 1000


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
    test_loader = DataLoader(testset, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

def save_img(imgs, labels, num=0):
    # save image[0]  using plt
    plt.imshow(imgs[num].squeeze(0), cmap="gray")
    plt.savefig(f"./imgs/{labels[num]}.png")
    return labels[num]


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
    plt.imshow(images[0].squeeze(0), cmap="gray")
    plt.savefig(f"./imgs/{out}.png")
    
    
    
   
    
