# import library

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# set the super parameters

batch_size_train = 64
batch_size_test = 1000



def prepare_data():
    """
    Prepare the MNIST dataset for training and testing.

    Returns:
    train_loader (DataLoader): DataLoader object for training data.
    test_loader (DataLoader): DataLoader object for testing data.
    """
    # 1.data preparation
    transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
    # 2.load train and test data
    trainset = datasets.MNIST('./data/MNIST_data/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('./data/MNIST_data/', download=True, train=False, transform=transform)

    # 3.create data loader
    train_loader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader
