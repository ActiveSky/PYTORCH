import torch
from torch import nn
import torch.optim as optim
from prepare_data import prepare_data

# super parameters
n_epochs = 3
learning_rate = 0.01
momentum = 0.5
log_interval = 10


def net_train(net: nn.Module, trainloader: torch.utils.data.DataLoader):
    """
    Train the neural network model.
    """
    # set the optimizer and loss function

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print("Start training...")
    for epoch in range(n_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:   # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def net_test(net: nn.Module, testloader: torch.utils.data.DataLoader):
    print("Start testing...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print("Finished Testing")
 