import torch
from torch import nn
import torch.optim as optim
from prepare_data import prepare_data
from torch.utils.tensorboard import SummaryWriter

logger = SummaryWriter("./pytorch_tb/train_test")
# super parameters
n_epochs = 10
learning_rate = 0.01

# write the predicted and labels to csv file
def write_to_csv(filename,data: torch.Tensor):
    #  convert the tensor to numpy
    data = data.numpy()
    # write the data to csv file
    with open(filename, 'w') as f:
        # write column names
        f.write('pred,labels\n')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                f.write(str(data[i][j]) + ',')
            f.write('\n')
    print("Write to csv file successfully!")

def net_train(net: nn.Module, trainloader: torch.utils.data.DataLoader):
    """
    Train the neural network model.
    """
    # set the optimizer and loss function

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print("Start training...")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data

            # 1.input data
            outputs = net(inputs)  # forward
            # 2.calculate loss
            loss = criterion(outputs, labels)  # calculate the loss
            # 3.gradient to zero
            optimizer.zero_grad()  # zero the parameter gradients
            # 4.backpropagation
            loss.backward()  # backpropagation
            # 5.update parameters
            optimizer.step()  # update parameters
             
            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Calculate the number of correct predictions
            correct = (predicted == labels).sum().item()

            # Calculate the accuracy
            accuracy = correct / labels.size(0)

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                logger.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                logger.add_scalar('training accuracy', accuracy, epoch * len(trainloader) + i)
                running_loss = 0.0
    print("Finished Training")


def net_test(net: nn.Module, testloader: torch.utils.data.DataLoader):
    print("Start testing...")
    correct = 0
    total = 0
    with torch.no_grad():
        for i ,data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics in every 100 mini-batches
            print("[%5d] Accuracy of the network on the %d test images: %d %%" % ((i + 1)*1000, total, 100 * correct / total))
            # write the predicted and labels to csv file
            # join the predicted and labels
            predicted_for_write = predicted.reshape(-1,1)
            labels_for_write = labels.reshape(-1,1)
            data = torch.cat((predicted_for_write,labels_for_write),1)
            # write to csv file
            write_to_csv('./csv/predicted_labels.csv',data)
        print(
            "Accuracy of the network on the test images: %d %%"
            % (100 * correct / total)
        )
    print("Finished Testing")
