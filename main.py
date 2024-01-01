from net import Full_Net
from train_test import net_train, net_test
from prepare_data import prepare_data


if __name__ == '__main__':
    # 1.get data
    trainloader, testloader = prepare_data()
    # 2.create network
    net = Full_Net()
    # 3.train network
    net_train(net, trainloader)
    # 4.test network
    net_test(net, testloader)
    


