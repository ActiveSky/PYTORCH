import torch
from net import Full_Net, Conv_Net
from train_test import net_train, net_test
from prepare_data import prepare_data
from predict import predict_digit

# multi-cpu training

torch.set_num_threads(4) # 设置为你想要的线程数

def print_data(data_loader):
    # Assuming `data_loader` is an instance of torch.utils.data.DataLoader
    flag = 0
    for i, data in enumerate(data_loader):
        if flag == 1:
            break
        inputs, labels = data
        print(f"Batch {i+1}:")
        print("Inputs:", inputs)
        print("Labels:", labels)
        # print shape of inputs and labels
        print("Inputs shape:", inputs.shape)
        print("Labels shape:", labels.shape)
        flag += 1


if __name__ == "__main__":
    # set random seed
    random_seed = 1
    torch.manual_seed(random_seed)

    # 1.get data
    trainloader, testloader = prepare_data()

    # print data
    # print_data(trainloader)

    # 2.create network
    net = Conv_Net()

    # 3.1 train network
    net_train(net, trainloader)
    # # 3.2 load weights
    # net.load_state_dict(torch.load("CovNet__weights.pth"))

    # 4.test network
    net_test(net, testloader)
    # 5.save weights
    torch.save(net.state_dict(), "CovNet__weights.pth")
    
    # # 6.predict
    # image_path = "./imgs/1.png"
    # res = predict_digit(image_path, net)
    # print(res)
