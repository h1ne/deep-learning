import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#ネットワーク構造の定義
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12*12*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1,12*12*64)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)

#MNISTデータセットのロード
def load_MNIST(batch=128):
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=2)

    val_set = torchvision.datasets.MNIST(root="./data",
                                         train=False,
                                         download=True,
                                         transform=transform)
    val_loader =torch.utils.data.DataLoader(val_set,
                                            batch_size=batch,
                                            shuffle=True,
                                            num_workers=2)

    return {"train":train_loader, "validation":val_loader}

def main():
    #エポック数
    epoch = 20
    batch_size = 64

    #学習結果の保存
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_acc": []
    }

    #データのロード
    data_loder = load_MNIST(batch=batch_size)

    #GPUが使えるときは使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #ネットワーク構造の構築
    net = MyNet().to(device)
    print(net)

    #最適化方法の設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    for e in range(epoch):
        """ 学習部分 """
        loss = None
        train_loss = 0.0
        net.train() #学習モード
        print("\nTrain start")
        for i,(data,target) in enumerate(data_loder["train"]):
            data,target = data.to(device),target.to(device)

            #勾配の初期化
            optimizer.zero_grad()
            #順伝搬 -> 逆伝搬 -> 最適化
            output = net(data)
            loss = f.nll_loss(output,target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print("Training: {} epoch. {} iteration. Loss: {}".format(e+1,i+1,loss.item()))

        train_loss /= len(data_loder["train"])
        print("Training loss (ave.): {}".format(train_loss))
        history["train_loss"].append(train_loss)


        """検証部分"""
        print("\nValidation start")
        net.eval() #検証モード(Validation)
        val_loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for data,target in data_loder["validation"]:
                data,target = data.to(device),target.to(device)

                #順伝搬の計算
                output = net(data)
                loss = f.nll_loss(output,target).item()
                val_loss += f.nll_loss(output,target,reduction='sum').item()
                predict = output.argmax(dim=1,keepdim=True)
                accuracy += predict.eq(target.view_as(predict)).sum().item()

        val_loss /= len(data_loder["validation"].dataset)
        accuracy /= len(data_loder["validation"].dataset)

        print("Validation loss: {}, Accuracy: {}\n".format(val_loss,accuracy))

        history["validation_loss"].append(val_loss)
        history["validation_acc"].append(accuracy)

    PATH = "./my_mnist_model.pt"
    torch.save(net.state_dict(), PATH)

    #結果
    print(history)
    plt.figure()
    plt.plot(range(1, epoch+1), history["train_loss"], label="train_loss")
    plt.plot(range(1, epoch+1), history["validation_loss"], label="validation_loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(range(1, epoch+1), history["validation_acc"])
    plt.title("test accuracy")
    plt.xlabel("epoch")
    plt.savefig("test_acc.png")

if __name__ == "__main__":
    main()

