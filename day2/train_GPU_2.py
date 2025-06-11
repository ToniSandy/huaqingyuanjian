# # 完整的模型训练套路(以CIFAR10为例)
# import time
#
# import torch.optim
# import torchvision
# from torch import nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
#
# # from model import *
#
# # 定义训练的设备
# device = torch.device("cuda:0")
# # device = torch.device("cuda")# 单显卡写法没问题
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 常用写法
#
#
# # 准备数据集
# train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
#                                          train=True,
#                                          transform=torchvision.transforms.ToTensor(),
#                                          download=True)
#
# test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
#                                          train=False,
#                                          transform=torchvision.transforms.ToTensor(),
#                                          download=True )
#
# # 数据集长度
# train_data_size = len(train_data)
# test_data_size = len(test_data)
# print(f"训练数据集的长度{train_data_size}")
# print(f"测试数据集的长度{test_data_size}")
#
# # 加载数据集
# train_loader = DataLoader(train_data,batch_size=64)
# test_loader = DataLoader(test_data,batch_size=64)
#
# # 创建网络模型
# class Chen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 32, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self,x):
#         x = self.model(x)
#         return x
#
# chen = Chen()
# chen = chen.to(device)
#
# # 创建损失函数
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = loss_fn.to(device)
#
# # 优化器
# # learning_rate = 1e-2 相当于(10)^(-2)
# learning_rate = 0.01
# optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)
#
# # 设置训练网络的一些参数
# total_train_step = 0 #记录训练的次数
# total_test_step = 0 # 记录测试的次数
# epoch = 10 # 训练的轮数
#
# # 添加tensorboard
# writer = SummaryWriter("../logs_train")
#
# # 添加开始时间
# start_time = time.time()
#
# for i in range(epoch):
#     print(f"-----第{i+1}轮训练开始-----")
#     # 训练步骤
#     for data in train_loader:
#         imgs, targets = data
#         imgs = imgs.to(device)
#         targets = targets.to(device)
#         outputs = chen(imgs)
#         loss = loss_fn(outputs,targets)
#
#         # 优化器优化模型
#         optim.zero_grad()  # 梯度清零
#         loss.backward()  # 反向传播
#         optim.step()
#
#         total_train_step += 1
#         if total_train_step % 200 == 0:
#             print(f"第{total_train_step}的训练的loss:{loss.item()}")
#             writer.add_scalar("train_loss",loss.item(),total_train_step)
#
#     end_time = time.time()
#     print(f"训练时间{end_time - start_time}")
#     # 测试步骤（以测试数据上的正确率来评估模型）
#     total_test_loss = 0.0
#     # 整体正确个数
#     total_accuracy = 0
#     with torch.no_grad():
#         for data in test_loader:
#             imgs, targets = data
#             imgs = imgs.to(device)
#             targets = targets.to(device)
#             outputs = chen(imgs)
#             # 损失
#             loss = loss_fn(outputs,targets)
#             total_test_loss += loss.item()
#             # 正确率
#             accuracy = (outputs.argmax(1) == targets).sum()
#             total_accuracy += accuracy
#
#     print(f"整体测试集上的loss:{total_test_loss}")
#     print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
#     writer.add_scalar("test_loss", total_test_loss, total_test_step)
#     writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
#     total_test_step += 1
#
#     # 保存每一轮训练模型
#     torch.save(chen,f"model_save\\chen_{i}.pth")
#     print("模型已保存")
#
# writer.close()

# 完整的AlexNet训练脚本（GPU版本）
import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from alexnet import AlexNet  # 导入自定义的AlexNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理（适配AlexNet的224x224输入）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),  # 缩放到256x256
    torchvision.transforms.CenterCrop(224),  # 中心裁剪为224x224
    torchvision.transforms.ToTensor(),
])

# 加载CIFAR10数据集
train_data = torchvision.datasets.CIFAR10(
    root="../dataset_chen",
    train=True,
    transform=transform,
    download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="../dataset_chen",
    train=False,
    transform=transform,
    download=True
)

# 数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# 初始化模型、损失函数和优化器
model = AlexNet(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练参数
epochs = 10
total_train_step = 0
writer = SummaryWriter("../logs_train")

# 训练循环
start_time = time.time()
for epoch in range(epochs):
    print(f"----- Epoch {epoch + 1}/{epochs} -----")
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        # 前向传播
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 日志记录
        if batch_idx % 100 == 0:
            print(f"Train Step: {total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)
        total_train_step += 1

    # 测试阶段
    model.eval()
    total_test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()

    # 打印测试结果
    test_loss = total_test_loss / len(test_loader)
    accuracy = correct / len(test_data)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)

    # 保存模型
    torch.save(model.state_dict(), f"model_save/alexnet_{epoch}.pth")
    print(f"Model saved to model_save/alexnet_{epoch}.pth")

writer.close()
print(f"Total training time: {time.time() - start_time:.2f} seconds")