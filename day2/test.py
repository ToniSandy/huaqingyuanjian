# import torch
# import torchvision
# from PIL import Image
# from model import *
# from torchvision import transforms
#
# image_path = "../Image/img.png"
# image = Image.open(image_path)
# print(image)
# # png格式是四个通道，除了RGB三个通道外，还有一个透明通道,调用image = image.convert('RGB')
# image = image.convert('RGB')
#
# # 改变成tensor格式
# trans_reszie = transforms.Resize((32, 32))
# trans_totensor = transforms.ToTensor()
# transform = transforms.Compose([trans_reszie, trans_totensor])
# image = transform(image)
# print(image.shape)
#
# # 加载训练模型
# model = torch.load("model_save\\chen_9.pth").to("cuda")
#
# # print(model)
#
# image = torch.reshape(image, (1, 3, 32, 32)).to("cuda")
# # image = image.cuda()
#
# # 将模型转换为测试模型
# model.eval()
# with torch.no_grad():
#     output = model(image)
# # print(output)
#
# print(output.argmax(1))
# 测试脚本（加载AlexNet模型）
import torch
import torchvision
from PIL import Image
from alexnet import AlexNet  # 导入自定义的AlexNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理（与训练一致）
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
])

# 加载测试图像
image_path = "../Image/img.png"
image = Image.open(image_path).convert("RGB")  # 确保RGB格式
image = transform(image).unsqueeze(0).to(device)  # 增加batch维度并送GPU

# 加载模型
model = AlexNet(num_classes=10).to(device)
model.load_state_dict(torch.load("model_save/alexnet_9.pth", map_location=device))
model.eval()

# 预测
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(1).item()

# 输出结果
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Predicted class: {class_names[predicted_class]}")