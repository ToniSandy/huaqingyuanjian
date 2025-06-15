Dataset preprocessing, new engineering projects

deal_with_datasets.py
import os
import shutil
from sklearn.model_selection import train_test_split
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 数据集路径
dataset_dir = r'D:\Desktop\tcl\dataset\image2'  # 替换为你的数据集路径
train_dir = r'D:\Desktop\tcl\dataset\image2\train'  # 训练集输出路径
val_dir = r'D:\Desktop\tcl\dataset\image2\val'  # 验证集输出路径

# 划分比例
train_ratio = 0.7

# 创建训练集和验证集目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_name in os.listdir(dataset_dir):
    if class_name not in ["train","val"]:
        class_path = os.path.join(dataset_dir, class_name)


        # 获取该类别下的所有图片
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # 确保图片路径包含类别文件夹
        images = [os.path.join(class_name, img) for img in images]

        # 划分训练集和验证集
        train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

        # 创建类别子文件夹
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # 复制训练集图片
        for img in train_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(train_dir, img)
            shutil.move(src, dst)

        # 复制验证集图片
        for img in val_images:
            src = os.path.join(dataset_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.move(src, dst)

	      shutil.rmtree(class_path)

### prepare.py

import os

# 创建保存路径的函数
def create_txt_file(root_dir, txt_filename):
    # 打开并写入文件
    with open(txt_filename, 'w') as f:
        # 遍历每个类别文件夹
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # 遍历该类别文件夹中的所有图片
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")

create_txt_file(r'D:\Desktop\tcl\dataset\image2\train', 'train.txt')
create_txt_file(r'D:\Desktop\tcl\dataset\image2\val', "val.txt")


In the end, train.txt and val.txt were obtained

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            # img_path = os.path.join(self.data_dir, self.folder_name, img_path)
            self.labels.append(label)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, i):
        path, label = self.imgs_path[i], self.labels[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

Replace the function arguments in the Call the GPU to train

Take our model, data, tags, and push it to the GPU using ".cuda()".

Check if the machine can use the GPU
