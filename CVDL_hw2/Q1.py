import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from models.VGG19_BN import VGG19_BN
import torch
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
from PyQt5.QtWidgets import QMessageBox

# Q1-1
def load_image(image_folder):
    # 讀取圖片
    images = []
    for file_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, file_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            images.append(image)
        else:
            print(f"找不到圖片: {image_path}")

    # 數據增強
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30)
    ])
    augmented_images = [transform(image) for image in images]

    # 將增強後的圖片合併
    grid_size = 3
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, augmented_image in enumerate(augmented_images):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].imshow(augmented_image)
        axes[row, col].axis('on')

    plt.tight_layout()
    return fig

# Q1-2
def show_model_summary():
    model = VGG19_BN(num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 顯示模型結構
    summary(model, (3, 32, 32))

# Q1-3
def show_training_validation_metrics(image_path):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(16, 8)) # 調整視窗大小
    plt.imshow(img)
    plt.axis('off')  
    plt.title('Training and Validation Metrics')
    plt.show()

# CIFAR-10 class
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Q1-4
def inference_and_show(image_path, result_label):
    try : 
        # 數據增強
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-10 圖片大小為 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 載入圖片並應用數據增強
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        # 模型推論
        model = VGG19_BN(num_classes=10)
        model.load_state_dict(torch.load('best_vgg19_bn.pth'))
        model.eval()
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
            predicted_class = np.argmax(probabilities)

        # 在 GUI 上顯示預測的類別標籤
        result_label.setText(f"Predicted: {CIFAR10_CLASSES[predicted_class]}")

        # 顯示概率分佈直方圖
        plt.figure()
        plt.bar(CIFAR10_CLASSES, probabilities)
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.xticks(rotation=45)
        plt.show()
        
    except Exception as e:
        error_message = QMessageBox()
        error_message.setIcon(QMessageBox.Critical)
        error_message.setText("讀取不到圖片")
        error_message.setInformativeText(str(e))
        error_message.setWindowTitle("Error")
        error_message.exec_()
