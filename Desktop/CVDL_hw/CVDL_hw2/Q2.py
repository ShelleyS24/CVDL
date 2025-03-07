import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import numpy as np
import torchvision.utils as vutils
from models.DCGAN import Generator, Discriminator

# Q2-1
def process_and_display_mnist(dataroot='./Q2_images/mnist/'):

    # 定義資料增強的轉換
    transform_augmented = transforms.Compose([
        transforms.RandomRotation(60),  
        transforms.ToTensor()           
    ])

    # 定義原始資料的轉換
    transform_original = transforms.Compose([
        transforms.ToTensor()
    ])

    # 載入分別應用轉換的資料集
    original_dataset = dsets.MNIST(root=dataroot, train=True, download=True, transform=transform_original)
    augmented_dataset = dsets.MNIST(root=dataroot, train=True, download=True, transform=transform_augmented)

    original_loader = DataLoader(original_dataset, batch_size=64, shuffle=True)
    augmented_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)

    # 獲取一批原始與增強後的影像
    original_images, _ = next(iter(original_loader))
    augmented_images, _ = next(iter(augmented_loader))

    # 為原始影像建立網格
    original_grid = vutils.make_grid(original_images[:64], padding=2, normalize=True)
    
    # 為增強後影像建立網格
    augmented_grid = vutils.make_grid(augmented_images[:64], padding=2, normalize=True)
    
    # 設置畫布
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 顯示原始影像
    axes[0].imshow(np.transpose(original_grid.cpu(), (1, 2, 0)))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 顯示增強後影像
    axes[1].imshow(np.transpose(augmented_grid.cpu(), (1, 2, 0)))
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')
    
    return fig

# Q2-2
def show_model_structure():

    # 初始化權重
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = 1  # 設定使用的 GPU 數量

    # model
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # 應用初始權重
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)

# Q2-3
def show_loss(image_path):
    img = mpimg.imread(image_path)
    plt.figure(figsize=(16, 8)) 
    plt.imshow(img)
    plt.axis('off')  
    plt.show()

# Q2-4
def inference(model_path=r'weights\netG_epoch_50.pth', dataroot='./Q2_images/mnist/'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ngpu = 1
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(model_path))
    netG.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    # 載入數據集
    dataset = dsets.MNIST(root=dataroot, train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 獲取一批真實影像
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    # 推斷生成假影像
    with torch.no_grad():
        noise = torch.randn(64, 100, 1, 1, device=device)
        fake_images = netG(noise)

    # 反標準化
    real_images = real_images.cpu() * 0.3081 + 0.1307
    fake_images = fake_images.cpu() * 0.3081 + 0.1307

    # 為真實影像建立網格
    real_grid = vutils.make_grid(real_images, padding=2, normalize=True)
    
    # 為生成的假影像建立網格
    fake_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    
    # 設置畫布
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 顯示真實影像
    axes[0].imshow(np.transpose(real_grid, (1, 2, 0)))
    axes[0].set_title('Real Image')
    axes[0].axis('off')
    
    # 顯示生成的假影像
    axes[1].imshow(np.transpose(fake_grid, (1, 2, 0)))
    axes[1].set_title('Fake Image')
    axes[1].axis('off')

    plt.show()