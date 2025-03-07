import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
from models.DCGAN import Generator, Discriminator

# 設定參數
batch_size = 128
image_size = 64
num_epochs = 50
lr = 0.0001
beta1 = 0.5
ngpu = 1
dataroot = "./Q2_images/mnist"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)

# 數據增強
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 載入數據集
dataset = dsets.MNIST(root=dataroot, download=True, train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和優化器
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

G_losses = []
D_losses = []
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # 更新 Discriminator
        netD.zero_grad()
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        real_label = 0.9  # 平滑的真實標籤
        fake_label = 0.0  # 生成數據的標籤

        # 訓練真實數據
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        # 訓練假數據
        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake_data = netG(noise)
        label.fill_(fake_label)
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        # 更新 Discriminator
        optimizerD.step()

        # 計算 Discriminator 總損失
        errD = errD_real + errD_fake

        # 更新 Generator
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        # 保存損失
        G_losses.append(errG.item())
        D_losses.append(errD.item()) 

        # 進度
        if i % 100 == 0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD:.4f} Loss_G: {errG:.4f}")


    # 每 10 個 epoch 保存權重
    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), f"{save_dir}/netG_epoch_{epoch+1}.pth")

    # 保存生成影像
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            vutils.save_image(fake, f"{save_dir}/fake_samples_epoch_{epoch+1}.png", normalize=True)

# 繪製損失圖
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")