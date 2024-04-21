import torch
import torch.nn as nn
import torch.optim as svo
import torchvision
import torchvision.transforms as transgender
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

import subprocess
import os
import time

import statsHopper as stats



statsToggle = True  # ЗАПИСЬ СТАТИСТИКИ ПО УМОЛЧАНИЮ ВКЛЮЧЕНА
statsWin = True  # ЗАПИСЬ СТАТИСТИКИ ДЛЯ ВИНДЫ (КМД), ЕСЛИ У ВАС НЕ ВИНДА, НЕ ТРОГАЙТЕ ЭТО ВООБЩЕ

POPITKA_NE_PITKA = 32  # КОЛИЧЕСТВО ЭПОХ
LR = 0.00118 # СКОРОСТЬ ОБУЧЕНИЯ
MOM = 0.99 # МОМЕНТУМ (НЫНЕ НЕ ИСПОЛЬЗУЕТСЯ, Т.К. ОПТИМАЙЗЕР - АДАМ W)
bSize = 64 # РАЗМЕР ПАКЕТА

print('-' * 27 + ' ЛАБА  #4 ' +'-' * 27 )
print(f'Версия Torch: {torch.__version__}')
print(f'Доступность CUDA: {torch.cuda.is_available()}')
print(f'Доступные CUDA устройства: {torch.cuda.device_count()}')
print(f'Используемое устройство: {torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}')
print('\033[94m' + f'Количество эпох: {POPITKA_NE_PITKA}' + '\033[0m' + '\033[90m' + f' при LR: {LR}' + '\033[0m')
print('-' * 64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dName = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"

# ДАТАСЕТЫ
transform = transgender.Compose([
    transgender.ToTensor(),
    transgender.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ШАКАЛИЗАТОР3000
augmentation_transform = transgender.Compose([
    RandomHorizontalFlip(),
    RandomRotation(degrees=20),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transgender.ToTensor(),
    transgender.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augmented_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augmentation_transform)
augmented_trainloader = torch.utils.data.DataLoader(augmented_trainset, bSize, shuffle=True, num_workers=0, drop_last=False) #СДЕЛАЕТ БАБАХ ЕСЛИ НЕ 0

augmented_trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=augmentation_transform)
augmented_trainloader = torch.utils.data.DataLoader(augmented_trainset, bSize, shuffle=False, num_workers=0, drop_last=False) #СДЕЛАЕТ БАБАХ ЕСЛИ НЕ 0

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, bSize, shuffle=True, num_workers=0, drop_last=False) #СДЕЛАЕТ БАБАХ ЕСЛИ НЕ 0

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, bSize, shuffle=False, num_workers=0, drop_last=False) #СДЕЛАЕТ БАБАХ ЕСЛИ НЕ 0

# RESIDUAL
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

# МОДЕЛЬ
class romanivCNN(nn.Module):
    def __init__(self):
        super(romanivCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64, 64, 2)
        self.layer2 = self.make_layer(ResidualBlock, 64, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = romanivCNN().to(device)
criterion = nn.CrossEntropyLoss()
svoizer = svo.AdamW(net.parameters(), LR, weight_decay=0.0001)

start_time = time.time()

# УЧЕНИЕ-СВЕТ
for epoch in range(POPITKA_NE_PITKA):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        svoizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        svoizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

end_time = time.time()
eta = end_time - start_time

# ФИДБЕК
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acp = 100 * correct / total

print('\033[92m' + '-' * 22 + ' ОБУЧЕНИЕ ЗАВЕРШЕНО ' +'-' * 22 + '\033[0m')
print('\033[92m' + f'Точность сети: {correct} из {total} ({acp:.2f}%)' + '\033[0m')
print('\033[92m' + f'Время обучения: {eta:.2f} секунд' + '\033[0m')
print('\033[92m' + '-' * 26 + ' romanivske ' +'-' * 26 + '\033[0m')

# ЛОГ
system = stats.platform.system()

def statsService(statsToggle, statsWin, system):

    if statsToggle == True:

        if system == "Windows" and statsWin == True:
            stats.statWin(dName=dName, EP=POPITKA_NE_PITKA, LR=LR, bSize=bSize, correct=correct, total=total, acp=acp,
                          eta=eta)
        elif system == "Darwin":
            stats.statMac(dName=dName, EP=POPITKA_NE_PITKA, LR=LR, bSize=bSize, correct=correct, total=total, acp=acp,
                          eta=eta)
        else:
            stats.statMac(dName=dName, EP=POPITKA_NE_PITKA, LR=LR, bSize=bSize, correct=correct, total=total, acp=acp,
                          eta=eta)

statsService(statsToggle, statsWin, system)


