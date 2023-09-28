import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import numpy as np
import torch.nn.functional as F

#Data Preprocessing#
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Load Dataset#
train_set = torchvision.datasets.CIFAR10(root='./', train=True, download = True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./', train=False, download = True, transform=transform_test)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

print("Training Cifar10 Figure 5 - Group 1 - 0.9 - Black Line")

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learningratevalue = 0.1
weightdecayvalue = 5e-4
momentumvalue = 0.9

num_epochs = 150
iteration = 0
writer = SummaryWriter("./runs/Cifar10-0.1-m0.9-30*0.99985")

model = WideResNet(28, 10, 10, dropRate=0.0)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = learningratevalue, momentum = momentumvalue, weight_decay = weightdecayvalue)#, nesterov = True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    lossaverage = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        print("\r                                                                                             ",end="")
        if epoch > 30:
            learningratevalue = 0.99985 * learningratevalue #0.99993441
            optimizer.param_groups[0]['lr'] = learningratevalue        
        print("\rTraining Epoch {} ... {:.2f}% ... Current Learning Rate: {}".format(epoch + 1, 100*i/len(train_loader), learningratevalue),end="")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        lossaverage = running_loss / (i + 1)
        iteration = iteration + 1
        
        ###LCAM###
        #if loss.item() < lossaverage:
            #optimizer.param_groups[0]['momentum'] = 0.9 #Right Sparse
        #else:
            #optimizer.param_groups[0]['momentum'] = 0.95 #Left Non-Sparse
        ###LCAM###
        
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        i = 0
        for data, target in test_loader:
            print("\r                                                                                             ",end="")
            print("\rTesting Epoch {} .... {:.2f}%".format(epoch + 1, 100*i/len(test_loader)),end="")
            i = i + 1
            images = data.to(device)
            labels = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        
    writer.add_scalar('Learning Rate / Epoch', learningratevalue, epoch + 1)
    writer.add_scalar('Weight Decay / Epoch', weightdecayvalue, epoch + 1)
    writer.add_scalar('TestError / Epoch', 100 - test_acc, epoch + 1)
    writer.add_scalar('TrainLoss / Epoch', lossaverage, epoch + 1)
    
    print('\rEpoch [{}/{}] Iteration [{}] Test Error: {:.2f}% Loss: {:.6f} Learning Rate: {}'.format(epoch + 1, num_epochs, iteration, 100 - test_acc, lossaverage, learningratevalue))    

print("Training Cifar10 Figure 5 - Group 2 - 0.9_0.95 - Green Line")

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learningratevalue = 0.1
weightdecayvalue = 5e-4
momentumvalue = 0.9

num_epochs = 150
iteration = 0
writer = SummaryWriter("./runs/Cifar10-0.1-m0.9_0.95-30*0.99985")

model = WideResNet(28, 10, 10, dropRate=0.0)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = learningratevalue, momentum = momentumvalue, weight_decay = weightdecayvalue)#, nesterov = True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    lossaverage = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        print("\r                                                                                             ",end="")
        if epoch > 30:
            learningratevalue = 0.99985 * learningratevalue #0.99993441
            optimizer.param_groups[0]['lr'] = learningratevalue        
        print("\rTraining Epoch {} ... {:.2f}% ... Current Learning Rate: {}".format(epoch + 1, 100*i/len(train_loader), learningratevalue),end="")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        lossaverage = running_loss / (i + 1)
        iteration = iteration + 1
        
        ###LCAM###
        if loss.item() < lossaverage:
            optimizer.param_groups[0]['momentum'] = 0.9 #Right Sparse
        else:
            optimizer.param_groups[0]['momentum'] = 0.95 #Left Non-Sparse
        ###LCAM###
        
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        i = 0
        for data, target in test_loader:
            print("\r                                                                                             ",end="")
            print("\rTesting Epoch {} .... {:.2f}%".format(epoch + 1, 100*i/len(test_loader)),end="")
            i = i + 1
            images = data.to(device)
            labels = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        
    writer.add_scalar('Learning Rate / Epoch', learningratevalue, epoch + 1)
    writer.add_scalar('Weight Decay / Epoch', weightdecayvalue, epoch + 1)
    writer.add_scalar('TestError / Epoch', 100 - test_acc, epoch + 1)
    writer.add_scalar('TrainLoss / Epoch', lossaverage, epoch + 1)
    
    print('\rEpoch [{}/{}] Iteration [{}] Test Error: {:.2f}% Loss: {:.6f} Learning Rate: {}'.format(epoch + 1, num_epochs, iteration, 100 - test_acc, lossaverage, learningratevalue))    

#Load Dataset#
train_set = torchvision.datasets.CIFAR100(root = './', train = True, download = True, transform = transform_train)
test_set = torchvision.datasets.CIFAR100(root = './', train = False, download = True, transform = transform_test)

print("Training Cifar100 Figure 5 - Group 1 - 0.9 - Black Line")

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learningratevalue = 0.1
weightdecayvalue = 5e-4
momentumvalue = 0.9

num_epochs = 150
iteration = 0
writer = SummaryWriter("./runs/Cifar100-0.1-m0.9-30*0.99985")

model = WideResNet(28, 100, 10, dropRate=0.0)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = learningratevalue, momentum = momentumvalue, weight_decay = weightdecayvalue)#, nesterov = True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    lossaverage = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        print("\r                                                                                             ",end="")
        if epoch > 30:
            learningratevalue = 0.99985 * learningratevalue #0.99993441
            optimizer.param_groups[0]['lr'] = learningratevalue        
        print("\rTraining Epoch {} ... {:.2f}% ... Current Learning Rate: {}".format(epoch + 1, 100*i/len(train_loader), learningratevalue),end="")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        lossaverage = running_loss / (i + 1)
        iteration = iteration + 1

        ###LCAM###
        #if loss.item() < lossaverage:
            #optimizer.param_groups[0]['momentum'] = 0.9 #Right Sparse
        #else:
            #optimizer.param_groups[0]['momentum'] = 0.95 #Left Non-Sparse
        ###LCAM###
        
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        i = 0
        for data, target in test_loader:
            print("\r                                                                                             ",end="")
            print("\rTesting Epoch {} .... {:.2f}%".format(epoch + 1, 100*i/len(test_loader)),end="")
            i = i + 1
            images = data.to(device)
            labels = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        
    writer.add_scalar('Learning Rate / Epoch', learningratevalue, epoch + 1)
    writer.add_scalar('Weight Decay / Epoch', weightdecayvalue, epoch + 1)
    writer.add_scalar('TestError / Epoch', 100 - test_acc, epoch + 1)
    writer.add_scalar('TrainLoss / Epoch', lossaverage, epoch + 1)
    
    print('\rEpoch [{}/{}] Iteration [{}] Test Error: {:.2f}% Loss: {:.6f} Learning Rate: {}'.format(epoch + 1, num_epochs, iteration, 100 - test_acc, lossaverage, learningratevalue))    

print("Training Cifar100 Figure 5 - Group 2 - 0.93_0.9 - Green Line")

batch_size = 128

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learningratevalue = 0.1
weightdecayvalue = 5e-4
momentumvalue = 0.9

num_epochs = 150
iteration = 0
writer = SummaryWriter("./runs/Cifar100-0.1-m0.93_0.9-30*0.99985")

model = WideResNet(28, 100, 10, dropRate=0.0)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = learningratevalue, momentum = momentumvalue, weight_decay = weightdecayvalue)#, nesterov = True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    running_loss = 0.0
    lossaverage = 0
    model.train()
    for i, data in enumerate(train_loader, 0):
        print("\r                                                                                             ",end="")
        if epoch > 30:
            learningratevalue = 0.99985 * learningratevalue #0.99993441
            optimizer.param_groups[0]['lr'] = learningratevalue        
        print("\rTraining Epoch {} ... {:.2f}% ... Current Learning Rate: {}".format(epoch + 1, 100*i/len(train_loader), learningratevalue),end="")
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        lossaverage = running_loss / (i + 1)
        iteration = iteration + 1

        ###LCAM###
        if loss.item() < lossaverage:
            optimizer.param_groups[0]['momentum'] = 0.93 #Right Sparse
        else:
            optimizer.param_groups[0]['momentum'] = 0.9 #Left Non-Sparse
        ###LCAM###
        
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        i = 0
        for data, target in test_loader:
            print("\r                                                                                             ",end="")
            print("\rTesting Epoch {} .... {:.2f}%".format(epoch + 1, 100*i/len(test_loader)),end="")
            i = i + 1
            images = data.to(device)
            labels = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        
    writer.add_scalar('Learning Rate / Epoch', learningratevalue, epoch + 1)
    writer.add_scalar('Weight Decay / Epoch', weightdecayvalue, epoch + 1)
    writer.add_scalar('TestError / Epoch', 100 - test_acc, epoch + 1)
    writer.add_scalar('TrainLoss / Epoch', lossaverage, epoch + 1)
    
    print('\rEpoch [{}/{}] Iteration [{}] Test Error: {:.2f}% Loss: {:.6f} Learning Rate: {}'.format(epoch + 1, num_epochs, iteration, 100 - test_acc, lossaverage, learningratevalue))    