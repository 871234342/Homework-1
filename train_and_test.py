import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from utils import DataWithLabel, DataNoLabel, train_model, test_model

# NOTE:
# change label processing using labels.csv
# weight decay = 0.000001 might be too small, but seems to do something about overfitting?
# weight decay = 0.00001, fail


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# Hyper-parameters
weight_decay = 0.0005
batch_size = 12
total_epoch = 25
start_epoch = 0
skip_train = False
if start_epoch  == 0:
    weight_load_path = None
else:
    weight_load_path = "checkpoint_" + str(start_epoch) + ".pth"


# Log file
with open("log.txt", "w") as file:
    file.write("Resnet101 batch_size:{} SGD weight decay:{}\n".format(
        batch_size, weight_decay
    ))


# Data preprocess
all_labels = pd.read_csv('training_labels.csv')
names = pd.read_csv('labels.csv')
names = np.array(names.iloc[:,0])
names = names.tolist()
categories = [[] for i in range(196)]

for idx, name in enumerate(all_labels.iloc[:,1]):
    all_labels.iloc[idx,1] = names.index(name)
    categories[names.index(name)].append(idx)

val_num = 5     #number of validation images per class
train_idx = []
val_idx = []
for category in categories:
    random.shuffle(category)
    for _, idx in enumerate(category):
        if _ < val_num:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

# Transform
mean = [0.46951652, 0.45906892, 0.45407504]
std = [0.29279944, 0.2917506, 0.2999347]

def add_noise(img):
    L = random.random() / 2
    img = img*(1 - L) + L * torch.randn(img.shape)
    return img

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomHorizontalFlip(),
     transforms.Resize(512),
     transforms.RandomCrop((448, 448)),
     transforms.ToTensor(),
     transforms.Normalize(mean, std),
     transforms.Lambda(lambda x: add_noise(x))
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(512),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = DataWithLabel(data_csv=all_labels, root_dir='training_data/',
                    desire_idx=train_idx, transform=train_transform)
val_set = DataWithLabel(data_csv=all_labels, root_dir='training_data/',
                    desire_idx=val_idx, transform=val_transform)
infer_set = DataNoLabel(root_dir='testing_data/', transform=val_transform)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          num_workers=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        num_workers=2, shuffle=True)
loaders={'train': train_loader, 'val': val_loader}
infer_loader = DataLoader(infer_set, num_workers=2, shuffle=False)


# Construct model
net = models.resnet152(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 196)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("number of GPU(s): {}".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
print(device)
net = net.to(device)

# Criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
optimizer2 = optim.SGD(net.parameters(), lr = 0.01, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)
scheduler2 = lr_scheduler.ReduceLROnPlateau(optimizer,
                                            "max", patience=3, verbose=True)

# Load checkpoint, if any
if weight_load_path:
    checkpoint = torch.load(weight_load_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler2.load_state_dict(checkpoint['scheduler_state_dict'])


#Train model
if not skip_train:
    print("Start training...")
    net= train_model(
        net, criterion, optimizer2, scheduler2, device, loaders,
        total_epoch - start_epoch, len(train_set), len(val_set))
    PATH = 'weights_val.pth'
    torch.save(net.state_dict(), PATH)


# Inference
print("Inferring ...")
net.load_state_dict(torch.load('weights_val.pth'))
test_model(net, infer_loader, names, device, 'testing_labels.csv')

