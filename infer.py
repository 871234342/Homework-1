import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from utils import DataNoLabel, test_model

# This program uses Resnet152 with weights loaded from weights.pth
# to predict label of a given image. testing_labels.csv will be
# created after inference completes.


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# Get labels
all_labels = pd.read_csv('training_labels.csv')
names = pd.read_csv('labels.csv')
names = np.array(names.iloc[:,0])
names = names.tolist()

# Transform
mean = [0.46951652, 0.45906892, 0.45407504]
std = [0.29279944, 0.2917506, 0.2999347]

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(512),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

infer_set = DataNoLabel(root_dir='testing_data/', transform=val_transform)
infer_loader = DataLoader(infer_set, num_workers=2, shuffle=False)


# Construct model
net = models.resnet152(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 196)

# Use GPU
device = torch.device('cuda' if use_cuda else 'cpu')
print("number of GPU(s): {}".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
print(device)
net = net.to(device)

# Inference
print("Inferring ...")
net.load_state_dict(torch.load('weights.pth'))
test_model(net, infer_loader, names, device, 'testing_labels.csv')

