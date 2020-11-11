import torch
import pandas as pd
import os
import copy
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import time


class DataWithLabel(Dataset):

    def __init__(self, data_csv, root_dir, desire_idx, transform=None):
        """ Args:
        data_csv: Dataframe with image file names and labels (numeric).
        root_dir: Directory with all the images.
        desire_idx: data to choose
        transform: Optional transform to be applied on a sample.
        """
        self.images = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform

        for idx in desire_idx:
            self.images.append(data_csv.iloc[idx, 0])
            self.labels.append(data_csv.iloc[idx, 1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.images[idx]).zfill(6) + '.jpg')
        image = io.imread(img_name)
        img_class = []
        for i in range(196):
            if i == self.labels[idx]:
                img_class.append(1.0)
            else:
                img_class.append(0.0)
        img_class = torch.FloatTensor(img_class)

        if self.transform:
            # Deal with gray scale image
            if len(image.shape) == 2:
                image_expand = np.expand_dims(image, 2)
                image = np.concatenate(
                    (image_expand, image_expand, image_expand), axis=2)

            image = self.transform(image)

        sample = (image, img_class, self.labels[idx])

        return sample

class DataNoLabel:
    def __init__(self, root_dir, transform=None):
        """Args:
        root_dir: Directory with all images
        transform: Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        for root, dirs, files in os.walk(self.root_dir):
            self.images = files
            break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = io.imread(img_name)
        if self.transform:
            # Deal with gray scale image
            if len(image.shape) == 2:
                image_expand = np.expand_dims(image, 2)
                image = np.concatenate(
                    (image_expand, image_expand, image_expand), axis=2)

            image = self.transform(image)

        return image, self.images[idx]


def train_model(model, criterion, optimizer, scheduler, device,
                loaders, num_epochs, len_train, len_val):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    log_train = 0
    log_val = 0
    total_time = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            batch_count = 0
            running_loss = 0.0
            running_corrects = 0

            for inputs, label_list, label in loaders[phase]:
                inputs = inputs.to(device)
                label_list = label_list.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label.data)
                batch_count += 1
                #if batch_count % 50 == 0:
                #    print("{} batches done in {}".format(batch_count,
                #                                         time.time() - since))
            if phase == 'train':
                epoch_loss = running_loss / len_train
                epoch_acc = running_corrects.double() / len_train
                log_train = epoch_acc
            else:
                epoch_loss = running_loss / len_val
                epoch_acc = running_corrects.double() / len_val
                log_val = epoch_acc

            scheduler.step(log_val)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        total_time += time_elapsed
        print("Epoch{} in {:.0f}m {:.0f}s".format(
            epoch  + 1, time_elapsed // 60, time_elapsed % 60
        ))
        print()

        with open("log.txt", mode="a+") as file:
            file.write("{}: train: {:.4f}, val: {:.4f} for {}m {:.2f}s\n".format(
                str(epoch +1).zfill(3), log_train,
                log_val, time_elapsed // 60, time_elapsed % 60
            ))

        if (epoch + 1) % 5 == 0:
            PATH = 'checkpoint_' + str(epoch + 1) + '.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, PATH)

    print("Best val Acc: {:.4f}".format(best_acc))
    print("Total training time: {:.0f}h {:.0f}m {:.0f}s".format(
        total_time // 3600 ,total_time // 60, total_time % 60
    ))
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, loader, class_names, device, result_path):
    img_names = []
    preds = []
    model.eval()

    with torch.no_grad():
        for input, img_name in loader:
            # input = data['image']
            input = input.to(device)

            outputs = model(input)
            _, pred = torch.max(outputs, 1)

            # img_name = data['name']
            img_name = list(img_name)
            for i, ele in enumerate(img_name):
                img_names.append(ele.strip('.jpg'))

            for i, label in enumerate(pred):
                preds.append(class_names[label])

    df = pd.DataFrame(data={'id': img_names, 'label': preds})
    df.to_csv(result_path, index=False)
