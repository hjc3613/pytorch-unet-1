import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import myhelper
import simulation
import cv2

input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
for x in [input_images, target_masks]:
    print(x.shape)
    print(x.min(), x.max())
    
input_images_rgb = [x.astype(np.uint8) for x in input_images]
target_masks_rgb = [myhelper.masks_to_colorimg(x) for x in target_masks]

print(input_images_rgb[0].shape,  input_images_rgb[0].max(), input_images_rgb[0].min())
print(target_masks_rgb[0].shape,  target_masks_rgb[0].max(), target_masks_rgb[0].min())
# i=0
# for img, mask in zip(input_images_rgb, target_masks_rgb):
#     cv2.imshow(f'img{i}', img)
#     cv2.imshow(f'mask{i}', mask)
#     i += 1
    
# cv2.waitKey()

##########################################################################################
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch

class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
            
        return [image, mask]
    
trans = transforms.Compose([
    transforms.ToTensor(),
])

train_set = SimDataset(2000, transform=trans)
val_set = SimDataset(200, transform=trans)
# indexes = np.random.randint(0, 20, size=(7,))
# print(indexes)
# for idx in indexes:
#     img, mask = val_set[idx]
#     img = img.numpy().astype(np.uint8) * 155
#     img = np.transpose(img, (1, 2, 0))
#     cv2.imshow(f'img{idx}', img)
#     cv2.imshow(f'mask{idx}', myhelper.masks_to_colorimg(mask))
    
# cv2.waitKey()

image_datasets = {
    'train':train_set, 'val':val_set
}

batch_size = 50

dataloaders = {
    'train':DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val':DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
}

dataset_sizes = {
    x:len(image_datasets[x]) for x in image_datasets.keys()
}
print(dataset_sizes)

import torchvision.utils
from torchsummary import summary
import torch
import torch.nn as nn
import pytorch_unet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ', device)
model = pytorch_unet.UNet(6)
model = model.to(device)
summary(model, input_size=(3, 224, 224), device=device)


################################################################################

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss
import time
import torch

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce*bce_weight + dice*(1-bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k]/epoch_samples))
    print("{}:{}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    if os.path.exists('state_dict.pt'):
        model.load_state_dict(torch.load('state_dict.pt'))
        print('load existing model state')
    else:
        print('no state_dict.pt exists, please download from baidu yun if you wish to train more quickly!!, now the model trained from random initializer state.')
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print('LR', param_group['lr'])
                model.train()
            else:
                model.eval()
            
            metrics = defaultdict(float)
            epoch_samples = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs=model(inputs)
                    # print(outputs.shape)
                    loss = calc_loss(outputs, labels, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss']/epoch_samples
            
            if phase == 'val' and epoch_loss < best_loss:
                print('saving best model')
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'state_dict.pt')
        time_elapsed = time.time()-since
        print('{: .0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model

#############################################################################################
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

num_classes = 6
model = pytorch_unet.UNet(num_classes)
optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=100)



# predict
import math

model.eval()   # Set model to evaluate mode

test_dataset = SimDataset(5, transform = trans)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=0)
        
inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

pred = model(inputs)

pred = pred.data.cpu().numpy()
print(pred.shape)
print([i.max() for i in pred])

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    
    return inp

# Change channel-order and make 3 channels for matplot
input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

# Map each channel (i.e. class) to each color
target_masks_rgb = [myhelper.masks_to_colorimg(x) for x in labels.cpu().numpy()]
pred_rgb = [myhelper.masks_to_colorimg(x) for x in pred]
i = 0
for img, target, pred in zip(input_images_rgb, target_masks_rgb, pred_rgb):
    print(pred.max(), pred.min())
    concated = np.concatenate((img, target, pred), axis=1)
    #cv2.imshow('a', concated)
    cv2.imwrite(f'test_{i}.jpg', concated)
    #cv2.imshow('img'+str(i), img)
    #cv2.imshow('target'+str(i), target)
    #cv2.imshow('pred'+str(i), pred)
    i += 1
    
#cv2.waitKey()
    

