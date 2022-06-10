from pyexpat import model
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from optuna.visualization import plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
import sys
import os
import time
import seaborn as sns
import optuna

seed_value = 47
torch.seed = seed_value


PATH = r'C:\Users\Asus\Documents\deep learning\deep_learn\starters\PetImages'

BATCH_SIZE = 64
tsfm = T.Compose([T.ToTensor(), T.Resize((200, 200)), T.Normalize(0, 1)])
folder = ImageFolder(PATH, transform=tsfm)
loader = DataLoader(folder, batch_size=BATCH_SIZE)

img, lab = next(iter(loader))
val_fol = ImageFolder(
    r'C:\Users\Asus\Documents\deep learning\deep_learn\starters\PetTest')
val_loader = DataLoader(val_fol, batch_size=BATCH_SIZE)

# plt.imshow(make_grid(img).permute(1, 2, 0).squeeze())
# plt.show()

# now i can create a NN
# but instead i will create objects of NN layer which can be accessed and
# i can use them to visualize the feature maps

# but i would also like to show some math behind what i am doing
# and how its working
# in background

# math for convolution
# a convolution starts from kernels or filter multiplications
# a kernel is made can be of any size (2x2,3x3,5x5,10x10)
# a kernel product sum every matrix and makes a feature map

# lets visualize the feature maps
# Feature detectors or filters help identify different features present
# in an image like edges, vertical lines, horizontal lines, bends, etc.


# out_channels = 5
# conv1 = nn.Conv2d(3, out_channels, 3)  # gives 6 kernels


# conv2 = nn.Conv2d(out_channels, out_channels*2, 3)

# conv3 = nn.Conv2d(out_channels*2, out_channels*3, 3, 2)
# seq = nn.Sequential(
#     conv1,
#     nn.MaxPool2d(out_channels),
#     nn.Dropout(0.2),
#     nn.BatchNorm2d(out_channels),
#     conv2,
# nn.MaxPool2d(out_channels*2),
#     nn.Dropout(0.2),
#     nn.BatchNorm2d(out_channels*2),
#     conv3)
# map3 = seq(img)


# map1 = conv1(img)
# print(map1[0].shape)
# nn.BatchNorm2d(198)
# map2 = conv2(map1)


# def show_tensor_images(image_tensor, num_images=out_channels*2, size=(1, 28, 28)):
#     '''
#     Function for visualizing images: Given a tensor of images, number of images, and
#     size per image, plots and prints the images in an uniform grid.
#     '''
#     image_tensor = (image_tensor + 1) / 2
#     image_unflat = image_tensor.detach().cpu()
#     image_grid = make_grid(image_unflat[:num_images], nrow=2)
#     plt.imshow(image_grid.permute(1, 2, 0).squeeze())
#     plt.show()

########################################################################

# for map, chan in zip([map1, map2, map3], [1, 2, 3]):
#     fig, ax = plt.subplots(BATCH_SIZE, out_channels*chan, figsize=(20, 20))
#     for i in range(0, BATCH_SIZE):
#         for j in range(0, out_channels*chan):
#             ax[i][j].set_xticklabels([])
#             ax[i][j].set_yticklabels([])
#             ax[i][j].set_yticks([])
#             ax[i][j].set_xticks([])
#             plt.tight_layout(pad=1.01)
#             ax[i][j].imshow(map[i][j].detach().cpu())
#     plt.show()
#     plt.savefig(f'map{chan}_output_with_{out_channels*chan}channels.png')
# # I wanna see all the maps after i have applied maxpool and batchnorm
# # earlier i used only batch norm


# now i would like to see the distribution of
# different image channels before and after the batch_norm

# _, ax = plt.subplots(1, 3)
# ax[0].hist(map1[0][1].detach().cpu())
# ax[1].hist(map2[0][1].detach().cpu())
# ax[2].hist(map3[0][1].detach().cpu())

# sns.histplot(map1[0][1].detach().cpu().numpy(),
#              ax=ax[0], kde=True, legend=False)
# sns.histplot(map3[0][1].detach().cpu().numpy(),
#              ax=ax[2], kde=True, legend=False)
# sns.histplot(map2[0][1].detach().cpu().numpy(),
#              ax=ax[1], kde=True, legend=False)
# ax[0].legend([])
# ax[1].legend([])
# ax[2].legend([])
# plt.show()

# 09-06-22
# today we will be visualising loss and optimisers outputs
# for that we need to make an NN first !
# lets start !!

class Net(nn.Module):
    def __init__(self, img=None, label=None):
        super(Net, self).__init__()
        self.data = img
        mapper = {'Cat': 0, 'Dog': 1}
        self.lab = label

        self.convs = nn.Sequential(
            self.conv_block(3, 5, 5, 1),
            self.conv_block(5, 10, 5, 1),
            self.conv_block(10, 12, 5, 1),

        )
        self.fc = self.conv_block(432, 2, final=True)

    def conv_block(self, in_channels, out_channels, k_size=5, padding=1, final: bool = False):
        if final:
            return nn.Sequential(nn.Linear(in_channels, out_channels),
                                 nn.Softmax(),
                                 #  nn.Linear(out_channels, 2),
                                 #  nn.Softmax()
                                 )
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding),
                             nn.MaxPool2d(3),
                             nn.Dropout(0.2),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU()
                             )

    def forward(self, x):
        print(x.shape)
        x = self.convs(x)
        # write FORMULA FOR CONV-------------
        # Cout[198] = ((Cin[200]+2P-Kernel_size)/(stride))+1
        # after 3 conv layers
        # 194*194 having 12 channels
        print(x.shape)
        x = x.view(64, 6*6*12)
        # flatten
        x = self.fc(x)
        return x


# this is 388 in length so i have to limit the number of batches that are getting loaded!!!!
# print(len(loader))
# pred = Net()(img)
# model = Net()
# print(pred)
# print(torch.round(pred))
# print(torch.argmax(pred, dim=-1))
# print(torch.round(torch.max(pred)))
criterion = nn.BCELoss()
# if batches go above this >>>> the loop breaksss
N_TRAIN_SAMPLES = BATCH_SIZE * 40
N_VALID_SAMPLES = BATCH_SIZE * 5


def train_model(model, optimizer, train_loader):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * BATCH_SIZE >= N_TRAIN_SAMPLES:
            break
        data, target = data.view(
            data.size(0), -1).to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        losses.append(loss)
        optimizer.step()

        # == the actual targetsss
        accuracy = torch.sum(target) - \
            torch.sum(torch.argmax(output))/BATCH_SIZE
        print(accuracy)
        return losses


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            if batch_idx * BATCH_SIZE >= N_VALID_SAMPLES:
                break
            data, target = data.view(
                data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / min(len(valid_loader.dataset), N_VALID_SAMPLES)
    return accuracy


DIR = ".."
DEVICE = torch.device("cpu")


def objective(trial):

    model = Net().to(DEVICE)
    # Sample the initial learning rate from [1e-5, 1e-1] in log space.
    optimizer = torch.optim.Adam(
        nn.ParameterList(model.parameters()), trial.suggest_float(
            "lr_init", 1e-5, 1e-1, log=True)
    )

    for step in range(10):
        model.train()
        losses = train_model(model, optimizer, loader)

        accuracy = eval_model(model, val_loader)

        # Report intermediate objective value.
        trial.report(accuracy, step)
        trial.report(losses, step)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy


start = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=47),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=20, timeout=600)
end = time.time()
time_taken = end-start
print(f'time_taken: {time_taken}')

plot_optimization_history(study)
plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"])
plot_contour(study)
# model = Net()
# pred = model(img)
# print(pred)
