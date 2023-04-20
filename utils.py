import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    #pipeline to apply transformations sequentially
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        #torchvision.transforms.RandomResizedCrop(args.image_size,scale=(0.8,1.0)), #take a random crop
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #Normalize the mean and standard deviation values RGB three values
        #the images are scaled to [-1,1] ddpm paper
    ])
    
    dataset=torchvision.datasets.ImageFolder(args.dataset_path,transform=transforms) #load data and apply preprocess to the images
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    return dataloader
    
def setup_logging(run_name):
    #os.makedirs is for creating directories
    os.makedirs("/content/drive/MyDrive/stable_diffusion/models", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/stable_diffusion/results",exist_ok=True)
    os.makedirs(os.path.join("/content/drive/MyDrive/stable_diffusion/models",run_name),exist_ok=True)
    os.makedirs(os.path.join("/content/drive/MyDrive/stable_diffusion/results",run_name),exist_ok=True)