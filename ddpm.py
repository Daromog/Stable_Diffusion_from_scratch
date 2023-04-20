import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.insert(0, '/content/drive/MyDrive/stable_diffusion/')

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import * #Calling another script
from modules import UNet # Calling another function
import logging
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

#-----------------------------------------------------------------------------------------------------------------

class Diffusion:
    #Here the start and the end of noise are defined and the image size
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device) # The schedule for the noise
        self.alpha = 1. - self.beta 
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #For multipling the alphas       
        
    def prepare_noise_schedule(self): #For scheduling the noise generation for the forward process
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self,n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)) #Make a randon vector of the size of the batch
    
    def noise_images(self, x, t):
        #x = images t=random vector
        #This function apply the main equation to add noise to the images
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] #make a tensor with new dimensions and the multiplication of the alphas
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        Ɛ = torch.randn_like(x) #random vector taken from a normal distribution and std=1
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ   
    
    def sample(self, model, n):
        #n=size of the batch
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,3,self.img_size,self.img_size)).to(self.device) #[batch_size,3,img_size,img_size]
            for i in tqdm(reversed(range(1,self.noise_steps)),position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alpha[t][:,None,None,None]
                alpha_hat = self.alpha_hat[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x 
    

def see_images(dataloader):
    num_images = 4 #num of images you want to see  
    data_iter = iter(dataloader) #get a batch
    images, _ = data_iter.next()

    #create a grid of images
    grid = np.transpose(images[:num_images],(0,2,3,1))
    
    #plot the images
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    for i,ax in enumerate(axes.flat):
        ax.imshow(grid[i])
        ax.axis("off")
    #show the plot
    plt.tight_layout()
    plt.show()    


def train(args):
    setup_logging(args.run_name) #Create directories to save results and checkpoints
    device = args.device
    dataloader = get_data(args) #get the data preprocessed
    #see_images(dataloader)  #uncomment if you want to see the images
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("/content/drive/MyDrive/stable_diffusion/runs",args.run_name))
    l = len(dataloader)
    
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i,(images,_) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device) #A vector 1 to number of noise steps of size batch-size
            x_t, noise = diffusion.noise_images(images,t) #This apply the main equation to apply noise to the images         
            predicted_noise = model(x_t, t) # images [12,3,224,224] 
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE",loss.item(),global_step=epoch*l + i)
        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("/content/drive/MyDrive/stable_diffusion/results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("/content/drive/MyDrive/stable_diffusion/models", args.run_name, f"ckpt.pt"))
               
            
    

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500 #Epochs
    args.batch_size = 16 #batch_size
    args.image_size = 64 #image size
    args.dataset_path = r"/content/drive/MyDrive/stable_diffusion/landscape_images"
    args.device = "cuda" 
    args.lr = 3e-4 #Learning Rate
    train(args)


if __name__=="__main__":
    launch()
