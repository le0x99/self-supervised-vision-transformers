import json, pickle, warnings, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp
import torch
import torch.optim as optim
import warnings
import torch.nn as nn
from torchvision import transforms
import torchvision
import subprocess
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.loss import SmoothL1Loss, MSELoss
from PatchLocalizer import ContextLearner, MaskedContextLearner, MaskedPatchRegressor

torch.manual_seed(42)


pt_train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=96, scale=(0.6, 1.0)),
                                       transforms.RandomGrayscale(p=0.2),
                                       torchvision.transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))], 0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])
ft_train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=96, scale=(0.6, 1.0)),
                                       transforms.RandomGrayscale(p=0.2),
                                       torchvision.transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))], 0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])
ft_test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])




class Trainer(object):
    def __init__(self, s):
        self.pt_settings = s
        self.name = f'[{s["METHOD"]},{s["G"]},{s["CROP"]}]-[{s["LEARNING_RATE"]},{s["BATCH_SIZE"]}]-[{s["N_BLOCKS"]},{s["HEADS"]},{s["HIDDEN_MULT"]},{1 if s["PROJECT_PATCH_EMB"] else 0},{1 if s["NN_HEAD"] else 0}, {s["DROPOUT"]}]'
        self.name += f'-[{s["COMMENT"]}]' if len(s["COMMENT"]) else ''
        self.dir = f'./{self.name}'
        try:
            os.mkdir(self.dir)
        except:
            pass
    def save_state(self, model, optimizer, step, epoch):
        self.logger.flush()
        state = {
        'epoch': epoch,
            "step" : step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
            "pt_settings" : self.pt_settings,
                    }
        torch.save(state, self.dir + "/last_state")
         
    def imshow(self, trainloader, n=6):
        imgs = torch.stack([trainloader.dataset[i][0] for i in range(n)])
        img_grid = torchvision.utils.make_grid(imgs, nrow=n, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid)
        plt.axis('off')
        plt.show()
        plt.close()
        
    def load_pt_data(self):
        s = self.pt_settings
        dataset = torchvision.datasets.STL10("../STL10/unlabeled/",
                                                 split="unlabeled",
                                                  transform=pt_train_transforms)

        if s["SAMPLE"]:
            dataset, _ = torch.utils.data.random_split(dataset, [8192, 91808])
            
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=s["BATCH_SIZE"],
                                                  shuffle=True, num_workers=4, drop_last=True,
                                                  pin_memory=True)
        return trainloader
    
    
    def train(self, continue_training=False, verbose=False):
        print("[...] - Initialize Pre-Training") if not continue_training else print("[...] - Continue Pre-Training / Load from Checkpoint")
        s = self.pt_settings
        if verbose:
            print("[...] - Training Config : ")
            print("\n")
            pp(s)
            print("\n")
        trainloader = self.load_pt_data()
        self.logger = SummaryWriter(log_dir = "./runs/"+self.name)
        subprocess.Popen('tensorboard --logdir=runs --port=6007', shell=True)
        step = 0
        if s["METHOD"] == "PLT":
            model = ContextLearner(self.pt_settings)
            optimizer = optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], weight_decay=s["WEIGHT_DECAY"])
            if continue_training:
                state = torch.load(self.dir + "/last_state")
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                step = state["step"]
            print("[...] - Starting Training ")
            self.train_plt(model, optimizer, trainloader, step)
            
        elif s["METHOD"] == "PLT_MASKED":
            model = MaskedContextLearner(self.pt_settings)
            optimizer = optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], weight_decay=s["WEIGHT_DECAY"])
            if continue_training:
                state = torch.load(self.dir + "/last_state")
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                step = state["step"]
            print("[...] - Starting Training ")  
            self.train_plt_m(model, optimizer, trainloader, step)
            
        elif s["METHOD"] == "MPR":
            model = MaskedPatchRegressor(self.pt_settings)
            optimizer = optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], weight_decay=s["WEIGHT_DECAY"])
            if continue_training:
                state = torch.load(self.dir + "/last_state")
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                step = state["step"]
            print("[...] - Starting Training ")
            self.train_mpr(model, optimizer, trainloader, step)
            
        else:
            raise NotImplemented 
             
    
    def train_plt(self, model, optimizer, trainloader, step):
        s = self.pt_settings
        criterion = CrossEntropyLoss()
        for epoch in range(1, s["NUM_EPOCHS"]+1):
            for data in trainloader:
                X = data[0].to(s["DEVICE"])
                target = torch.arange(model.T, device=s["DEVICE"]).repeat(s["BATCH_SIZE"])
                optimizer.zero_grad()

                Y = model(X)
                loss = criterion(Y, target)
                
                loss.backward()
                optimizer.step()
                
                acc = torch.where(Y.detach().argmax(1) == target.detach(), 1., 0.).mean()
                self.logger.add_scalar("Loss/train", loss, step)
                self.logger.add_scalar("Accuracy/train", acc, step)
                step += 1   
                
            self.save_state(model, optimizer, step, epoch)
                
    def train_plt_m(self, model, optimizer, trainloader, step=0):
        s = self.pt_settings
        criterion = CrossEntropyLoss()
        for epoch in range(1, s["NUM_EPOCHS"]+1):
            for data in trainloader:
                X = data[0].to(s["DEVICE"])
                target = torch.arange(model.T, device=s["DEVICE"]).repeat(s["BATCH_SIZE"])
                optimizer.zero_grad()
    
                Y, idx = model(X)
                loss = criterion(Y[~idx], target[~idx])

                loss.backward()
                optimizer.step()
    
                acc = torch.where(Y.detach().argmax(1) == target.detach(), 1., 0.).mean()
                self.logger.add_scalar("Loss/train", loss, step)
                self.logger.add_scalar("Accuracy/train", acc, step)
                step += 1
        
            self.save_state(model, optimizer, step, epoch)
                
    def train_mpr(self, model, optimizer, trainloader, step=0):
        s = self.pt_settings
        criterion = SmoothL1Loss() #MSELoss()
        pass
    def train_plt_AMP(self, model, trainloader):
        pass
    def train_plt_m_AMP(self, model, trainloader):
        pass
    def train_mpr_AMP(self, model, trainloader):
        pass


class Tuner(object):
    def __init__(self, s):
        self.model_name = s["MODEL_NAME"]
        self.name = f'({1 if s["HEAD_ONLY"] else 0},{1 if s["MLP_HEAD"] else 0},{1 if s["FREEZE_ENCODER"] else 0},{s["BATCH_SIZE"]},{s["LEARNING_RATE"]},{s["WEIGHT_DECAY"]}) - '+ self.model_name
        self.name += f'-[{s["COMMENT"]}]' if len(s["COMMENT"]) else ''
        self.settings = s
        self.port = 6007
        self.dir = f'./{self.model_name}'
        
    def save_state(self, model, optimizer, step, epoch):
        self.logger.flush()
        state = {
        'epoch': epoch,
            "step" : step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
            "pt_settings" : self.pt_settings,
                    }
        torch.save(state, self.dir + "/last_state")
         
    def imshow(self, trainloader, n=6):
        imgs = torch.stack([trainloader.dataset[i][0] for i in range(n)])
        img_grid = torchvision.utils.make_grid(imgs, nrow=n, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid)
        plt.axis('off')
        plt.show()
        plt.close()
    
    def load_test_split(self, cifar=False):
        s = self.settings
        if cifar:
            train = torchvision.datasets.CIFAR10(root='../CIFAR10', train=True,
                                                    download=True, transform=preprocess_aug)
            trainloader = torch.utils.data.DataLoader(train, batch_size=s["BATCH_SIZE"],
                                                      shuffle=True, num_workers=2,
                                                      drop_last=False, pin_memory=True)

            test = torchvision.datasets.CIFAR10(root='../CIFAR10', train=False,
                                                   download=True,transform=preprocess)
            testloader = torch.utils.data.DataLoader(test, batch_size=s["BATCH_SIZE"],
                                                      shuffle=True, num_workers=2,
                                                      drop_last=False, pin_memory=True)
        else:
            train = torchvision.datasets.STL10("../STL10/train/",
                                                 split="train",
                                                  transform=ft_train_transforms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=s["BATCH_SIZE"],
                                                      shuffle=True, num_workers=2,
                                                      drop_last=False, pin_memory=True)
            test = torchvision.datasets.STL10("../STL10/train/",
                                                 split="test",
                                                  transform=ft_test_transforms)
            testloader = torch.utils.data.DataLoader(test, batch_size=s["BATCH_SIZE"],
                                                      shuffle=True, num_workers=2,
                                                      drop_last=False, pin_memory=True)

        return trainloader, testloader
    
    def train(self):
        print("[...] - Initialize Fine-Tuning") 
        s = self.settings
        trainloader, testloader = self.load_test_split()
        self.logger = SummaryWriter(log_dir = "./ft_runs/"+self.name, comment=s["COMMENT"])
        subprocess.Popen(f'tensorboard --logdir=ft_runs --port={self.port}', shell=True)
        state = torch.load(self.dir + "/last_state")
        pts = state["pt_settings"]
        if pts["METHOD"] == "PLT":
            model = ContextLearner(pts)         
            model.load_state_dict(state['state_dict'])
            model.ft_mode(s)
            print("[...] - Starting FineTune Training ")
            self.fine_tune(model, trainloader, testloader)
            
        elif pts["METHOD"] == "PLT_MASKED":
            model = MaskedContextLearner(pts)
            model.load_state_dict(state['state_dict'])
            model.ft_mode(s)
            print("[...] - Starting FineTune Training ")
            self.fine_tune(model, trainloader, testloader)
            
        elif pts["METHOD"] == "MPR":
            model = MaskedPatchRegressor(pts)
            model.load_state_dict(state['state_dict'])
            model.ft_mode(s)
            print("[...] - Starting FineTune Training ")
            self.fine_tune(model, trainloader, testloader)
            
        else:
            raise NotImplemented 
             
    
    def fine_tune(self, model, trainloader, testloader):
        s = self.settings
        step = 0
        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], weight_decay=s["WEIGHT_DECAY"])
        for epoch in range(1, s["NUM_EPOCHS"]+1):
            for data in trainloader:
                X = data[0].to(s["DEVICE"])
                target = data[1].to(s["DEVICE"])

                optimizer.zero_grad()

                Y = model.ft_forward(X)
                loss = criterion(Y, target)

                loss.backward()
                optimizer.step()

                acc = torch.where(Y.detach().argmax(1) == target.detach(), 1., 0.).mean()
                self.logger.add_scalar("Loss/train", loss, step)
                self.logger.add_scalar("Accuracy/train", acc, step)
                step += 1
            if step%s["EVAL_STEPS"]==0:
                with torch.no_grad():
                    accs = []
                    losses = []
                    for data in testloader:
                        X = data[0].to(s["DEVICE"])
                        target = data[1].to(s["DEVICE"])

                        Y = model.ft_forward(X)
                        loss = criterion(Y, target)

                        accs += [torch.where(Y.detach().argmax(1) == target.detach(), 1., 0.).mean().item()]
                        losses += [loss.item()]

                    accs = np.mean(accs)
                    losses = np.mean(losses)
                    self.logger.add_scalar("Loss/val", losses, step)
                    self.logger.add_scalar("Accuracy/val", accs, step)


  
