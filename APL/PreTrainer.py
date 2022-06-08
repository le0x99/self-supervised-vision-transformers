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
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.nn.modules.loss import SmoothL1Loss, MSELoss
from PatchLocalizer import MaskedContextLearner
from copy import deepcopy as copy
from utils.base_models import PatchDropout, CNN


torch.manual_seed(42)


pt_train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=96, scale=(0.6, 1.0)),
                                       transforms.RandomGrayscale(p=0.2),
                                       torchvision.transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))], 0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                       ])
ft_train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
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
    def __init__(self, s, verbose=False):
        self.pt_settings = s
        self.name = f'[{s["PDO"]},{s["G"]},{s["CROP"]}]-[{s["LEARNING_RATE"]},{s["BATCH_SIZE"]}]-[{s["N_BLOCKS"]},{s["HEADS"]},{s["HIDDEN_MULT"]},{1 if s["NN_HEAD"] else 0}, {s["DROPOUT"]}]'
        self.name += "-[CNN]" if s["ENCODER"] == "CNN" else ''
        self.name += f'-[{s["COMMENT"]}]' if len(s["COMMENT"]) else ''
        self.dir = f'./{self.name}'
        try:
            os.mkdir(self.dir)
        except:
            pass
        if verbose:
            print(' - - - - Configuration - - - - '+ "\n")
            print(f'G                      : {s["G"]}')
            print(f'T                      : {s["G"]**2}')
            print(f'K                      : {s["CROP"]}')
            print(f'Patch Size Original    : {s["IMAGE_N"] // s["G"]}')
            print(f'Patch Size Cropped     : {s["IMAGE_N"] // s["G"] - s["CROP"]}')
            print(f'Dimensionality         : {3 * (s["IMAGE_N"] // s["G"] - s["CROP"])**2}')
            print(f'Information Loss       : {1 - (3 * (s["IMAGE_N"] // s["G"] - s["CROP"])**2) / (3 * (s["IMAGE_N"] // s["G"])**2)}'+ "\n")
            print(' - - - - Settings - - - - - - - '+ "\n")
            for k in s:
                print(k, (21 - len(k))*" ", ":", s[k])  
            
    def save_state(self, model, optimizer, scheduler, step, epoch):
        self.logger.flush()
        state = {
        'epoch': epoch,
            "step" : step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
            "pt_settings" : self.pt_settings,
            "scheduler" : scheduler.state_dict()
                    }
        torch.save(state, self.dir + "/last_state")
         
        
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
        
        self.pt_settings["steps_per_epoch"] = len(trainloader)
        return trainloader

    
    def train(self, continue_training=False):
        print(" - - - - Initialize Pre-Training - - - - ") if not continue_training else print(" - - - - Continue Pre-Training / Load from Checkpoint  - - - - ")
        s = self.pt_settings
        trainloader = self.load_pt_data()
        self.logger = SummaryWriter(log_dir = "./runs/"+self.name)
        if s["SPAWN_TENSORBOARD"]:
            subprocess.Popen(f'tensorboard --logdir=runs --port={self.port}', shell=True)
            
        step = 0
        start_epoch = 1
        model = MaskedContextLearner(self.pt_settings)
        optimizer = optim.Adam(model.parameters(), lr=s["LEARNING_RATE"], weight_decay=s["WEIGHT_DECAY"])
        warmup_epochs = int(s["NUM_EPOCHS"] * 0.05)
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5 , total_iters=warmup_epochs)
        cosan = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["NUM_EPOCHS"], eta_min= 0.1 * s["LEARNING_RATE"])
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup, cosan])
        if continue_training:
            state = torch.load(self.dir + "/last_state")
            state["scheduler"]["_schedulers"][-1]["T_max"] = s["NUM_EPOCHS"]
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state["scheduler"])
            step = state["step"]
            start_epoch = state["epoch"]
            del state
            torch.cuda.empty_cache()
        print(" - - - - Starting Training - - - - ")
        print(f'PID : {os.getpid()}')
        self.train_(model, optimizer, scheduler, trainloader, step, start_epoch)
            
    @torch.no_grad()
    def top_acc(self, Y, target, k):
        probs, classes = Y.topk(k)
        return torch.where(classes == target.unsqueeze(dim=1).expand_as(classes), 1., 0.).mean(0).cumsum(0)

            
    def train_(self, model, optimizer, scheduler, trainloader, step=0, start_epoch=1):
        s = self.pt_settings
        criterion = CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        torch.cuda.empty_cache()
        for epoch in range(start_epoch, s["NUM_EPOCHS"]+1):
            for data in trainloader:
                X = data[0].to(s["DEVICE"])
                target = torch.arange(model.T, device=s["DEVICE"]).repeat(s["BATCH_SIZE"])
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    logits, idx = model(X)
                    loss = criterion(logits[~idx], target[~idx])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    
                accs = self.top_acc(logits, target, 5)
                self.logger.add_scalar("Loss/train", loss, step)
                self.logger.add_scalar("Accuracy/train", accs[0], step)
                self.logger.add_scalar("Accuracy/train3", accs[2], step)
                self.logger.add_scalar("Accuracy/train5", accs[4], step)
                step += 1
                
            scheduler.step()
            self.save_state(model, optimizer, scheduler, step, epoch)
            lr = scheduler.get_last_lr()[0]
            self.logger.add_scalar("Loss/lr", lr, step)
                


class Tuner(object):
    def __init__(self, s):
        self.model_name = s["MODEL_NAME"]
        if self.model_name not in ["S-CNN", "S-ResNet"]:
            self.name = f'({1 if s["HEAD_ONLY"] else 0},{1 if s["MLP_HEAD"] else 0},{1 if s["FREEZE_ENCODER"] else 0},{s["BATCH_SIZE"]},{s["LEARNING_RATE"]},{s["WEIGHT_DECAY"]},{s["DROPOUT"]},{s["PDO"]}) - '+ self.model_name
            self.name += f'-[{s["COMMENT"]}]' if len(s["COMMENT"]) else ''
            self.name += "-[BENCHMARK]-" if s["BENCHMARK"] else ''
            self.dir = f'./{self.model_name}'
            self.baseline = False
        else:
            self.name = self.model_name
            self.name += f'-[{s["COMMENT"]}]' if len(s["COMMENT"]) else ''
            self.baseline = True
        self.port = 6008
        self.settings = s
        
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
         
    
    def load_test_split(self):
        s = self.settings
        npc =  s["NPC"]
        train = torchvision.datasets.STL10("../STL10/train/",
                                             split="train",
                                              transform=ft_train_transforms)
        train = self.reduce_training_size(train, npc) if npc < 500 else train
        trainloader = torch.utils.data.DataLoader(train, batch_size=s["BATCH_SIZE"],
                                                              shuffle=True, num_workers=4,
                                                              drop_last=False, pin_memory=False)

        test = torchvision.datasets.STL10("../STL10/train/",
                                             split="test",
                                              transform=ft_test_transforms)
        testloader = torch.utils.data.DataLoader(test, batch_size=s["BATCH_SIZE"],
                                                  shuffle=True, num_workers=4,
                                                  drop_last=True, pin_memory=False)
        return trainloader, testloader
            
        
    def reduce_training_size(self, dataset, npc, n_classes=10):
        np.random.seed(self.settings["rng"])
        classes = pd.Series([ _[1] for _ in dataset ]).reset_index().rename(columns={ 0 : "class"})
        idx = []
        for c in range(n_classes):
            idx += classes[classes["class"] == c].sample(npc, replace=False).index.to_list()
        np.random.shuffle(idx)
        
        return torch.utils.data.Subset(dataset, idx)
    
    def train(self):
        print(" - - - - Initialize Fine-Tuning - - - - ") 
        s = self.settings
        trainloader, testloader = self.load_test_split()
        self.logger = SummaryWriter(log_dir = "./ft_runs/"+self.name, comment=s["COMMENT"])
        if self.settings["SPAWN_TENSORBOARD"]:
            subprocess.Popen(f'tensorboard --logdir=ft_runs --port={self.port}', shell=True)
        if not self.baseline:
            state = torch.load(self.dir + "/last_state")
            pts = copy(state["pt_settings"])
            pts["DROPOUT"] = s["DROPOUT"]
            model = MaskedContextLearner(pts)
            if not self.settings["BENCHMARK"]:
                print("Loading Model...")
                model.load_state_dict(state['state_dict'])
                model.ft_mode(s)
            del state
            torch.cuda.empty_cache()
        else:
            if "S-CNN" in self.name:
                model = CNN().to(s["DEVICE"])
                model.ft_forward = model.forward
            elif "S-ResNet" in self.name:
                model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(s["DEVICE"])
                model.ft_forward = model.forward
        print(" - - - - Starting FineTune Training - - - - ")
        self.fine_tune(model, trainloader, testloader)

            
    @torch.no_grad()
    def top_acc(self, Y, target, k):
        probs, classes = Y.topk(k)
        return torch.where(classes == target.unsqueeze(dim=1).expand_as(classes), 1., 0.).mean(0).cumsum(0)
    
    @torch.no_grad()
    def test_model(self, model, criterion, testloader, epoch):
        s = self.settings
        name = "test" if s["NPC"] == 500 else f'test-{s["NPC"]}'
        accs = []
        losses = []
        for data in testloader:
            X = data[0].to(s["DEVICE"])
            target = data[1].to(s["DEVICE"])

            logits = model.ft_forward(X)
            loss = criterion(logits, target)

            accs += [self.top_acc(logits, target, 5).unsqueeze(-1)]
            losses += [loss.item()]

        accs = torch.cat(accs, -1).mean(-1)
        losses = np.mean(losses)
        self.logger.add_scalar(f'{name}/Acc@1', accs[0], epoch)
        self.logger.add_scalar(f'{name}/Acc@3', accs[2], epoch)
        self.logger.add_scalar(f'{name}/Acc@5', accs[4], epoch)
        self.logger.add_scalar(f'{name}/NLL', losses, epoch)
        
    def fine_tune(self, model, trainloader, testloader):
        s = self.settings
        step = 0
        name = "train" if s["NPC"] == 500 else f'train-{s["NPC"]}'
        criterion = CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        optimizer = optim.Adam(model.parameters(),
                              lr=s["LEARNING_RATE"],
                              weight_decay=s["WEIGHT_DECAY"])
        warmup_epochs = int(s["NUM_EPOCHS"] * 0.1)
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5 , total_iters=warmup_epochs)
        cosan = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=s["NUM_EPOCHS"],
                                                     eta_min= 0.1 * s["LEARNING_RATE"])
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup, cosan])
        for epoch in range(1, s["NUM_EPOCHS"]+1):
            for data in trainloader:
                X = data[0].to(s["DEVICE"])
                target = data[1].to(s["DEVICE"])

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model.ft_forward(X)
                    loss = criterion(logits, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                accs = self.top_acc(logits, target, 5)
                self.logger.add_scalar(f'{name}/NLL', loss, step)
                self.logger.add_scalar(f'{name}/Acc@1', accs[0], step)
                self.logger.add_scalar(f'{name}/Acc@3', accs[2], step)
                self.logger.add_scalar(f'{name}/Acc@5', accs[4], step)
                step += 1
                
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            self.logger.add_scalar("train/lr", lr, step)
            
            if epoch%s["EVAL_EPS"]==0:
                model.eval()
                if not self.baseline:
                    model.pdo = PatchDropout(0.)
                self.test_model(model, criterion, testloader, epoch)
                if not self.baseline:
                    model.pdo = PatchDropout(self.settings["PDO"])
                model.train()



  
