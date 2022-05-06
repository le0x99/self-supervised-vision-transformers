import torch
import torch.nn as nn
import torch.nn.functional as F
import sys;sys.path.insert(0,'..')
from utils.base_models import *


class ContextLearner(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.G = s["G"]
        self.T = s["G"] ** 2
        if s["ENCODER"] == "ResNet":
            self.encoder = ResNetEncoder(s).to(s["DEVICE"])
        elif s["ENCODER"] == "CNN":
            self.encoder = CNNEncoder(s).to(s["DEVICE"])
        elif s["ENCODER"] == "MECHANICAL":
            self.encoder = Encoder(s).to(s["DEVICE"])
        self.patch_emb_dim = self.encoder.emb_dim
        s["patch_emb_dim"] = self.encoder.emb_dim
        if s["NN_HEAD"]:
            self.head = MLP(self.encoder.emb_dim, self.T).to(s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, self.T, device=s["DEVICE"])
        self.contextualize = [Transformer(self.patch_emb_dim,
                                                 s["HEADS"],
                                                 s["DROPOUT"],
                                                 s["HIDDEN_MULT"]).to(device=s["DEVICE"]) for _ in range(s["N_BLOCKS"])]

        self.contextualize = nn.Sequential(*self.contextualize)
        
        self.project_patch_emb = nn.Linear(self.patch_emb_dim,
                                           self.patch_emb_dim,
                                           bias=False,
                                           device=s["DEVICE"]) if s["PROJECT_PATCH_EMB"] else nn.Identity()
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    def ft_mode(self, ft_s):
        if ft_s["HEAD_ONLY"]:
            for param in self.parameters():
                param.requires_grad = False
        if ft_s["MLP_HEAD"]:
            self.head = MLP(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        if ft_s["FREEZE_ENCODER"]:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        B, T = X.shape[0], self.T
        X = self.encoder(X)
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        Y = self.head(X)
        Y = Y.view(B * T, T)
        return Y
    
    
    def ft_forward(self, X):
        X = self.encoder(X)
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        Y = X.mean(1)
        Y = self.head(Y)
        return Y
        
class MaskedContextLearner(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.G = s["G"]
        self.T = s["G"] ** 2
        self.settings = s
        if s["ENCODER"] == "CNN":
            self.encoder = MaskedCNNEncoder(s).to(s["DEVICE"])
        else:
            self.encoder = MaskedEncoder(s).to(s["DEVICE"])
        self.patch_emb_dim = self.encoder.emb_dim
        s["patch_emb_dim"] = self.encoder.emb_dim
        if s["NN_HEAD"]:
            self.head = MLP(self.encoder.emb_dim, self.T).to(s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, self.T, device=s["DEVICE"])
        self.contextualize = [Transformer(self.patch_emb_dim,
                                                  s["HEADS"],
                                                  s["DROPOUT"],
                                                  s["HIDDEN_MULT"]).to(device=s["DEVICE"]) for _ in range(s["N_BLOCKS"])]
        self.contextualize = nn.Sequential(*self.contextualize)
        
        self.project_patch_emb = nn.Linear(self.patch_emb_dim,
                                           self.patch_emb_dim,
                                           bias=False,
                                           device=s["DEVICE"]) if s["PROJECT_PATCH_EMB"] else nn.Identity()
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    def ft_mode(self, ft_s):
        self.encoder = Encoder(self.settings).to(ft_s["DEVICE"])
        if ft_s["HEAD_ONLY"]:
            for param in self.parameters():
                param.requires_grad = False
        if ft_s["MLP_HEAD"]:
            self.head = MLP(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        if ft_s["FREEZE_ENCODER"]:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        B, T = X.shape[0], self.T
        X, idx = self.encoder(X)
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        Y = self.head(X)
        Y = Y.view(B * T, T)
        return Y, idx
    
    
    def ft_forward(self, X):
        X = self.encoder(X)
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        Y = X.mean(1)
        Y = self.head(Y)
        return Y
    
class MaskedPatchRegressor(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.G = s["G"]
        self.T = s["G"] ** 2
        self.settings = s
        if s["ENCODER"] != "MECHANICAL":
            raise NotImplemented
        self.encoder = MaskedEncoderForRegression(s).to(s["DEVICE"])
        self.patch_emb_dim = self.encoder.emb_dim
        s["patch_emb_dim"] = self.encoder.emb_dim
        if s["NN_HEAD"]:
            self.head = MLP(self.encoder.emb_dim, self.encoder.emb_dim).to(s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, self.encoder.emb_dim, device=s["DEVICE"])
        self.contextualize = [Transformer(self.patch_emb_dim,
                                                  s["HEADS"],
                                                  s["DROPOUT"],
                                                  s["HIDDEN_MULT"]).to(device=s["DEVICE"]) for _ in range(s["N_BLOCKS"])]
        self.contextualize = nn.Sequential(*self.contextualize)
        
        self.project_patch_emb = nn.Linear(self.patch_emb_dim,
                                           self.patch_emb_dim,
                                           bias=False,
                                           device=s["DEVICE"]) if s["PROJECT_PATCH_EMB"] else nn.Identity()
        self.pos_emb = nn.Embedding(self.T, self.patch_emb_dim, device=s["DEVICE"])
        self.positions = torch.arange(self.T, device=s["DEVICE"])
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    def ft_mode(self, ft_s):
        self.encoder = Encoder(self.settings).to(ft_s["DEVICE"])
        if ft_s["HEAD_ONLY"]:
            for param in self.parameters():
                param.requires_grad = False
        if ft_s["MLP_HEAD"]:
            self.head = MLP(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        else:
            self.head = nn.Linear(self.patch_emb_dim, 10).to(ft_s["DEVICE"])
        if ft_s["FREEZE_ENCODER"]:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, X):
        X, idx, targets = self.encoder(X)
        B, T, D = X.size()
        positions = self.pos_emb(self.positions)[None, :, :].expand(B, T, D)
        X = X + positions
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        X = X.mean(1)
        Y = self.head(X)
        #Y = Y.view(B * T, T)
        #return Y, idx
        return Y, targets
    
    def ft_forward(self, X):
        X = self.encoder(X)
        B, T, D = X.size()
        positions = torch.arange(T)
        positions = self.pos_emb(positions)[None, :, :].expand(B, T, D)
        X = X + positions
        X = self.project_patch_emb(X)
        X = self.contextualize(X)
        Y = X.mean(1)
        Y = self.head(Y)
        return Y
    
class MLP(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.fc1 = nn.Linear(in_, 3 * in_)
        self.fc2 = nn.Linear(3 * in_, out_)
        self.do = nn.Dropout(0.125)
        
    def forward(self, X0):
        X0 = self.do(X0)
        X1 = self.do(F.relu(self.fc1(X0)))
        X2 = self.fc2(X1)
        return X2
