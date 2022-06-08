import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from base_models import *



class MaskedContextLearner(nn.Module):
    def __init__(self, s):
        super().__init__()
        G, device = s["G"], s["DEVICE"]
        self.G = G
        self.T = G ** 2
        self.settings = s
        self.pdo = PatchDropout(s["PDO"])
        if s["ENCODER"] == "CNN":
            self.encoder = CNNEncoder(s)
        else:
            self.encoder = Encoder(s)
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
        if ft_s["PDO"] != "auto":
            self.pdo = PatchDropout(ft_s["PDO"])
            
        
    def forward(self, X):
        B, T = X.shape[0], self.T

        X = self.encoder(X)
    
        X, idx = self.pdo(X)
        X = self.contextualize(X)
        Y = self.head(X)

        Y = Y.view(B * T, T)
        return Y, idx
    
    
    def ft_forward(self, X):
        X = self.encoder(X)
        X, _ = self.pdo(X)
        X = self.contextualize(X)
        Y = X.mean(1)
        Y = self.head(Y)
        return Y
    
    
    
class MLP(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.fc1 = nn.Linear(in_, 3 * in_)
        self.fc2 = nn.Linear(3 * in_, out_)
        self.do = nn.Dropout(0.1)
        
    def forward(self, X0):
        X1 = self.do(F.relu(self.fc1(X0)))
        X2 = self.fc2(X1)
        return X2
