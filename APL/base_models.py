import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop
import torchvision

class Attention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads

        self.W_k    = nn.Linear(emb, emb, bias=False)
        self.W_q = nn.Linear(emb, emb, bias=False)
        self.W_v  = nn.Linear(emb, emb, bias=False)
        self.W_u = nn.Linear(emb, emb)

    def forward(self, X):

        b, t, e = X.size()
        h = self.heads
        #assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'
        # chunksize of e, i.e. head dim
        s = e // h
        # query, key, value model
        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)
        # split
        K = K.view(b, t, h, s)
        Q = Q.view(b, t, h, s)
        V = V.view(b, t, h, s)
        # prepare for dot product and scale (pbloem)
        K = K.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))
        Q = Q.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))
        V = V.transpose(1,2).contiguous().view(b * h, t, s) / (e ** (1/4))

        W = Q@K.transpose(1,2)
        W = F.softmax(W, dim=2)

        #assert W.size() == (b*h, t, t)

        Y = W@V
        Y = Y.view(b, h, t, s)

        # re-arange and unify heads 
        Y = Y.transpose(1, 2).contiguous().view(b, t, s * h)
        Y = self.W_u(Y)
        return Y
    
class Transformer(nn.Module):

    def __init__(self, emb=2048, heads=32,dropout=0.25,ff_hidden_mult=2):
        super().__init__()
        self.attention = Attention(emb, heads=heads)
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(inplace=True),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x
    
class Encoder(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.crop = CenterCrop(self.N2 - s["CROP"]) if s["CROP"] > 0 else nn.Identity()
        self.emb_dim = 3*(s["IMAGE_N"] // s["G"] - s["CROP"])**2
        
    @torch.no_grad()
    def forward(self, X):
        N2 = self.N2
        X = X.unfold(2, N2, N2).unfold(3, N2, N2)
        X = self.crop(X)
        X = X.flatten(2,3)
        X = X.transpose(1,2)
        X = X.flatten(2,4)
        return X
      
      
class PatchDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.sampler = torch.distributions.binomial.Binomial(1, torch.tensor([p]))
    @torch.no_grad()
    def forward(self, X):
        B, T, E = X.size()
        idx = self.sampler.sample(sample_shape=(B * T,)).bool().ravel()
        X.view(B * T, E)[idx] = 0.
        return X, idx


## Learnable Encoders, not very useful in practice.
    
class ResNetEncoder(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.K =  s["CROP"]
        self.crop = CenterCrop(self.N2 - s["CROP"]) if s["CROP"] > 0 else nn.Identity()
        self.resnet = torchvision.models.resnet18(pretrained=False, num_classes=512)
        self.resnet.fc = nn.Identity()
        self.emb_dim = 512
        
    def forward(self, X):
        with torch.no_grad():
            B, T, E = X.shape[0], self.T, self.emb_dim
            N2, K = self.N2, self.K
            N2K = N2 - K
            X = X.unfold(2, N2, N2).unfold(3, N2, N2)
            X = self.crop(X)
            X = X.flatten(2,3)
            X = X.transpose(1,2)
            X = X.contiguous().view(B * T, 3, N2K, N2K)
        
        X = self.resnet(X)
        X = X.view(B, T, E)   
        return X

class CNNEncoder(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.K =  s["CROP"]
        self.N2K = self.N2 - self.K
        if s["CROP"] > 0:
            self.crop = CenterCrop(self.N2 - s["CROP"])
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, stride=1, device=s["DEVICE"])
            self.conv2 = nn.Conv2d(64, 128 , 3, padding=1, stride=1, device=s["DEVICE"])
        else:
            self.crop = nn.Identity()
            self.conv1 = nn.Conv2d(3, 64, 3, device=s["DEVICE"])
            self.conv2 = nn.Conv2d(64, 128 , 3, device=s["DEVICE"])
        self.do = nn.Dropout2d(s["ENCODER_DO"])  
        self.emb_dim = 128

    def forward(self, X):
        B, C, H, W = X.size()
        T = self.T
        N2 = self.N2
        E = self.emb_dim
        
        with torch.no_grad():
            X = X.unfold(2, N2, N2).unfold(3, N2, N2)
            X = self.crop(X)
            X = X.flatten(2,3)
            X = X.transpose(1,2)
            X = X.flatten(0,1)
            
        X = F.relu(self.conv1(X))
        X = self.do(X)
        X = F.relu(self.conv2(X))
        X = X.mean((2,3))
        X = X.view(B, T, E) 
        return X
    

    
class MaskedEncoderForRegression(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.crop = CenterCrop(self.N2 - s["CROP"]) if s["CROP"] > 0 else nn.Identity()
        self.emb_dim = 3*(s["IMAGE_N"] // s["G"] - s["CROP"])**2
        self.sampler = torch.distributions.binomial.Binomial(1, torch.tensor([self.T**-1]))
        self.dev = s["DEVICE"]
        
    @torch.no_grad()
    def forward(self, X): 
        N2, N2K, T, B  = self.N2, self.emb_dim, self.T, X.shape[0]
        
        X = X.unfold(2, N2, N2).unfold(3, N2, N2)
        X = self.crop(X)
        X = X.flatten(2,3)
        X = X.transpose(1,2)
        X = X.flatten(2,4)
        X = X.view(B * T, N2K)
        idx = self.sampler.sample(sample_shape=(B * T,)).bool().ravel()
        targets = X[idx].clone()
        X[idx] = torch.tanh(torch.randn(N2K, device=self.dev))
        X = X.view(B, T, N2K)
        return X, idx, targets

    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
