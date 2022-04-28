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
    

class CNNEncoder2(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.crop = CenterCrop(self.N2 - s["CROP"]) if s["CROP"] > 0 else nn.Identity()
        self.emb_dim = 16 * ((self.N2 - s["CROP"])// 2 )**2
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.AvgPool2d((2,2))
        self.conv2 = nn.Conv2d(6, 16 , 3, padding=1)
        self.do = nn.Dropout2d(s["ENCODER_DO"])
        self.fc = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, X):
        B, T, E = X.shape[0], self.T, self.emb_dim
        N2 = self.N2
        X = X.unfold(2, N2, N2).unfold(3, N2, N2)
        X = self.crop(X)
        X = X.flatten(2,3)
        X = X.transpose(1,2)
        X = X.contiguous().view(B * T, 3, N2, N2)
        
        x = F.relu(self.bn1(self.conv1(X)))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.do(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn2(x)
        x = self.do(x)
        x = x.flatten(1)
        x = self.fc(x)
        X = x.view(B, T, E)
        
        
        return X
    
class CNNEncoder(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.T = s["G"] ** 2
        self.N2 = s["IMAGE_N"] // s["G"]
        self.K =  s["CROP"]
        self.crop = CenterCrop(self.N2 - s["CROP"]) if s["CROP"] > 0 else nn.Identity()
        self.do = nn.Dropout2d(s["ENCODER_DO"])     
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(6, 32 , 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 256, 3, padding=1, stride=1)
        self.emb_dim = 256
        
    @torch.no_grad()
    def extract(self, X):
        B, T, E = X.shape[0], self.T, self.emb_dim
        N2, K = self.N2, self.K
        N2K = N2 - K
        X = X.unfold(2, N2, N2).unfold(3, N2, N2)
        X = self.crop(X)
        X = X.flatten(2,3)
        X = X.transpose(1,2)
        X = X.contiguous().view(B * T, 3, N2K, N2K)

    def forward(self, X):
        X = self.extract(X)
        X = F.relu(self.conv1(X))
        X = self.do(X)
        X = F.relu(self.conv2(X))
        X = self.do(X)
        X = F.relu(self.conv3(X))
        X = self.do(X)
        X = F.relu(self.conv4(X))
        X = X.amax((2,3))
        X = X.view(B, T, E)
        return X
    
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
    
class MaskedEncoder(nn.Module):
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
        X[idx] = torch.tanh(torch.randn(N2K, device=self.dev))
        X = X.view(B, T, N2K)
        return X, idx

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

    
# Supervised Baselines

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(3200, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.do = nn.Dropout2d(0.25)
        self.do_fc = nn.Dropout(0.25)
        self.pool = nn.AvgPool2d(2, 2)
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
 
    def ft_forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.do(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.do(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.do(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        x = self.do(x)
        x = self.fc3(x)
        return x
    
    
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = conv_block(3, 64,pool=True)
        self.conv2 = conv_block(64, 128, pool=True) # output: 128 x 24 x 24
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True) # output: 256 x 12 x 12
        self.conv4 = conv_block(256, 512, pool=True) # output: 512 x 6 x 6
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.25),
                                        nn.Linear(512, 10))
        
    def total_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
        
    def ft_forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

