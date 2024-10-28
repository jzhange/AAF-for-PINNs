import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class Net_adaptive(nn.Module):
    def __init__(self, seq_net,a=0,b=0,c=0,d=0,e=0,p1=0.,p2=0.,beta=0.,name='MLP'):
        super().__init__()
        self.features = OrderedDict()
        self.weight_list=list()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)

        self.a=a
        self.b =b
        self.c =c
        self.d = d
        self.e = e
        self.list=[self.a,self.b,self.c,self.d,self.e]

        self.p1=p1
        self.p2 = p2
        self.beta = beta


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                # nn.init.xavier_normal_(m.weight)



    def activation_assembly(self,x):
        denominator=(self.a+self.b+self.c+self.d+self.e)
        assem=((self.a/denominator)*torch.sin(x)
               +(self.b/denominator)*F.tanh(x)
               +(self.c/denominator) * F.gelu(x)
               +(self.d/denominator)*F.silu(x)
               +(self.e/denominator)*F.softplus(x))

        return assem

    def activation_assembly_ABU_PINNS(self,x):
        exp_a=torch.exp(self.a)
        exp_b = torch.exp(self.b)
        exp_c = torch.exp(self.c)
        exp_d = torch.exp(self.d)
        exp_e = torch.exp(self.e)
        exp_total=(exp_a+exp_b+exp_c+exp_d+exp_e)
        assem=((exp_a/exp_total)*torch.sin(x)
               +(exp_b/exp_total)*torch.tanh(x)
               +(exp_c/exp_total) * F.gelu(x)
               +(exp_d/exp_total)*F.silu(x)
               +(exp_e/exp_total)*F.softplus(x))
        return assem

    def activation_assembly_identity(self,x):
        assem=(self.a*torch.sin(x)
               +self.b*F.tanh(x)
               +self.c* F.gelu(x)
               +self.d*F.silu(x)
               +self.e*F.softplus(x))
        return assem

    def activation_assembly_sigmoid(self,x):
        assem=(torch.sigmoid(self.a)*torch.sin(x)
               +torch.sigmoid(self.b)*F.tanh(x)
               +torch.sigmoid(self.c) * F.gelu(x)
               +torch.sigmoid(self.d)*F.silu(x)
               +torch.sigmoid(self.e)*F.softplus(x))
        return assem

    def activation_assembly_L1norm(self,x):
        denominator=(abs(self.a)+abs(self.b)+abs(self.c)+abs(self.d)+abs(self.e))
        assem=((self.a/denominator)*torch.sin(x)
               +(self.b/denominator)*F.tanh(x)
               +(self.c/denominator) * F.gelu(x)
               +(self.d/denominator)*F.silu(x)
               +(self.e/denominator)*F.softplus(x))
        return assem

    def activation_assembly_L2norm(self,x):
        weights_combined=torch.cat([self.a,self.b,self.c,self.d,self.e])
        denominator=torch.norm(weights_combined)
        assem=((self.a/denominator)*torch.sin(x)
               +(self.b/denominator)*F.tanh(x)
               +(self.c/denominator) * F.gelu(x)
               +(self.d/denominator)*F.silu(x)
               +(self.e/denominator)*F.softplus(x))
        return assem

    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1: break
            i += 1
            x = self.activation_assembly_L2norm(x)
        return x



class Net(nn.Module):
    def __init__(self, seq_net, name='MLP',activation=torch.tanh):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
        self.features = nn.ModuleDict(self.features)
        self.active=activation


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        # x = x.view(-1, 2)
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1: break
            i += 1
            x = self.active(x)
        return x

