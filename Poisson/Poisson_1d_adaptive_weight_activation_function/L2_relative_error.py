from net import Net_adaptive
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def Ground_truth(x):
    return torch.sin(0.7*x)+torch.cos(1.5*x)-0.1*x

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sin_weight = np.load('result_data/sin-weight_8000(Weighted_Average).npy')
tanh_weight = np.load('result_data/tanh-weight_8000(Weighted_Average).npy')
GELU_weight = np.load('result_data/GELU-weight_8000(Weighted_Average).npy')
Swish_weight = np.load('result_data/Swish-weight_8000(Weighted_Average).npy')
Softplus_weight = np.load('result_data/Softplus-weight_8000(Weighted_Average).npy')

print('final sin_weight:{:.04f}'.format(sin_weight[-1]))
print('final tanh_weight:{:.04f}'.format(tanh_weight[-1]))
print('final GELU_weight:{:.04f}'.format(GELU_weight[-1]))
print('final Swish_weight:{:.04f}'.format(Swish_weight[-1]))
print('final Softplus_weight:{:.04f}'.format(Softplus_weight[-1]))

a = torch.tensor([sin_weight[-1]], dtype=torch.float32, device=device)
b = torch.tensor([tanh_weight[-1]], dtype=torch.float32, device=device)
c = torch.tensor([GELU_weight[-1]], dtype=torch.float32, device=device)
d = torch.tensor([Swish_weight[-1]], dtype=torch.float32, device=device)
e = torch.tensor([Softplus_weight[-1]], dtype=torch.float32, device=device)


PINN = Net_adaptive(seq_net=[1,50,50,50,1],a=a,b=b,c=c,d=d,e=e).to(device)

PINN.load_state_dict(torch.load('result_data/PINN2(8000)_(Weighted_Average).pth'))

def L2_relative_error():

    xx = torch.linspace(-10, 10, 10000).reshape(-1, 1).to(device)
    yy = PINN(xx)
    zz = Ground_truth(xx)

    L2_relative_error=torch.norm(yy-zz)/torch.norm(zz)
    print('L2_relative_error:{:.4f}%'.format(L2_relative_error.item()*100))
    np.save('./result_data/l2_relative_error({})_(Weighted_Average).npy'.format(8000),
            'L2_relative_error(Weighted_Average):{:.4f}%'.format(L2_relative_error.item()*100))

if __name__=='__main__':
    L2_relative_error()
    # l2_Softmax=np.load('./result_data/l2_relative_error(20000)_(Softmax).npy')
    # l2_adap_weight = np.load('./result_data/l2_relative_error(20000)_(adap_weight).npy')
    # l2_fixed = np.load('./result_data/l2_relative_error(20000)_(fixed).npy')
    # l2_Identity = np.load('./result_data/l2_relative_error(20000)_(Identity).npy')
    # l2_L1norm = np.load('./result_data/l2_relative_error(20000)_(L1norm).npy')
    # l2_L2norm = np.load('./result_data/l2_relative_error(20000)_(L2norm).npy')
    # l2_Sigmoid = np.load('./result_data/l2_relative_error(20000)_(Sigmoid).npy')
    # print(l2_fixed)
    # print(l2_Sigmoid)
    # print(l2_L1norm)
    # print(l2_Identity)
    # print(l2_Softmax)
    # print(l2_adap_weight)
    # print(l2_L2norm)

