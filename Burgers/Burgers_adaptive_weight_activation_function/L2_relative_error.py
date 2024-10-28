from net import Net_adaptive
from net import Net
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os



os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sin_weight = np.load('result_data/sin-weight_20000(L2norm).npy')
tanh_weight = np.load('result_data/tanh-weight_20000(L2norm).npy')
GELU_weight = np.load('result_data/GELU-weight_20000(L2norm).npy')
Swish_weight = np.load('result_data/Swish-weight_20000(L2norm).npy')
Softplus_weight = np.load('result_data/Softplus-weight_20000(L2norm).npy')

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


PINN = Net_adaptive(seq_net=[2, 20, 20, 20, 20, 20, 20, 1],a=a,b=b,c=c,d=d,e=e).to(device)
# PINN = Net(seq_net=[2, 20, 20, 20, 20, 20, 20, 1]).to(device)

PINN.load_state_dict(torch.load('result_data/PINN2(20000)_(L2norm).pth'))

def L2_relative_error():

    data = scipy.io.loadmat('./Data/burgers_shock.mat')

    Exact = np.real(data['usol']).T
    u_star = Exact.flatten()[:, None]
    u_star = torch.tensor(u_star, dtype=torch.float32, device=device, requires_grad=True)

    t = data['t']
    x = data['x']
    X, T = np.meshgrid(x, t)
    s_shape = X.shape
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # 在水平方向上平铺(25600, 2)
    X_star = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True)


    x_pred = X_star[:, 0:1]
    t_pred = X_star[:, 1:2]
    u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))

    L2_relative_error=torch.norm(u_pred-u_star)/torch.norm(u_star)
    print('L2_relative_error:{:.4f}%'.format(L2_relative_error.item()*100))
    # np.save('./result_data/l2_relative_error({})_(Sigmoid).npy'.format(20000),
    #         'L2_relative_error(Sigmoid):{:.4f}%'.format(L2_relative_error.item()*100))

def relative_error_visulization():
    epochs=8000


    data = scipy.io.loadmat('./Data/burgers_shock.mat')

    Exact = np.real(data['usol']).T
    u_star = Exact.flatten()[:, None]
    u_star = torch.tensor(u_star, dtype=torch.float32, device=device, requires_grad=True)

    t = data['t']
    x = data['x']
    X, T = np.meshgrid(x, t)
    s_shape = X.shape
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # 在水平方向上平铺(25600, 2)
    X_star = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True)


    x_pred = X_star[:, 0:1]
    t_pred = X_star[:, 1:2]
    u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))

    plt.cla()
    mse_test = abs(u_pred - u_star)/max(abs(u_star))
    plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
                   mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(0, 0.2)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('./result_plot/Burger1d_relative_error_{}(L2norm).png'.format(epochs), bbox_inches='tight', format='png')
    plt.close()

if __name__=='__main__':
    # relative_error_visulization()
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

