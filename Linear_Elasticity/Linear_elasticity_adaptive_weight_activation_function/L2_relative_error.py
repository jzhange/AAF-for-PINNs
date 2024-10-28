from net import Net
from net import Net_adaptive
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from math import pi

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sin_weight = np.load('result_data/sin-weight_8000(L2norm).npy')
tanh_weight = np.load('result_data/tanh-weight_8000(L2norm).npy')
GELU_weight = np.load('result_data/GELU-weight_8000(L2norm).npy')
Swish_weight = np.load('result_data/Swish-weight_8000(L2norm).npy')
Softplus_weight = np.load('result_data/Softplus-weight_8000(L2norm).npy')

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


PINN = Net_adaptive(seq_net=[2, 50, 50, 50, 50, 5],a=a,b=b,c=c,d=d,e=e).to(device)

PINN.load_state_dict(torch.load('result_data/PINN(8000)_(L2norm).pth'))

def L2_relative_error():
    def d(f, x):
        return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def fun_exact(x, y, Lambda, mu, q):
        ux = torch.cos(2 * pi * x) * torch.sin(pi * y)
        uy = torch.sin(pi * x) * q * (y ** 4) / 4
        exx = d(ux, x)
        eyy = d(uy, y)
        exy = (d(ux, y) + d(uy, x)) / 2
        sigmaxx = (Lambda + 2 * mu) * exx + Lambda * eyy
        sigmayy = (Lambda + 2 * mu) * eyy + Lambda * exx
        sigmaxy = (2 * mu) * exy
        return ux, uy, sigmaxx, sigmayy, sigmaxy, exx, eyy, exy

    Lambda, mu, q = 1, 0.5, 4


    x_test = torch.linspace(0, 1, 200, device=device, requires_grad=True)
    y_test = torch.linspace(0, 1, 200, device=device, requires_grad=True)
    xs, ys = torch.meshgrid([x_test, y_test])
    s1 = xs.shape
    x_test = xs.reshape(-1, 1)
    y_test = ys.reshape(-1, 1)
    ux_test, uy_test, sigmaxx_test, sigmayy_test, sigmaxy_test, exx_test, eyy_test, exy_test \
        = fun_exact(x_test, y_test, Lambda, mu, q)


    solution_pred \
        = PINN(torch.cat([x_test, y_test], dim=1))
    exx_pred = d(solution_pred[:,0:1], x_test)
    eyy_pred = d(solution_pred[:,1:2], y_test)
    exy_pred = (d(solution_pred[:,0:1], y_test) + d(solution_pred[:,1:2], x_test)) / 2

    ux_pred = solution_pred[:,0:1]
    uy_pred = solution_pred[:,1:2]
    sigmaxx_pred = solution_pred[:,2:3]
    sigmayy_pred = solution_pred[:,3:4]
    sigmaxy_pred = solution_pred[:,4:5]

    epochs=8000

    L2_relative_error_ux=torch.norm(ux_pred-ux_test)/torch.norm(ux_test)
    print('L2_relative_error_ux:{:.4f}%'.format(L2_relative_error_ux.item()*100))
    # np.save('./result_data/l2_relative_error_ux({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_ux(Sigmoid):{:.4f}%'.format(L2_relative_error_ux.item()*100))

    L2_relative_error_uy=torch.norm(uy_pred-uy_test)/torch.norm(uy_test)
    print('L2_relative_error_uy:{:.4f}%'.format(L2_relative_error_uy.item()*100))
    # np.save('./result_data/l2_relative_error_uy({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_uy(Sigmoid):{:.4f}%'.format(L2_relative_error_uy.item()*100))

    L2_relative_error_sigmaxx=torch.norm(sigmaxx_pred-sigmaxx_test)/torch.norm(sigmaxx_test)
    print('L2_relative_error_sigmaxx:{:.4f}%'.format(L2_relative_error_sigmaxx.item()*100))
    # np.save('./result_data/l2_relative_error_sigmaxx({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_sigmaxx(Sigmoid):{:.4f}%'.format(L2_relative_error_sigmaxx.item()*100))

    L2_relative_error_sigmayy=torch.norm(sigmayy_pred-sigmayy_test)/torch.norm(sigmayy_test)
    print('L2_relative_error_sigmayy:{:.4f}%'.format(L2_relative_error_sigmayy.item()*100))
    # np.save('./result_data/l2_relative_error_sigmayy({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_sigmayy(Sigmoid):{:.4f}%'.format(L2_relative_error_sigmayy.item()*100))

    L2_relative_error_sigmaxy=torch.norm(sigmaxy_pred-sigmaxy_test)/torch.norm(sigmaxy_test)
    print('L2_relative_error_sigmaxy:{:.4f}%'.format(L2_relative_error_sigmaxy.item()*100))
    # np.save('./result_data/l2_relative_error_sigmaxy({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_sigmaxy(Sigmoid):{:.4f}%'.format(L2_relative_error_sigmaxy.item()*100))

    L2_relative_error_exx=torch.norm(exx_pred-exx_test)/torch.norm(exx_test)
    print('L2_relative_error_exx:{:.4f}%'.format(L2_relative_error_exx.item()*100))
    # np.save('./result_data/l2_relative_error_exx({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_exx(Sigmoid):{:.4f}%'.format(L2_relative_error_exx.item()*100))

    L2_relative_error_eyy=torch.norm(eyy_pred-eyy_test)/torch.norm(eyy_test)
    print('L2_relative_error_eyy:{:.4f}%'.format(L2_relative_error_eyy.item()*100))
    # np.save('./result_data/l2_relative_error_eyy({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_eyy(Sigmoid):{:.4f}%'.format(L2_relative_error_eyy.item()*100))

    L2_relative_error_exy=torch.norm(exy_pred-exy_test)/torch.norm(exy_test)
    print('L2_relative_error_exy:{:.4f}%'.format(L2_relative_error_exy.item()*100))
    # np.save('./result_data/l2_relative_error_exy({})_(Sigmoid).npy'.format(epochs),
    #         'L2_relative_error_exy(Sigmoid):{:.4f}%'.format(L2_relative_error_exy.item()*100))

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
    #
