from net import Net
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
from itertools import chain
from math import pi
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def f_force_x_y(x, y, Lambda, mu, q):
    out_x = Lambda * (4 * (pi ** 2) * torch.cos(2 * pi * x) * torch.sin(pi * y) - pi * torch.cos(pi * x) * q * (y ** 3)) \
            + mu * (9 * (pi ** 2) * torch.cos(2 * pi * x) * torch.sin(pi * y) - pi * torch.cos(pi * x) * q * (y ** 3))
    out_y = Lambda * (
            (-3) * torch.sin(pi * x) * q * (y ** 2) + 2 * (pi ** 2) * torch.sin(2 * pi * x) * torch.cos(pi * y)) \
            + mu * ((-6) * torch.sin(pi * x) * q * (y ** 2) + 2 * (pi ** 2) * torch.sin(2 * pi * x) * torch.cos(
        pi * y) + (pi ** 2) * torch.sin(pi * x) * q * (y ** 4) / 4)
    return out_x, out_y


def fun_exact(x, y, Lambda, mu,q):
    ux = torch.cos(2 * pi * x) * torch.sin(pi * y)
    uy = torch.sin(pi * x) * q * (y ** 4) / 4
    exx = d(ux, x)
    eyy = d(uy, y)
    exy = (d(ux, y) + d(uy, x)) / 2
    sigmaxx=(Lambda + 2 * mu) * exx + Lambda * eyy
    sigmayy=(Lambda + 2 * mu) * eyy + Lambda * exx
    sigmaxy=(2 * mu) * exy
    return ux, uy,sigmaxx,sigmayy,sigmaxy,exx,eyy,exy

def PDEs(x,y,Lambda, mu, q,ux,uy,sigmaxx,sigmayy,sigmaxy):
    fx, fy = f_force_x_y(x, y, Lambda, mu, q)
    exx = d(ux, x)
    eyy = d(uy, y)
    exy = (d(ux, y) + d(uy, x)) / 2
    PDE_1 = d(sigmaxx, x) + d(sigmaxy, y) + fx
    PDE_2 = d(sigmayy, y) + d(sigmaxy, x) + fy
    PDE_3 = (Lambda + 2 * mu) * exx + Lambda * eyy - sigmaxx
    PDE_4 = (Lambda + 2 * mu) * eyy + Lambda * exx - sigmayy
    PDE_5 = (2 * mu) * exy - sigmaxy
    return PDE_1,PDE_2,PDE_3,PDE_4,PDE_5,exx,eyy,exy



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Lambda, mu, q = 1, 0.5, 4
x_left=0.
x_right=1.
y_left=0.
y_right=1.



x_test = torch.linspace(0, 1, 200, device=device, requires_grad=True)
y_test = torch.linspace(0, 1, 200, device=device, requires_grad=True)
xs, ys = torch.meshgrid([x_test, y_test])
s1 = xs.shape
x_test = xs.reshape(-1, 1)
y_test = ys.reshape(-1, 1)
ux_test, uy_test, sigmaxx_test, sigmayy_test, sigmaxy_test, exx_test, eyy_test, exy_test \
    = fun_exact(x_test, y_test, Lambda, mu, q)
ux_test = ux_test.reshape(s1).cpu().T.detach().numpy()
uy_test = uy_test.reshape(s1).cpu().T.detach().numpy()
sigmaxx_test = sigmaxx_test.reshape(s1).cpu().T.detach().numpy()
sigmayy_test = sigmayy_test.reshape(s1).cpu().T.detach().numpy()
sigmaxy_test = sigmaxy_test.reshape(s1).cpu().T.detach().numpy()
exx_test = exx_test.reshape(s1).cpu().T.detach().numpy()
eyy_test = eyy_test.reshape(s1).cpu().T.detach().numpy()
exy_test = exy_test.reshape(s1).cpu().T.detach().numpy()


plt.cla()
plt.imshow(ux_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-0.8, vmax=0.8)
plt.title('Exact Ux')
plt.colorbar()
plt.savefig('./result_plot/ux_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(uy_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=0, vmax=0.8)
plt.title('Exact Uy')
plt.colorbar()
plt.savefig('./result_plot/uy_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(sigmaxx_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-10, vmax=10)
plt.title('Exact Sigma-xx')
plt.colorbar()
plt.savefig('./result_plot/sigmaxx_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(sigmayy_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-6, vmax=8)
plt.title('Exact Sigma-yy')
plt.colorbar()
plt.savefig('./result_plot/sigmayy_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(sigmaxy_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-3, vmax=2)
plt.title('Exact Sigma-xy')
plt.colorbar()
plt.savefig('./result_plot/sigmaxy_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(exx_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-6, vmax=6)
plt.title('Exact E-xx')
plt.colorbar()
plt.savefig('./result_plot/exx_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(eyy_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=0, vmax=4)
plt.title('Exact E-yy')
plt.colorbar()
plt.savefig('./result_plot/eyy_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

plt.cla()
plt.imshow(exy_test, interpolation='nearest', cmap='rainbow',
           extent=[x_left, x_right, y_left, y_right],
           origin='lower', aspect='auto', vmin=-3, vmax=2)
plt.title('Exact E-xy')
plt.colorbar()
plt.savefig('./result_plot/exy_exact.png', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()