from net import Net
from net import Net_adaptive
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import grad
from itertools import chain
from math import pi
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed = 42
# set_seed(seed)
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


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断gpu可用是否可用


    n_f = 4000
    n_b_bc = 400
    Lambda, mu, q = 1, 0.5, 4
    x_left=0.
    x_right=1.
    y_left=0.
    y_right=1.
    epochs = 8000
    lr=0.001


    a = torch.tensor([1/5], dtype=torch.float32, device=device, requires_grad=True)
    a.grad = torch.ones((1)).to(device)

    b = torch.tensor([1/5], dtype=torch.float32, device=device, requires_grad=True)
    b.grad = torch.ones((1)).to(device)

    c = torch.tensor([1/5], dtype=torch.float32, device=device, requires_grad=True)
    c.grad = torch.ones((1)).to(device)

    d = torch.tensor([1/5], dtype=torch.float32, device=device, requires_grad=True)
    d.grad = torch.ones((1)).to(device)

    e = torch.tensor([1/5], dtype=torch.float32, device=device, requires_grad=True)
    e.grad = torch.ones((1)).to(device)



    PINN = Net_adaptive(seq_net=[2, 50, 50, 50, 50, 5],a=a,b=b,c=c,d=d,e=e).to(device)  # to(device)将Net放到GPU上，若电脑无GPU可将全文.to(device)或者device删掉


    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    criterion=torch.nn.MSELoss()
    optimizer_weights = torch.optim.Adam([{'params': a, 'lr': 0.001},
                                          {'params': b, 'lr': 0.001},
                                          {'params': c, 'lr': 0.001},
                                          {'params': d, 'lr': 0.001},
                                          {'params': e, 'lr': 0.001}])

    loss_history = []
    a_history = []
    b_history = []
    c_history = []
    d_history = []
    e_history = []

    t0 = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        optimizer_weights.zero_grad()

        x_internal = ((x_left+x_right)/2+ (x_right-x_left) *
                      (torch.rand(size=(n_f, 1), dtype=torch.float32, device=device) - 0.5)
                      ).requires_grad_(True)
        y_internal = ((y_left+y_right)/2 + (y_right-y_left) *
                      (torch.rand(size=(n_f, 1), dtype=torch.float32, device=device) - 0.5)
                      ).requires_grad_(True)
        solution_internal\
            =PINN(torch.cat([x_internal, y_internal], dim=1))
        PDE_1,PDE_2,PDE_3,PDE_4,PDE_5,_,_,_=(
            PDEs(x_internal,y_internal,Lambda, mu, q,
                 solution_internal[:,0:1],solution_internal[:,1:2],solution_internal[:,2:3],solution_internal[:,3:4],solution_internal[:,4:5]))


        mse_PDE = (criterion(PDE_1, torch.zeros_like(PDE_1)) +
                   criterion(PDE_2, torch.zeros_like(PDE_2)) +
                   criterion(PDE_3, torch.zeros_like(PDE_3)) +
                   criterion(PDE_4, torch.zeros_like(PDE_4)) +
                   criterion(PDE_5, torch.zeros_like(PDE_5)))


        x_bc_down = ((x_left+x_right)/2 + (x_right-x_left) *
                      (torch.rand(size=(n_b_bc, 1), dtype=torch.float, device=device) - 0.5)
                      ).requires_grad_(True)
        y_bc_down = (torch.zeros_like(x_bc_down)).requires_grad_(True)
        solution_down= PINN(torch.cat([x_bc_down, y_bc_down], dim=1))
        mse_bc_d_1 = criterion(solution_down[:,0:1], torch.zeros_like(solution_down[:,0:1]))
        mse_bc_d_2 = criterion(solution_down[:,1:2], torch.zeros_like(solution_down[:,1:2]))


        y_bc_l = ((y_left+y_right) / 2 + (y_right-y_left) *
                      (torch.rand(size=(n_b_bc, 1), dtype=torch.float, device=device) - 0.5)
                      ).requires_grad_(True)
        y_bc_r = y_bc_l
        x_bc_l = torch.zeros_like(y_bc_l).requires_grad_(True)
        x_bc_r = torch.ones_like(y_bc_r).requires_grad_(True)
        solution_l = PINN(torch.cat([x_bc_l, y_bc_l], dim=1))
        solution_r = PINN(torch.cat([x_bc_r, y_bc_r], dim=1))

        mse_bc_l_1 = criterion(solution_l[:,1:2], torch.zeros_like(solution_l[:,1:2]))
        mse_bc_l_2 = criterion(solution_l[:,2:3], torch.zeros_like(solution_l[:,2:3]))

        mse_bc_r_1 = criterion(solution_r[:,1:2], torch.zeros_like(solution_r[:,1:2]))
        mse_bc_r_2 = criterion(solution_r[:,2:3], torch.zeros_like(solution_r[:,2:3]))


        x_bc_up = ((x_left+x_right) / 2 + (x_right-x_left) *
                      (torch.rand(size=(n_b_bc, 1), dtype=torch.float, device=device) - 0.5)
                      ).requires_grad_(True)
        y_bc_up = torch.ones_like(x_bc_up).requires_grad_(True)
        solution_up = PINN(torch.cat([x_bc_up, y_bc_up], dim=1))
        mse_bc_up_1 = criterion(solution_up[:,0:1], torch.zeros_like(solution_up[:,0:1]))
        mse_bc_up_2 = criterion(solution_up[:,3:4]-(Lambda + 2 * mu) * q * torch.sin(
            pi * x_bc_up), torch.zeros_like(solution_up[:,3:4]))


        mse_bc = mse_bc_d_1 + mse_bc_d_2 + mse_bc_l_1 + mse_bc_r_1 + mse_bc_l_2+ mse_bc_r_2 + mse_bc_up_1 + mse_bc_up_2


        loss = 1 * mse_PDE + 1 * mse_bc

        if (epoch+1) % 100 == 0:
            print(
                'epoch:{:05d}，  loss: {:.08e},  a: {:.08e},b: {:.08e},c: {:.08e},d: {:.08e},e: {:.08e}'.format(
                    (epoch+1), loss.item(),a.item(),b.item(),c.item(),d.item(),e.item()
                )
            )

        loss_history.append(loss.item())
        a_history.append(a.item())
        b_history.append(b.item())
        c_history.append(c.item())
        d_history.append(d.item())
        e_history.append(e.item())

        loss.backward()
        optimizer.step()
        optimizer_weights.step()


    t1 = time.time()
    print("total time:{:.02f}s".format(t1 - t0))


    np.save('./result_data/training_loss({})_(Sigmoid).npy'.format(epochs), loss_history)
    np.save('./result_data/sin-weight_{}(Sigmoid).npy'.format(epochs), a_history)
    np.save('./result_data/tanh-weight_{}(Sigmoid).npy'.format(epochs), b_history)
    np.save('./result_data/GELU-weight_{}(Sigmoid).npy'.format(epochs), c_history)
    np.save('./result_data/Swish-weight_{}(Sigmoid).npy'.format(epochs), d_history)
    np.save('./result_data/Softplus-weight_{}(Sigmoid).npy'.format(epochs), e_history)
    np.save('./result_data/train-time({})_(Sigmoid).npy'.format(epochs), (t1 - t0))
    torch.save(PINN.state_dict(), './result_data/PINN({})_(Sigmoid).pth'.format(epochs))

if __name__ == '__main__':
    train()
