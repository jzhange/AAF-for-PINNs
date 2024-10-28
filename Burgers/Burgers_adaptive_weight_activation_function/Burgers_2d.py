from net import Net
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from math import pi
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def PDE(u, t, x, nu):
    return d(u, t) + u * d(u, x) - nu * d(d(u, x), x)

def train():

    nu = 0.01 / pi
    lr = 0.001
    epochs = 8000
    t_left, t_right = 0., 1.
    x_left, x_right = -1., 1.
    n_f, n_b_1, n_b_2 = 10000, 500, 500


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


    plt.cla()
    mse_test = u_star
    plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
                   mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('./result_plot/Burger1d_exact.png', bbox_inches='tight', format='png')
    plt.close()



    PINN = Net(seq_net=[2, 20, 20, 20, 20, 20, 20, 1]).to(device)
    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    criterion = torch.nn.MSELoss()



    mse_loss = []


    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        t_f = ((t_left + t_right) / 2 + (t_right - t_left) *
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)

        x_f = ((x_left + x_right) / 2 + (x_right - x_left) *
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)

        u_f = PINN(torch.cat([t_f, x_f], dim=1))
        PDE_ = PDE(u_f, t_f, x_f, nu)
        mse_PDE = criterion(PDE_, torch.zeros_like(PDE_))


        x_rand = ((x_left + x_right) / 2 + (x_right - x_left) *
                  (torch.rand(size=(n_b_1, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
        t_b = (t_left * torch.ones_like(x_rand)
               ).requires_grad_(True)
        u_b_1 = PINN(torch.cat([t_b, x_rand], dim=1)) + torch.sin(pi * x_rand)
        mse_BC_1 = criterion(u_b_1, torch.zeros_like(u_b_1))



        t_rand = ((t_left + t_right) / 2 + (t_right - t_left) *
                  (torch.rand(size=(n_b_2, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
        x_b_1 = (x_left * torch.ones_like(t_rand)
                 ).requires_grad_(True)
        x_b_2 = (x_right * torch.ones_like(t_rand)
                 ).requires_grad_(True)

        u_b_2 = PINN(torch.cat([t_rand, x_b_1], dim=1))
        u_b_3 = PINN(torch.cat([t_rand, x_b_2], dim=1))
        mse_BC_2 = criterion(u_b_2, torch.zeros_like(u_b_2))
        mse_BC_3 = criterion(u_b_3, torch.zeros_like(u_b_3))

        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3


        loss = 1 * mse_PDE + 1 * mse_BC



        mse_loss.append(loss.item())



        if (epoch + 1) % 100 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e}'.format(
                    (epoch+1), mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )


        loss.backward()
        optimizer.step()


    t1 = time.time()
    print("total time:{:.02f}s".format(t1 - t0))

    #
    # x_pred = X_star[:, 0:1]
    # t_pred = X_star[:, 1:2]
    # u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))
    #
    # plt.cla()
    # plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
    #                u_pred.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.mappable.set_clim(-1, 1)
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.savefig('./result_plot/Burger1d_pred_{}(fixed).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()
    #
    # plt.cla()
    # mse_test = abs(u_pred - u_star)
    # plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
    #                mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.mappable.set_clim(0, 0.3)
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.savefig('./result_plot/Burger1d_error_{}(fixed).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()
    #
    #
    # plt.cla()
    # fig,ax=plt.subplots(1,3,figsize=(12,4))
    # x_25=torch.tensor(x,dtype=torch.float32,device=device,requires_grad=True)
    # t_25=(0.25*torch.ones_like(x_25)).requires_grad_(True)
    # u_25=PINN(torch.cat([t_25,x_25],dim=1))
    # ax[0].plot(x,Exact[25,:],'b-',linewidth=2,label='Exact')
    # ax[0].plot(x,u_25.reshape(-1,1).detach().cpu().numpy(),'r--',
    #                linewidth=2,label='PINN')
    # ax[0].set_xlabel('x')
    # ax[0].set_ylabel('u(t,x)')
    # ax[0].set_xlim([-1.1,1.1])
    # ax[0].set_ylim([-1.1,1.1])
    # ax[0].axis('square')
    # ax[0].set_title('t=0.25',fontsize=10)
    #
    # x_50 = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    # t_50 = (0.50 * torch.ones_like(x_25)).requires_grad_(True)
    # u_50 = PINN(torch.cat([t_50, x_50], dim=1))
    # ax[1].plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    # ax[1].plot(x, u_50.reshape(-1, 1).detach().cpu().numpy(),'r--',
    #                linewidth=2, label='PINN')
    # ax[1].set_xlabel('x')
    # ax[1].set_ylabel('u(t,x)')
    # ax[1].set_xlim([-1.1, 1.1])
    # ax[1].set_ylim([-1.1, 1.1])
    # ax[1].axis('square')
    # ax[1].set_title('t=0.50', fontsize=10)
    # ax[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.1),ncol=5,frameon=False)
    #
    # x_75 = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    # t_75 = (0.75 * torch.ones_like(x_75)).requires_grad_(True)
    # u_75 = PINN(torch.cat([t_75, x_75], dim=1))
    # ax[2].plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    # ax[2].plot(x, u_75.reshape(-1, 1).detach().cpu().numpy(),'r--',
    #                linewidth=2, label='PINN')
    # ax[2].set_xlabel('x')
    # ax[2].set_ylabel('u(t,x)')
    # ax[2].set_xlim([-1.1, 1.1])
    # ax[2].set_ylim([-1.1, 1.1])
    # ax[2].axis('square')
    # ax[2].set_title('t=0.75', fontsize=10)
    # plt.savefig('./result_plot/Burgers1d_t=0.25-0.50-0.75(fixed).png',bbox_inches='tight',format='png')
    # plt.close()
    #
    #
    # plt.cla()
    # plt.plot(mse_loss)
    # plt.yscale('log')
    # plt.ylim(1e-5, 1)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('./result_plot/Burger1d_loss_{}(fixed).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()
    #


    np.save('./result_data/training_loss({})_(fixed).npy'.format(epochs), mse_loss)
    np.save('./result_data/train-time({})_(fixed).npy'.format(epochs), (t1 - t0))
    torch.save(PINN.state_dict(), './result_data/PINN({})_(fixed).pth'.format(epochs))


if __name__ == '__main__':
    train()

