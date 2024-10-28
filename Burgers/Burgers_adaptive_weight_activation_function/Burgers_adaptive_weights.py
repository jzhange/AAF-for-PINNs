from net import Net_adaptive
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from math import pi
import time
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


def PDE(u, t, x, nu):
    return d(u, t) + u * d(u, x) - nu * d(d(u, x), x)

def train():

    nu = 0.01 / pi
    lr = 0.001
    epochs = 20000
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

    # p1=torch.randn(1,dtype=torch.float32,device=device,requires_grad=True)
    # p1.grad=torch.ones((1)).to(device)
    #
    # p2=torch.randn(1,dtype=torch.float32,device=device,requires_grad=True)
    # p2.grad=torch.ones((1)).to(device)
    #
    # beta=torch.randn(1,dtype=torch.float32,device=device,requires_grad=True)
    # beta.grad=torch.ones((1)).to(device)


    PINN = Net_adaptive(seq_net=[2, 20, 20, 20, 20, 20, 20, 1],a=a,b=b,c=c,d=d,e=e).to(device)
    # PINN = Net_adaptive(seq_net=[2, 20, 20, 20, 20, 20, 20, 1], p1=p1,p2=p2,beta=beta).to(device)
    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    optimizer_weights=torch.optim.Adam([{'params': a, 'lr': 0.001},
                                        {'params': b, 'lr': 0.001},
                                        {'params': c, 'lr': 0.001},
                                        {'params': d, 'lr': 0.001},
                                        {'params': e, 'lr': 0.001}])

    # optimizer_weights = torch.optim.Adam([{'params': p1, 'lr': 0.001},
    #                                       {'params': p2, 'lr': 0.001},
    #                                       {'params': beta, 'lr': 0.001}])

    criterion = torch.nn.MSELoss()

    #optimizer_weights.add_param_group({'params': a, 'lr': 0.001})
    #optimizer_weights.add_param_group({'params': b, 'lr': 0.001})
    #optimizer_weights.add_param_group({'params': c, 'lr': 0.001})


    mse_loss = []
    a_history=[]
    b_history = []
    c_history = []
    d_history = []
    e_history = []

    # p1_history=[]
    # p2_history = []
    # beta_history = []


    t0=time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        optimizer_weights.zero_grad()

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
        a_history.append(a.item())
        b_history.append(b.item())
        c_history.append(c.item())
        d_history.append(d.item())
        e_history.append(e.item())
        # p1_history.append(p1.item())
        # p2_history.append(p1.item())
        # beta_history.append(p1.item())



        if (epoch + 1) % 100 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e},  a: {:.08e},b: {:.08e},c: {:.08e},d: {:.08e},e: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item(), a.item(),b.item(),c.item(),d.item(),e.item()
                )
            )
        # if (epoch + 1) % 100 == 0:
        #     print(
        #         'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e},  p1: {:.08e},p2: {:.08e},beta: {:.08e}'.format(
        #             epoch, mse_PDE.item(), mse_BC.item(), loss.item(), p1.item(),p2.item(),beta.item()
        #         )
        #     )


        loss.backward()
        optimizer.step()
        optimizer_weights.step()

    t1 = time.time()
    print("total time:{:.02f}s".format(t1-t0))



    x_pred = X_star[:, 0:1]
    t_pred = X_star[:, 1:2]
    u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))

    # plt.cla()
    # plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
    #                u_pred.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.mappable.set_clim(-1, 1)
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.savefig('./result_plot/Burger1d_pred_{}(L2norm).png'.format(epochs), bbox_inches='tight', format='png')
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
    # plt.savefig('./result_plot/Burger1d_error2_{}(L2norm).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()

    # plt.cla()
    # mse_test = abs(u_pred - u_star)/max(abs(u_star))
    # plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
    #                mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.mappable.set_clim(0, 0.2)
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.savefig('./result_plot/Burger1d_relative_error2_{}(L2norm).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()


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
    # plt.savefig('./result_plot/Burgers1d_t=0.25-0.50-0.75_{}(L2norm).png'.format(epochs),bbox_inches='tight',format='png')
    # plt.close()


    # plt.cla()
    # plt.plot(mse_loss)
    # plt.yscale('log')
    # plt.ylim(1e-5, 1)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('./result_plot/Burger1d_loss2_{}(L2norm).png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()


    # plt.cla()
    # plt.plot(a_history,label='sin-weight',color='b')
    # plt.plot(b_history, label='tanh-weight', color='g')
    # plt.plot(c_history,label='GELU-weight',color='r')
    # plt.plot(d_history, label='Swish-weight', color='y')
    # plt.plot(e_history, label='Softplus-weight', color='m')
    # # plt.plot(p1_history,label='p1',color='b')
    # # plt.plot(p2_history, label='p2', color='g')
    # # plt.plot(beta_history,label='beta',color='r')
    # plt.yscale('log')
    # # plt.ylim(1e-4, 1)
    # plt.xlabel('Epoch')
    # plt.ylabel('activation-weights')
    # plt.legend(loc='best')
    # plt.savefig('./result_plot/Burgers_activation-weights(L2norm)_{}.png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()



    np.save('./result_data/training_loss2({})_(Sigmoid).npy'.format(epochs), mse_loss)

    np.save('./result_data/sin-weight_{}(Sigmoid).npy'.format(epochs), a_history)
    np.save('./result_data/tanh-weight_{}(Sigmoid).npy'.format(epochs),b_history)
    np.save('./result_data/GELU-weight_{}(Sigmoid).npy'.format(epochs),c_history)
    np.save('./result_data/Swish-weight_{}(Sigmoid).npy'.format(epochs),d_history)
    np.save('./result_data/Softplus-weight_{}(Sigmoid).npy'.format(epochs),e_history)

    np.save('./result_data/train-time({})_(Sigmoid).npy'.format(epochs),(t1-t0))
    torch.save(PINN.state_dict(), './result_data/PINN2({})_(Sigmoid).pth'.format(epochs))


if __name__ == '__main__':
    train()