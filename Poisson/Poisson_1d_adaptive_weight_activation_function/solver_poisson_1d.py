from net import Net_adaptive
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
import numpy as np
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

def d(f,x):
    return grad(f,x,grad_outputs=torch.ones_like(f),create_graph=True,only_inputs=True)[0]

def PDE(u,x):
    return d(d(u,x),x)+0.49*torch.sin(0.7*x)+2.25*torch.cos(1.5*x)

def Ground_truth(x):
    return torch.sin(0.7*x)+torch.cos(1.5*x)-0.1*x

def train():
    x_left,x_right=-10,10
    lr=0.001
    n_f=200
    epochs=8000

    os.environ['CUDA_VISIBLE_DEVICES']=('0')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 自适应激活函数权重的定义
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

    PINN=Net_adaptive([1,50,50,50,1],a=a,b=b,c=c,d=d,e=e).to(device)
    criterion=torch.nn.MSELoss()
    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    optimizer_weights = torch.optim.Adam([{'params': a, 'lr': 0.001},
                                          {'params': b, 'lr': 0.001},
                                          {'params': c, 'lr': 0.001},
                                          {'params': d, 'lr': 0.001},
                                          {'params': e, 'lr': 0.001}])


    loss_history=[]
    a_history=[]
    b_history = []
    c_history = []
    d_history = []
    e_history = []

    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        optimizer_weights.zero_grad()
        x_f=((x_left+x_right)/2+(x_right-x_left)*
             (torch.rand(size=(n_f,1),dtype=torch.float,device=device)-0.5)).requires_grad_(True)
        u_f=PINN(x_f)
        PDE_=PDE(u_f,x_f)
        mse_PDE=criterion(PDE_,torch.zeros_like(PDE_))

        x_b=torch.tensor([[-10.],[10.]]).requires_grad_(True).to(device)
        u_b=PINN(x_b)
        ground_b=Ground_truth(x_b)
        mse_BC=criterion(u_b,ground_b)

        loss=1*mse_PDE+1*mse_BC
        loss_history.append(loss.item())
        a_history.append(a.item())
        b_history.append(b.item())
        c_history.append(c.item())
        d_history.append(d.item())
        e_history.append(e.item())


        if (epoch + 1) % 100 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e},  a: {:.08e},b: {:.08e},c: {:.08e},d: {:.08e},e: {:.08e}'.format(
                    (epoch+1), mse_PDE.item(), mse_BC.item(), loss.item(), a.item(),b.item(),c.item(),d.item(),e.item()
                )
            )
        loss.backward()
        optimizer.step()
        optimizer_weights.step()

    t1 = time.time()
    print("total time:{:.02f}s".format(t1 - t0))



    # xx = torch.linspace(-10, 10, 10000).reshape(-1, 1).to(device)
    # yy = PINN(xx)
    # zz = Ground_truth(xx)

    # xx = xx.reshape(-1).data.detach().cpu().numpy()
    # yy = yy.reshape(-1).data.detach().cpu().numpy()
    # zz = zz.reshape(-1).data.detach().cpu().numpy()
    #
    # plt.cla()
    # plt.plot(xx, yy, label='PINN')
    # plt.plot(xx, zz, label='TRUE', color='r',linestyle='--')
    # plt.ylim(-3, 3)
    # plt.legend()
    # plt.title("Epoch({})".format(epochs))
    # plt.savefig('./result_plot/poisson1d_{}_(Weighted_Average).png'.format(epochs),
    #             bbox_inches='tight', format='png')
    # plt.close()
    #
    #

    # plt.plot(loss_history)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.yscale('log')
    # plt.savefig('./result_plot/poisson1d_loss_{}_(Weighted_Average).png'.format(epochs),bbox_inches='tight',format='png')
    # plt.close()
    #

    # plt.cla()
    # plt.plot(a_history,label='sin-weight',color='b')
    # plt.plot(b_history, label='tanh-weight', color='g')
    # plt.plot(c_history,label='GELU-weight',color='r')
    # plt.plot(d_history, label='Swish-weight', color='y')
    # plt.plot(e_history, label='Softplus-weight', color='m')
    # plt.yscale('log')
    # # plt.ylim(1e-4, 1)
    # plt.xlabel('Epoch')
    # plt.ylabel('activation-weights')
    # plt.legend(loc='best')
    # plt.savefig('./result_plot/Burgers_activation-weights(Weighted_Average)_{}.png'.format(epochs), bbox_inches='tight', format='png')
    # plt.close()


    np.save('./result_data/training_loss2({})_(Softmax).npy'.format(epochs), loss_history)
    np.save('./result_data/sin-weight_{}(Softmax).npy'.format(epochs), a_history)
    np.save('./result_data/tanh-weight_{}(Softmax).npy'.format(epochs),b_history)
    np.save('./result_data/GELU-weight_{}(Softmax).npy'.format(epochs),c_history)
    np.save('./result_data/Swish-weight_{}(Softmax).npy'.format(epochs),d_history)
    np.save('./result_data/Softplus-weight_{}(Softmax).npy'.format(epochs),e_history)
    np.save('./result_data/train-time({})_(Softmax).npy'.format(epochs),(t1-t0))
    torch.save(PINN.state_dict(), './result_data/PINN2({})_(Softmax).pth'.format(epochs))



if __name__=='__main__':
    train()