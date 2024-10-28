import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def Plot_loss():
    loss_adaptive=np.load('./result_data/training_loss(20000)_(adap_weight).npy')
    loss_softmax = np.load('./result_data/training_loss(20000)_(Softmax).npy')
    loss_L1norm = np.load('./result_data/training_loss(20000)_(L1norm).npy')
    loss_L2norm = np.load('./result_data/training_loss(20000)_(L2norm).npy')
    loss_Identity = np.load('./result_data/training_loss(20000)_(Identity).npy')
    loss_Sigmoid = np.load('./result_data/training_loss(20000)_(Sigmoid).npy')
    loss_fixed=np.load('./result_data/training_loss(20000)_(fixed).npy')
    plt.cla()
    plt.plot(loss_fixed,label='PINN with Fixed',
             color='darkgrey',linestyle='-',lw=1.2,zorder=1)
    plt.plot(loss_Sigmoid, label='PINN with Sigmoid',
             color='gold', linestyle='-', lw=1.2,zorder=2)
    plt.plot(loss_L1norm, label='PINN with L1norm',
             color='lightcoral', linestyle='-', lw=1.2,zorder=2)
    plt.plot(loss_Identity, label='PINN with Identity',
             color='c', linestyle='-', lw=1.2,zorder=3)
    plt.plot(loss_softmax, label='PINN with Softmax',
             color='royalblue', linestyle='-', lw=1.2,zorder=2)
    plt.plot(loss_adaptive,label='PINN with Weight Average',
             color='forestgreen',linestyle='-',lw=1.2,zorder=4)
    plt.plot(loss_L2norm, label='PINN with L2norm',
             color='darkorchid', linestyle='-', lw=1.2,zorder=4.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='best',prop = {'size':8})
    plt.savefig('./result_plot/loss_comparison_{}(all).png'.format(20000),bbox_inches='tight',dpi=600,format='png',
                pad_inches=0.1)
    plt.close()


def get_weights():
    sin_weight= np.load('result_data/sin-weight_8000.npy')
    tanh_weight=np.load('result_data/tanh-weight_8000.npy')
    GELU_weight=np.load('result_data/GELU-weight_8000.npy')
    Swish_weight=np.load('result_data/Swish-weight_8000.npy')
    Softplus_weight=np.load('result_data/Softplus-weight_8000.npy')

    print('final sin_weight:{:.02f}'.format(sin_weight[-1]))
    print('final tanh_weight:{:.02f}'.format(tanh_weight[-1]))
    print('final GELU_weight:{:.02f}'.format(GELU_weight[-1]))
    print('final Swish_weight:{:.02f}'.format(Swish_weight[-1]))
    print('final Softplus_weight:{:.02f}'.format(Softplus_weight[-1]))


def time_comparison():
    t_adap=np.load('./result_data/train-time(20000)_(adap_weight).npy')
    t_L2norm = np.load('./result_data/train-time(20000)_(L2norm).npy')
    t_Softmax = np.load('./result_data/train-time(20000)_(Softmax).npy')
    t_Identity = np.load('./result_data/train-time(20000)_(Identity).npy')
    t_L1norm = np.load('./result_data/train-time(20000)_(L1norm).npy')
    t_Sigmoid = np.load('./result_data/train-time(20000)_(Sigmoid).npy')
    t_fixed = np.load('./result_data/train-time(20000)_(fixed).npy')
    print('t_adap:{:.2f}s'.format(t_adap))
    print('t_L2norm:{:.2f}s'.format(t_L2norm))
    print('t_Softmax:{:.2f}s'.format(t_Softmax))
    print('t_Identity:{:.2f}s'.format(t_Identity))
    print('t_L1norm:{:.2f}s'.format(t_L1norm))
    print('t_Sigmoid:{:.2f}s'.format(t_Sigmoid))
    print('t_fixed:{:.2f}s'.format(t_fixed))

def Plot_Weights():
    epochs=20000
    a_history = np.load('./result_data/sin-weight_20000(L2norm).npy')
    b_history = np.load('./result_data/tanh-weight_20000(L2norm).npy')
    c_history = np.load('./result_data/GELU-weight_20000(L2norm).npy')
    d_history = np.load('./result_data/Swish-weight_20000(L2norm).npy')
    e_history = np.load('./result_data/Softplus-weight_20000(L2norm).npy')


    plt.cla()
    plt.plot(a_history, label='sin-weight', color='royalblue')
    plt.plot(b_history, label='tanh-weight', color='limegreen')
    plt.plot(c_history, label='GELU-weight', color='gold')
    plt.plot(d_history, label='Swish-weight', color='orchid')
    plt.plot(e_history, label='Softplus-weight', color='salmon')
    plt.yscale('log')
    # plt.ylim(1e-4, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Learnable-Coefficient')
    plt.legend(loc='best')
    plt.savefig('./result_plot/Burgers_activation-weights(L2norm)_{}.png'.format(epochs), bbox_inches='tight',
                format='png')
    plt.close()

def Plot_AF():

    x=torch.arange(-10,10,0.1,requires_grad=True)
    y1=torch.sin(x)
    y2=torch.tanh(x)
    y3=F.gelu(x)
    y4=F.silu(x)
    y5=F.softplus(x)

    plt.cla()
    plt.plot(x.detach(),y1.detach(),label='Sin',color='royalblue',linestyle='-',lw=1)
    plt.plot(x.detach(), y2.detach(), label='Tanh', color='limegreen', linestyle='-', lw=1)
    plt.plot(x.detach(), y3.detach(), label='GELU', color='gold', linestyle='-', lw=1)
    plt.plot(x.detach(), y4.detach(), label='Swish', color='orchid', linestyle='-', lw=1)
    plt.plot(x.detach(), y5.detach(), label='Softplus', color='salmon', linestyle='-', lw=1)
    plt.ylim(-2, 6)
    plt.xlabel('x')
    plt.ylabel('Standard Activation Functions')
    plt.legend(loc='upper left')
    plt.savefig('./result_plot/standard AF.png', bbox_inches='tight',
                format='png')
    plt.close()


    epochs = 20000
    a_history_weighted = torch.tensor(np.load('./result_data/sin-weight_20000(adap_weight).npy'))
    b_history_weighted = torch.tensor(np.load('./result_data/tanh-weight_20000(adap_weight).npy'))
    c_history_weighted = torch.tensor(np.load('./result_data/GELU-weight_20000(adap_weight).npy'))
    d_history_weighted= torch.tensor(np.load('./result_data/Swish-weight_20000(adap_weight).npy'))
    e_history_weighted = torch.tensor(np.load('./result_data/Softplus-weight_20000(adap_weight).npy'))

    a_history_L2norm = torch.tensor(np.load('./result_data/sin-weight_20000(L2norm).npy'))
    b_history_L2norm = torch.tensor(np.load('./result_data/tanh-weight_20000(L2norm).npy'))
    c_history_L2norm = torch.tensor(np.load('./result_data/GELU-weight_20000(L2norm).npy'))
    d_history_L2norm = torch.tensor(np.load('./result_data/Swish-weight_20000(L2norm).npy'))
    e_history_L2norm = torch.tensor(np.load('./result_data/Softplus-weight_20000(L2norm).npy'))

    a_history_Softmax = torch.tensor(np.load('./result_data/sin-weight_20000(Softmax).npy'))
    b_history_Softmax = torch.tensor(np.load('./result_data/tanh-weight_20000(Softmax).npy'))
    c_history_Softmax = torch.tensor(np.load('./result_data/GELU-weight_20000(Softmax).npy'))
    d_history_Softmax = torch.tensor(np.load('./result_data/Swish-weight_20000(Softmax).npy'))
    e_history_Softmax = torch.tensor(np.load('./result_data/Softplus-weight_20000(Softmax).npy'))

    a_history_Sigmoid = torch.tensor(np.load('./result_data/sin-weight_20000(Sigmoid).npy'))
    b_history_Sigmoid = torch.tensor(np.load('./result_data/tanh-weight_20000(Sigmoid).npy'))
    c_history_Sigmoid = torch.tensor(np.load('./result_data/GELU-weight_20000(Sigmoid).npy'))
    d_history_Sigmoid = torch.tensor(np.load('./result_data/Swish-weight_20000(Sigmoid).npy'))
    e_history_Sigmoid = torch.tensor(np.load('./result_data/Softplus-weight_20000(Sigmoid).npy'))

    a_history_Identity = torch.tensor(np.load('./result_data/sin-weight_20000(Identity).npy'))
    b_history_Identity = torch.tensor(np.load('./result_data/tanh-weight_20000(Identity).npy'))
    c_history_Identity = torch.tensor(np.load('./result_data/GELU-weight_20000(Identity).npy'))
    d_history_Identity = torch.tensor(np.load('./result_data/Swish-weight_20000(Identity).npy'))
    e_history_Identity = torch.tensor(np.load('./result_data/Softplus-weight_20000(Identity).npy'))

    a_history_L1norm = torch.tensor(np.load('./result_data/sin-weight_20000(L1norm).npy'))
    b_history_L1norm = torch.tensor(np.load('./result_data/tanh-weight_20000(L1norm).npy'))
    c_history_L1norm = torch.tensor(np.load('./result_data/GELU-weight_20000(L1norm).npy'))
    d_history_L1norm = torch.tensor(np.load('./result_data/Swish-weight_20000(L1norm).npy'))
    e_history_L1norm = torch.tensor(np.load('./result_data/Softplus-weight_20000(L1norm).npy'))

    denominator_weighted=(a_history_weighted[-1]+b_history_weighted[-1]+c_history_weighted[-1]+d_history_weighted[-1]+e_history_weighted[-1])
    y_weighted=((a_history_weighted[-1]/denominator_weighted)*torch.sin(x)
               +(b_history_weighted[-1]/denominator_weighted)*F.tanh(x)
               +(c_history_weighted[-1]/denominator_weighted) * F.gelu(x)
               +(d_history_weighted[-1]/denominator_weighted)*F.silu(x)
               +(e_history_weighted[-1]/denominator_weighted)*F.softplus(x))

    weights_combined = torch.tensor([a_history_L2norm[-1], b_history_L2norm[-1], c_history_L2norm[-1], d_history_L2norm[-1], e_history_L2norm[-1]])
    denominator_L2norm = torch.norm(weights_combined)
    y_L2norm = ((a_history_L2norm[-1] / denominator_L2norm) * torch.sin(x)
             + (b_history_L2norm[-1] / denominator_L2norm) * F.tanh(x)
             + (c_history_L2norm[-1] / denominator_L2norm) * F.gelu(x)
             + (d_history_L2norm[-1] / denominator_L2norm) * F.silu(x)
             + (e_history_L2norm[-1] / denominator_L2norm) * F.softplus(x))

    exp_a = torch.exp(a_history_Softmax[-1])
    exp_b = torch.exp(b_history_Softmax[-1])
    exp_c = torch.exp(c_history_Softmax[-1])
    exp_d = torch.exp(d_history_Softmax[-1])
    exp_e = torch.exp(e_history_Softmax[-1])
    exp_total = (exp_a + exp_b + exp_c + exp_d + exp_e)
    y_Softmax = ((exp_a / exp_total) * torch.sin(x)
             + (exp_b / exp_total) * torch.tanh(x)
             + (exp_c / exp_total) * F.gelu(x)
             + (exp_d / exp_total) * F.silu(x)
             + (exp_e / exp_total) * F.softplus(x))

    y_Sigmoid = (torch.sigmoid(a_history_Sigmoid[-1]) * torch.sin(x)
             + torch.sigmoid(b_history_Sigmoid[-1]) * F.tanh(x)
             + torch.sigmoid(c_history_Sigmoid[-1]) * F.gelu(x)
             + torch.sigmoid(d_history_Sigmoid[-1]) * F.silu(x)
             + torch.sigmoid(e_history_Sigmoid[-1]) * F.softplus(x))

    y_Identity = (a_history_Identity[-1] * torch.sin(x)
             + b_history_Identity[-1] * F.tanh(x)
             + c_history_Identity[-1] * F.gelu(x)
             + d_history_Identity[-1] * F.silu(x)
             + e_history_Identity[-1] * F.softplus(x))

    denominator_L1norm = (abs(a_history_L1norm[-1]) + abs(b_history_L1norm[-1]) + abs(c_history_L1norm[-1]) + abs(d_history_L1norm[-1]) + abs(e_history_L1norm[-1]))
    y_L1norm = ((a_history_L1norm[-1] / denominator_L1norm) * torch.sin(x)
             + (b_history_L1norm[-1] / denominator_L1norm) * F.tanh(x)
             + (c_history_L1norm[-1] / denominator_L1norm) * F.gelu(x)
             + (d_history_L1norm[-1] / denominator_L1norm) * F.silu(x)
             + (e_history_L1norm[-1] / denominator_L1norm) * F.softplus(x))

    plt.cla()
    plt.plot(x.detach(), y_weighted.detach(), label='Weighted Average', color='royalblue', linestyle='-', lw=1)
    plt.plot(x.detach(), y_L2norm.detach(), label='L2norm', color='limegreen', linestyle='-', lw=1)
    plt.plot(x.detach(), y_Softmax.detach(), label='Softmax', color='lightcoral', linestyle='-', lw=1)
    plt.plot(x.detach(), y_Sigmoid.detach(), label='Sigmoid', color='orange', linestyle='-', lw=1)
    plt.plot(x.detach(), y_Identity.detach(), label='Identity', color='gold', linestyle='-', lw=1)
    plt.plot(x.detach(), y_L1norm.detach(), label='L1norm', color='mediumorchid', linestyle='-', lw=1)
    plt.ylim(-2, 6)
    plt.xlabel('x')
    plt.ylabel('Adaptive Activation Functions')
    plt.legend(loc='best')
    plt.savefig('./result_plot/Final AF.png', bbox_inches='tight',
                format='png')
    plt.close()

if __name__=='__main__':
    Plot_Weights()
    Plot_AF()
    Plot_loss()




