from sklearn.linear_model import OrthogonalMatchingPursuitCV
import torch
import itertools
import torch 
import torch.nn as nn 

netG_A = # 定义生成器A，用于A转B的风格
netG_B = # 定义生成器B，用于B转A的风格
netD_A = # 定义判别器A，用于判别A的风格
netD_B = # 定义判别器B，用于判定B的风格
gan_Loss = nn.MSELoss() # 对抗生成网络可以选择MSE作为损失
#gan_Loss = nn.BCEWithLogitsLoss() # 也可以选择交叉熵作为损失
l1_loss = nn.L1Loss() # L1损失
optim_G = torch.optim.Adam(# 加入两个生成模型的可训练参数
    itertools.chain(netG_A.parameters(), netG_B.parameters()), 
    lr=1e-4)
optim_D = torch.optim.Adam(# 加入两个判别模型的可训练参数
    itertools.chain(netD_A.parameters(), netD_B.parameters()), 
    lr=1e-4)
# 迭代过程
for step in ... 
    # 训练判别器
    optim_D.zero_grad()
    # 生成器生成对应图像
    y_hat = netG_A(x_real)  # G_A(x)
    x_hat = netG_B(y_real)  # G_B(B)
    # 循环将生成图像输入回各自编码器
    x_rec = netG_B(y_hat)   # G_B(G_A(A))
    y_rec = netG_A(x_hat)   # G_A(G_B(B))
    # 判别器判定风格是否一致
    y_hat_fake = netD_B(y_hat.detach()) 
    y_hat_real = netD_B(y_real)
    x_hat_fake = netD_A(x_hat.detach()) 
    x_hat_real = netD_A(x_real)
    loss_D_A = gan_Loss(y_hat_real, torch.ones_like(y_hat_real)) + \
        gan_Loss(y_hat_fake, torch.zeros_like(y_hat_fake)) 
    loss_D_B = gan_Loss(x_hat_real, torch.ones_like(x_hat_real)) + \
        gan_Loss(x_hat_fake, torch.zeros_like(x_hat_fake)) 

    loss_D = loss_D_A + loss_D_B 
    loss_D.backward() 
    optim_D.step() 
    optim_D.zero_grad()

    # 训练生成器
    # 对抗生成损失
    optim_G.zero_grad() 
    x_hat_fake = netD_A(x_hat)
    loss_G_A = gan_Loss(x_hat_fake, torch.ones_like(x_hat_fake))
    y_hat_fake = netD_B(y_hat)
    loss_G_B = gan_Loss(y_hat_fake, torch.ones_like(y_hat_fake))
    # L1损失用于约束图像
    loss_L1_A = l1_loss(x_rec, x_real) 
    loss_L1_B = l1_loss(y_rec, y_real)

    loss_G = loss_G_A + loss_G_B + loss_L1_A + loss_L1_B
    loss_G.backward() 
    optim_G.step() 
    optim_G.zero_grad()
