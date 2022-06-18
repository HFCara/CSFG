import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import *
import argparse
import os
from tqdm import tqdm
import functools
from torchvision import utils as vutils
from networks import *
import torch.backends.cudnn as cudnn
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=22, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=8)  # 同上
parser.add_argument('--n_blocks', type=int, default=0)  # 6
parser.add_argument('--num_epochs', type=int, default=8, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=2, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--output_save_path', type=str, default="./output_2/", help='')  #
parser.add_argument('--model_dir', type=str, default="./model_2/", help='beta2 for Adam optimizer')
parser.add_argument('--netD', type=str, default='patchGAN',
                    help='selects model to use for netD')  # 'patchGAN','oriD','editD'
parser.add_argument('--use_dropout', action='store_true', default=False)
parser.add_argument('--netG', type=str, default='generator_style_2', help='selects model to use for netD')
params = parser.parse_args()
print(params)

##############################################

data_dir = '../img/'
model_dir = params.model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)



# Data pre-processing
transform = transforms.Compose([transforms.Resize((128, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                ])

# Train data
train_data = TrainDataset(data_dir, real_folder='reals_0511', label_folder='labels_0512', transform=transform)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True,
                                                drop_last=True)

# Test data
test_data = TestDataset(data_dir, test_folder="test", transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=False)

####################################### 定义网络 ###########################################################
norm_layer = get_norm_layer(norm_type="instance")
netG = define_G(3 + params.nz + 4+2, 3, int(params.ngf), str(params.netG), 4, int(params.n_blocks), use_dropout=False)


# D = define_D(6, int(params.ndf))
D = MultiscaleDiscriminator(6).cuda()
E = E_ResNet_both(norm_layer=norm_layer).cuda()
init_weights(E)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

w = [1, 3, 1, 1]
class_weights = torch.FloatTensor(w).cuda()
NLL_loss = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
NLL_loss2 = torch.nn.CrossEntropyLoss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(netG.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))
E_optimizer = torch.optim.Adam(E.parameters(), lr=params.lrE, betas=(params.beta1, params.beta2))

# Training GAN
step = 0
cudnn.benchmark = True

for epoch in tqdm(range(params.num_epochs)):

    # training
    accuracy_dic = 0.
    accuracy_type = 0.
    for i, (img, label_input, dic_class, river_class) in enumerate(train_data_loader):
        label_img = Variable(label_input.cuda())
        real_img = Variable(img.cuda())
        dic_class = Variable(dic_class.cuda())
        river_class = Variable(river_class.cuda())

        z = get_z_random(img.size(0), params.nz).cuda()
        gen_image = netG(label_img, z, dic=dic_class, river=river_class)

        # Train discriminator with real data
        # D_real_decisions = D(label_img, real_img)
        # D_real_loss = compute_D_loss(D_real_decisions, BCE_loss, True)
        #
        #
        # D_fake_decisions = D(label_img, gen_image.detach())
        # D_fake_loss = compute_D_loss(D_real_decisions, BCE_loss, False)
        (D_real_decision1, D_real_decision2, D_real_decision3) = D(label_img, real_img)

        real_1 = Variable(torch.full(D_real_decision1[0].size(), 0.9).cuda())
        real_2 = Variable(torch.full(D_real_decision2[0].size(), 0.9).cuda())
        real_3 = Variable(torch.full(D_real_decision3[0].size(), 0.9).cuda())
        # real_4 = Variable(torch.full(D_real_decision4[0].size(), 0.9).cuda())
        D_real_loss = BCE_loss(D_real_decision1[0], real_1) + BCE_loss(D_real_decision2[0], real_2) + \
                      BCE_loss(D_real_decision3[0], real_3)

        (D_fake_decision1, D_fake_decision2, D_fake_decision3) = D(label_img, gen_image.detach())
        fake_1 = Variable(torch.zeros(D_fake_decision1[0].size()).cuda())
        fake_2 = Variable(torch.zeros(D_fake_decision2[0].size()).cuda())
        fake_3 = Variable(torch.zeros(D_fake_decision3[0].size()).cuda())
        # fake_4 = Variable(torch.zeros(D_fake_decision4[0].size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision1[0], fake_1) + BCE_loss(D_fake_decision2[0], fake_2) + BCE_loss(
            D_fake_decision3[0], fake_3)

        dic, river = E(real_img)
        class_real_loss = NLL_loss(dic, dic_class) + NLL_loss2(river, river_class)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss)
        D.zero_grad()
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        E.zero_grad()
        class_real_loss.backward()
        E_optimizer.step()

        # D_fake_decisions = D(label_img, gen_image)
        # G_fake_loss = compute_D_loss(D_fake_decisions, BCE_loss, True)
        (D_fake_decision1, D_fake_decision2, D_fake_decision3) = D(label_img, gen_image)
        G_fake_loss = BCE_loss(D_fake_decision1[0], real_1) + BCE_loss(D_fake_decision2[0], real_2) + BCE_loss(
            D_fake_decision3[0], real_3)

        dic_fake, river_fake = E(gen_image)
        class_fake_loss = NLL_loss(dic_fake, dic_class) + NLL_loss2(river_fake, river_class)

        l1_loss = params.lamb * L1_loss(gen_image, real_img)

        # Back propagation
        G_loss = G_fake_loss + l1_loss + class_fake_loss * 5
        netG.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, l1_loss: %.4f, class_real_loss: %.4f'
                    % (epoch + 1, params.num_epochs, i + 1, len(train_data_loader), D_loss.item(), l1_loss.item(),
                       class_fake_loss.item()))

        step += 1
    '''test'''
    # Show result for test image
    set_requires_grad([netG, D], False)
    with torch.no_grad():
        if (epoch + 1) % 1 == 0:
            save_path = str(params.output_save_path) + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dic_num = 4
            river_num = 2
            for i, (img, img_name) in enumerate(test_data_loader):
                for di in range(dic_num):
                    for ri in range(2):
                        z = get_z_random(img.size(0), params.nz).cuda()
                        di_ = torch.tensor([di])
                        ri_ = torch.tensor([ri])
                        # print(c_.shape)
                        gen_image = netG(img.cuda(), z, di_, ri_)
                        gen_image = gen_image.cpu().data
                        file_name = save_path + str(img_name) + "_" + str(epoch) + "_" + str(di) + "_" + str(ri) + ".png"
                        vutils.save_image(gen_image, file_name)  # 已转为正常ＲＧＢ图像
    if (epoch + 1) == params.num_epochs:
        torch.save(netG.state_dict(), model_dir + 'generator_param_' + str(epoch) + '.pkl')
    set_requires_grad([netG, D], True)

