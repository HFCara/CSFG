import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, norm='instance', gpu_ids=0, use_dropout=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'generator_style':
        netG = Generator_add_all(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_dropout=use_dropout)
    elif netG == 'generator':
        netG = Generator_add_all_z(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_dropout=use_dropout)
    elif netG == 'generator_style_2':
        netG = Generator_add_all_2(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer,
                                   use_dropout=use_dropout)

    else:
        raise('generator not implemented!')
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    netG.cuda(gpu_ids)
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D=3, norm='instance', use_sigmoid=True, num_D=3, getIntermFeat=False, gpu_ids=0):
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    netD.cuda(gpu_ids)
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
##############################################################################
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)
def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=nn.ReLU, vaeLike=False, linear_input=512):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        self.linear_input = linear_input
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        self.fc = nn.Sequential(*[nn.Linear(self.linear_input, output_nc)])

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1) #[3, 2048]
        output = self.fc(conv_flat)
        return output

class E_ResNet_both(nn.Module):
    def __init__(self, input_nc=3, output_nc=4, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=nn.ReLU, vaeLike=False, linear_input=512):
        super(E_ResNet_both, self).__init__()
        self.vaeLike = vaeLike
        self.linear_input = linear_input
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        self.fc1 = nn.Sequential(*[nn.Linear(self.linear_input, output_nc)])
        self.fc2 = nn.Sequential(*[nn.Linear(self.linear_input, 2)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1) #[3, 2048]
        output1 = self.fc1(conv_flat)
        output2 = self.fc2(conv_flat)
        return output1, output2

# Generator
##############################################################################
def onehot(num, n=2):
    array = torch.zeros([len(num), n])
    for i, c in enumerate(num):
        array[i][c] = 1
    # b = torch.from_numpy(array)
    return array.cuda()

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_dropout=False):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        activation = nn.ReLU(True)

        encoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        resnet = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)]

        ### upsample
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.encoder = nn.Sequential(*encoder)
        self.resnet = nn.Sequential(*resnet)
        self.decoder = nn.Sequential(*decoder)
        # self.embedding = nn.Sequential(nn.Linear(4, 8))


    def forward(self, x, z=None, ll=None):
        # if z is not None:
        #     # print(".........concat z.........")
        #     z = z.view(z.size(0), z.size(1), 1, 1).expand(
        #         z.size(0), z.size(1), x.size(2), x.size(3))
        #     x = torch.cat([x, z], 1)
        if ll is not None:
            l = onehot(ll)
            # l = self.embedding(l)
            l = l.view(l.size(0), l.size(1), 1, 1).expand(
                l.size(0), l.size(1), x.size(2), x.size(3))
            x = torch.cat([x, l], 1)
        x = self.encoder(x)

        # x = x * (self.ca1(x))
        x = self.resnet(x)
        x = self.decoder(x)
        #print("decode后:", x.shape)#[5, 3, 256, 512]
        return x
    def getEncoder(self):
        return self.encoder

class Generator_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_dropout=False):
        assert (n_blocks >= 0)
        super(Generator_add_all, self).__init__()
        activation = nn.ReLU(True)

        encoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # self.en_s = nn.Sequential(nn.Linear(4, 8), activation)
        ### downsample
        # for i in range(n_downsampling):
        #     mult = 2 ** i
        #     encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
        #               norm_layer(ngf * mult * 2), activation]
        n_c = 2
        self.en1= nn.Sequential(
            nn.Conv2d(ngf+n_c, ngf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 2), activation
        )
        self.en2= nn.Sequential(
            nn.Conv2d(ngf*2+n_c, ngf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 4), activation
        )
        self.en3= nn.Sequential(
            nn.Conv2d(ngf*4+n_c, ngf * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 8), activation
        )
        self.en4= nn.Sequential(
            nn.Conv2d(ngf*8+n_c, ngf * 16, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 16), activation
        )

        ### resnet blocks
        # resnet = []
        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):
        #     resnet += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)]
        #
        self.res1 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)
        self.res2 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)
        self.res3 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)
        self.res4 = ResnetBlock(ngf * 16 , padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)
        ### upsample
        ### upsample
        # decoder = []
        # for i in range(n_downsampling):
        #     mult = 2 ** (n_downsampling - i)
        #     decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
        #                                  output_padding=1),
        #               norm_layer(int(ngf * mult / 2)), activation]

        self.de1= nn.Sequential(
            nn.ConvTranspose2d(ngf*16+n_c, ngf * 8, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 8), activation
        )
        self.de2= nn.Sequential(
            nn.ConvTranspose2d(ngf*8+n_c, ngf * 4, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 4), activation
        )
        self.de3= nn.Sequential(
            nn.ConvTranspose2d(ngf*4+n_c, ngf * 2, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 2), activation
        )
        self.de4= nn.Sequential(
            nn.ConvTranspose2d(ngf*2+n_c, ngf * 1, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 1), activation
        )
        decoder = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.encoder = nn.Sequential(*encoder)
        # self.resnet = nn.Sequential(*resnet)
        self.decoder = nn.Sequential(*decoder)
        # self.embedding = nn.Sequential(nn.Linear(4, 8))


    def forward(self, x, z=None, l=None):
        l = onehot(l)
        # l = self.en_s(l)
        if z is not None:
            # print(".........concat z.........")
            z = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x = torch.cat([x, z], 1)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.encoder(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en1(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en2(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en3(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en4(x)
        # x = self.resnet(x)
        x = self.res1(x)

        x = self.res2(x)

        x = self.res3(x)

        x = self.res4(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de1(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de2(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de3(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de4(x)

        x = self.decoder(x)
        #print("decode后:", x.shape)#[5, 3, 256, 512]
        return x
    def getEncoder(self):
        return self.encoder

class Generator_add_all_2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_dropout=False):
        assert (n_blocks >= 0)
        super(Generator_add_all_2, self).__init__()
        activation = nn.ReLU(True)

        encoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        k = 6
        self.en1= nn.Sequential(
            nn.Conv2d(ngf+k, ngf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 2), activation
        )
        self.en2= nn.Sequential(
            nn.Conv2d(ngf*2+k, ngf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 4), activation
        )
        self.en3= nn.Sequential(
            nn.Conv2d(ngf*4+k, ngf * 8, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 8), activation
        )
        self.en4= nn.Sequential(
            nn.Conv2d(ngf*8+k, ngf * 16, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 16), activation
        )

        self.res1 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=use_dropout)
        self.res2 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)
        self.res3 = ResnetBlock(ngf * 16, padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)
        self.res4 = ResnetBlock(ngf * 16 , padding_type=padding_type, activation=activation, norm_layer=norm_layer,
                                use_dropout=use_dropout)

        self.de1= nn.Sequential(
            nn.ConvTranspose2d(ngf*16+k, ngf * 8, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 8), activation
        )
        self.de2= nn.Sequential(
            nn.ConvTranspose2d(ngf*8+k, ngf * 4, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 4), activation
        )
        self.de3= nn.Sequential(
            nn.ConvTranspose2d(ngf*4+k, ngf * 2, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 2), activation
        )
        self.de4= nn.Sequential(
            nn.ConvTranspose2d(ngf*2+k, ngf * 1, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
            norm_layer(ngf * 1), activation
        )
        decoder = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x, z=None, dic=None, river=None):
        d = onehot(dic, 4)
        r = onehot(river, 2)
        l = torch.cat([d, r], -1)
        # l = self.en_s(l)
        if z is not None:
            # print(".........concat z.........")
            z = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x = torch.cat([x, z], 1)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.encoder(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en1(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en2(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en3(x)
        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.en4(x)
        # x = self.resnet(x)
        x = self.res1(x)

        x = self.res2(x)

        x = self.res3(x)

        x = self.res4(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de1(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de2(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de3(x)

        ll = l.view(l.size(0), l.size(1), 1, 1).expand(
            l.size(0), l.size(1), x.size(2), x.size(3))
        x = torch.cat([x, ll], 1)
        x = self.de4(x)

        x = self.decoder(x)
        return x
    def getEncoder(self):
        return self.encoder


# Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        print(use_dropout)
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator_noc(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=False, linear_input=665):
        super(NLayerDiscriminator_noc, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        # out = [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)


    def forward(self, input, l=None, c=None):
        if l is not None:
            input = torch.cat([input, l], 1)
        out_d = self.model(input)
        # out_d = self.out(feat)
        # out = self.out_C(feat)
        # c_out = self.out_l(out.view(out.size(0), -1))
        return out_d
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=True, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_noc(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            setattr(self, 'layer' + str(i), netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, x, l=None):
        #print(x.shape, l.shape)

        if l is not None:
            input = torch.cat([x, l], 1)
        else:
            input = x

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            # print(self.singleD_forward(model, input_downsampled)[-1].shape)
            out_d = self.singleD_forward(model, input_downsampled)[-1]
            result.append(out_d)
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        


from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
