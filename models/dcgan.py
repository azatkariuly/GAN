import torch.nn as nn

__all__ = ['dcgan']

class Generator(nn.Module):
    def __init__(self, gen_input=100, gen_feature=64, dis_input=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(gen_input, gen_feature * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_feature * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(gen_feature * 8, gen_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( gen_feature * 4, gen_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( gen_feature * 2, gen_feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_feature),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( gen_feature, dis_input, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, dis_input=3, dis_feature=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(dis_input, dis_feature, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(dis_feature, dis_feature * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dis_feature * 2, dis_feature * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dis_feature * 4, dis_feature * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dis_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dis_feature * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def dcgan(**kwargs):
    dataset = map(kwargs.get, ['dataset'])

    if dataset == 'cifar10':
        return Generator(), Discriminator()
