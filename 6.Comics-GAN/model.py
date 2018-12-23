from torch import nn
from config import opt

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
                    # Input dimension: inf x 1 x 1
                    nn.ConvTranspose2d(opt.inf, opt.gnf*8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(opt.gnf*8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(opt.gnf*8, opt.gnf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.gnf*4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(opt.gnf*4, opt.gnf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.gnf*2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(opt.gnf*2, opt.gnf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.gnf),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(opt.gnf, 3, 5, 3, 1, bias=False),
                    nn.Tanh()
                    # Output dimension: 3 x 96 x 96
        )
    
    def forward(self, input):
        return self.main(input)

#Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                    # Input dimension: 3 x 96 x 96
                    nn.Conv2d(3, opt.dnf, 5, 3, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(opt.dnf, opt.dnf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.dnf*2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(opt.dnf*2, opt.dnf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.dnf*4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(opt.dnf*4, opt.dnf*8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(opt.dnf*8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(opt.dnf*8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                    # Output Dimension: 1 (Probability)
        )

    def forward(self, input):
        return self.main(input).view(-1)


