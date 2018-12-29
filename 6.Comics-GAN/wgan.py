import os
import torch as t
from torch.autograd import Variable
from torch.optim import RMSprop
import torchvision as tv
import tqdm
from model import Generator, Discriminator
from config import opt

def train(**kwargs):

    #allow cuda
    device=t.device('cuda') if opt.gpu else t.device('cpu')

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    # visualize
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)

    # dataset
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.img_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         drop_last=True
                                         )

    # model
    netg, netd = Generator(), Discriminator()
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    
    def weight_init(m):
        # weight_initialization: important for wgan
        class_name=m.__class__.__name__
        if class_name.find('Conv')!=-1:
            m.weight.data.normal_(0,0.02)
        elif class_name.find('Norm')!=-1:
            m.weight.data.normal_(1.0,0.02)
        #     else:print(class_name)

    netd.apply(weight_init)
    netg.apply(weight_init)

    netd.to(device)
    netg.to(device)
    
    optimizer_g = RMSprop(netg.parameters(), opt.lr)
    optimizer_d = RMSprop(netd.parameters(), opt.lr)

    #====================================================#
    #                   START TRAINING                   #
    #====================================================#

    noises = Variable(t.randn(opt.batch_size, opt.inf, 1, 1)).to(device)
    fix_noises = Variable(t.randn(opt.batch_size, opt.inf, 1, 1).normal_(0,1)).to(device)
    
    one=t.FloatTensor([1]*32)
    mone=-1*one

    for epoch in range(opt.max_epoch):
        
        print("epoch %d:"%epoch)
        
        for ii, (imgs,_) in tqdm.tqdm(enumerate(dataloader)):
            imgs = Variable(imgs).to(device)
            one = one.to(device)
            mone = -1*one.to(device)

            # Train discriminator
            if ii % opt.train_d == 0:
                for p in netd.parameters(): p.data.clamp_(-0.01, 0.01)

                netd.zero_grad()
                ## train netd with real img
                output=netd(imgs)
                output.backward(one)
                ## train netd with fake img
                fake_imgs=netg(noises).detach()
                output2=netd(fake_imgs)
                output2.backward(mone)
                optimizer_d.step()

            
            # Train generator
            if ii % opt.train_g == 0:

                netg.zero_grad()
                noises.data.normal_(0,1)
                fake_pic=netg(noises)
                output=netd(fake_pic)
                output.backward(one)
                optimizer_g.step()


            #Visualize
            if opt.vis and ii % opt.train_vis == 0:
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().numpy() * 0.5 + 0.5, win='fixfake')
                vis.images(imgs.data.numpy() * 0.5 + 0.5, win='real')
                vis.plot('loss',output.data[0])
            
        #Save img
        if epoch % opt.train_save == 0:
            tv.utils.save_image(fake_imgs.data, '%s/%s.png' % (opt.save_path, epoch), normalize=True, range=(-1, 1))
            
        #Save model
        if (epoch+1) % opt.train_save_model == 0:
            t.save(netd.state_dict(), 'checkpoints-wgan/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints-wgan/netg_%s.pth' % epoch)

if __name__ == "__main__":
    import fire
    fire.Fire()