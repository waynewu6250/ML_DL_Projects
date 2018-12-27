import os
import torch as t
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
    netd.to(device)
    netg.to(device)
    
    optimizer_g = RMSprop(netg.parameters(), opt.lr)
    optimizer_d = RMSprop(netd.parameters(), opt.lr)

    #====================================================#
    #                   START TRAINING                   #
    #====================================================#

    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    noises = t.randn(opt.batch_size, opt.inf, 1, 1).to(device)
    fix_noises = t.randn(opt.batch_size, opt.inf, 1, 1).to(device)

    for epoch in range(opt.max_epoch):
        
        print("epoch %d:"%epoch)
        netd.train(); netg.train()
        
        for ii, (imgs,_) in tqdm.tqdm(enumerate(dataloader)):
            imgs = imgs.to(device)

            # Train discriminator
            if ii % opt.train_d == 0:
                for p in netd.parameters(): p.data.clamp_(-0.01, 0.01)
                optimizer_d.zero_grad()
                
                real_loss = netd(imgs)
                noises.data.copy_(t.randn(opt.batch_size, opt.inf, 1, 1))
                fake_imgs = netg(noises).detach() # we don't want train generator at this stage
                fake_loss = netd(fake_imgs)
                lossd = real_loss-fake_loss
                lossd.backward()
                
                optimizer_d.step()

            
            # Train generator
            if ii % opt.train_g == 0:

                optimizer_g.zero_grad()

                # noise should be classified as true_labels
                noises.data.copy_(t.randn(opt.batch_size, opt.inf, 1, 1))
                fake_imgs = netg(noises)
                labels = netd(fake_imgs)
                lossg = labels.mean(0).view(1)
                lossg.backward()

                optimizer_g.step()


            #Visualize
            if opt.vis and ii % opt.train_vis == 0:
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().numpy() * 0.5 + 0.5, win='fixfake')
                vis.images(imgs.data.numpy() * 0.5 + 0.5, win='real')
                vis.plot('loss',lossd.data.item())
            
            print("\nDLoss: ",lossd.data.item())
            print("GLoss: ",lossg.data.item())
            
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