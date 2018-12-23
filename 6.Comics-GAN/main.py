import os
import torch as t
from torch.optim import Adam
import torchvision as tv
import tqdm
from model import Generator, Discriminator
from config import opt
from torchnet.meter import AverageValueMeter

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
    
    optimizer_g = Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    #====================================================#
    #                   START TRAINING                   #
    #====================================================#

    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    noises = t.randn(opt.batch_size, opt.inf, 1, 1).to(device)
    fix_noises = t.randn(opt.batch_size, opt.inf, 1, 1).to(device)

    # METER
    errord = AverageValueMeter()
    errorg = AverageValueMeter()

    for epoch in range(opt.max_epoch):
        print("epoch %d:"%epoch)
        for ii, (imgs,_) in tqdm.tqdm(enumerate(dataloader)):
            imgs = imgs.to(device)

            # Train discriminator
            if ii % opt.train_d == 0:
                
                optimizer_d.zero_grad()
                
                # real img should be classified as true_labels
                labels = netd(imgs)
                loss_real = criterion(labels, true_labels)
                loss_real.backward()
                # noise should be classified as fake_labels
                noises.data.copy_(t.randn(opt.batch_size, opt.inf, 1, 1))
                fake_imgs = netg(noises).detach() # we don't want train generator at this stage
                labels = netd(fake_imgs)
                loss_fake = criterion(labels, fake_labels)
                loss_fake.backward()

                optimizer_d.step()

                errord.add((loss_real+loss_fake).item())
            
            # Train generator
            if ii % opt.train_g == 0:

                optimizer_g.zero_grad()

                # noise should be classified as true_labels
                noises.data.copy_(t.randn(opt.batch_size, opt.inf, 1, 1))
                fake_imgs = netg(noises)
                labels = netd(fake_imgs)
                loss_real_g = criterion(labels, true_labels)
                loss_real_g.backward()

                optimizer_g.step()

                errorg.add(loss_real_g.item())

            #Visualize
            if opt.vis and ii % opt.train_vis == 0:
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().numpy() * 0.5 + 0.5, win='fixfake')
                vis.images(imgs.data.numpy() * 0.5 + 0.5, win='real')
                vis.plot('errord', errord.value()[0])
                vis.plot('errorg', errorg.value()[0])
            
        #Save img
        if epoch % opt.train_save == 0:
            tv.utils.save_image(fake_imgs.data, '%s/%s.png' % (opt.save_path, epoch), normalize=True, range=(-1, 1))
            errord.reset()
            errorg.reset()
        #Save model
        if (epoch+1) % opt.train_save_model == 0:
            t.save(netd.state_dict(), 'checkpoints2/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'checkpoints2/netg_%s.pth' % epoch)

        

@t.no_grad()
def generate(**kwargs):

    #allow cuda
    device=t.device('cuda') if opt.gpu else t.device('cpu')

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    netg, netd = Generator().eval(), Discriminator().eval()
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    noises = t.randn(opt.gen_pool_num, opt.inf, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)

    #=============================#
    #select best images

    fake_imgs = netg(noises)
    labels = netd(fake_imgs).detach()

    indexes = labels.topk(opt.gen_select_num)[1] #select the top k indexes whose values are closest to 1 (real img)
    result = [fake_imgs.data[i] for i in indexes]
    
    # save images
    tv.utils.save_image(t.stack(result), opt.gen_imgs, normalize=True, range=(-1, 1))




if __name__ == "__main__":
    import fire
    fire.Fire()



