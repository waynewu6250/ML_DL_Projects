class Config:

    data_path = "data-for-imagefolder/"
    save_path = "gen_imgs-wgan/"
    img_size = 96
    batch_size = 32
    max_epoch = 400
    lr1 = 2e-4 #encoder learning rate
    lr2 = 2e-4 #decoder learning rate
    beta1 = 0.5

    #For WGAN
    lr = 1e-4
    wgan = False
    
    inf = 100 #noise feature map number
    gnf = 64 #generator feature map number
    dnf = 64 #discriminator feature map number

    vis = False
    env = 'GAN'
    netd_path = None #'checkpoints/netd_49.pth'  
    netg_path = None #'checkpoints/netg_49.pth' 

    train_d = 1 #how many batchs to train discriminator once
    train_g = 5 #how many batchs to train generator once
    train_vis = 20 #how many batchs to visualize once
    train_save = 10 #how many epochs to save image once
    train_save_model = 50 #how many epochs to save model once

    # generate parameters
    gen_imgs = "result.png"
    gen_pool_num = 5000
    gen_select_num = 64
    gen_mean = 0
    gen_std = 1
    
    gpu = False

opt = Config()


