class Config:

    data_path = "data-for-imagefolder-hatsune/"
    save_path = "gen_imgs-hatsune/"
    img_size = 96
    batch_size = 128
    max_epoch = 400
    lr1 = 2e-4 #encoder learning rate
    lr2 = 2e-4 #decoder learning rate
    beta1 = 0.5
    
    inf = 100 #noise feature map number
    gnf = 64 #generator feature map number
    dnf = 64 #discriminator feature map number

    vis = False
    env = 'GAN'
    netd_path = 'checkpoints-hatsune/netd_99.pth'  
    netg_path = 'checkpoints-hatsune/netg_99.pth' 

    train_d = 1 #how many batchs to train discriminator once
    train_g = 5 #how many batchs to train generator once
    train_vis = 20 #how many batchs to visualize once
    train_save = 10 #how many epochs to save image once
    train_save_model = 50 #how many epochs to save model once

    # generate parameters
    gen_imgs = "result-hatsune.png"
    gen_pool_num = 5000
    gen_select_num = 64
    gen_mean = 0
    gen_std = 1
    
    gpu = False

opt = Config()


