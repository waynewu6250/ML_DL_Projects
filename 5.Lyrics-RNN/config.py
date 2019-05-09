class Config:

    filename = 'data/files/all.txt'
    pickle_path = 'data/data_files/12-26-all.npz'
    model_path = 'checkpoints/12-26-all.pth'

    lr = 1e-3
    num_epoch = 40
    batch_size = 64
    embedding_dim = 128
    latent_dim =256
    
    print_every = 5
    use_gpu = False
    vis = False

    start_words = '黑色幽默' #First words
    prefix_words = '' #Style
    max_gen_len = 300


opt = Config()
