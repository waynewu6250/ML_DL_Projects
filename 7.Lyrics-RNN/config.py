class Config:

    filename = 'data/files/all.txt'
    pickle_path = '12-24.npz'
    model_path = 'checkpoints/12-24.pth'

    lr = 1e-3
    num_epoch = 40
    batch_size = 64
    embedding_dim = 128
    latent_dim =256
    
    print_every = 5
    use_gpu = False
    vis = False

    start_words = '青花瓷' #First words
    prefix_words = '七里香' #Style
    max_gen_len = 300


opt = Config()
