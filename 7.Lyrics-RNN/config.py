class Config:

    filename = 'data/files/chou.txt'
    pickle_path = 'chou.npz'
    model_path = 'checkpoints/99.pth'

    lr = 1e-3
    num_epoch = 100
    batch_size = 5
    embedding_dim = 128
    latent_dim =256
    
    print_every = 5
    use_gpu = False
    vis = False

    start_words = '吳亭緯' #First words
    prefix_words = '這是愛情' #Style
    max_gen_len = 200


opt = Config()
