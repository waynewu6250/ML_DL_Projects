class Config:
    caption_path = "data/caption.pth"
    img_feature_path = "data/image_feature.pth"
    img_path = "data/imgs/"
    model_path = "checkpoints/9.pth"

    batch_size = 256
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    lr = 1e-3
    max_epoch = 10
    
    save_model = 1

    test_img = 'data/imgs/0.jpg'


opt = Config()