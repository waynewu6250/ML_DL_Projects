class Config:

    data_path = "cell_images/"
    img_size = 224
    batch_size = 64
    max_epoch = 10
    lr = 1e-3

    model_path = "checkpoints/9.pth"
    train_save_model = 10
    train_loss = 1

    train_split = 0.7
    val_split = 0.2
    test_split = 0.1


opt = Config()
