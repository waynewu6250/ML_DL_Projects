class Config:

    data_path = 'eval_model/results.pkl'
    model_path = 'eval_model/checkpoints/19.pth'
    vocab_path = 'eval_model/vocab_list.pth'
    batch_size = 64
    embedding_dim = 300
    hidden_dim = 256
    num_layers = 2
    max_epoch = 20
    lr = 0.001
    num_classes = 47

    save_model = 5


    


opt = Config()