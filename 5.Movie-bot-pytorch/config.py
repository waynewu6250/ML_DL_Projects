import torch

class Config:

    # Define hyper parameter
    use_cuda = True if torch.cuda.is_available() else False

    # for dataset
    data_path = "data/movie_lines.tsv"
    toks_path = "data/all_toks_new.bin"

    # for training
    epochs = 10
    batch_size = 256
    learning_rate = 2e-3

    # for model
    char_dim = 50
    latent_dim = 50
    mxlen= 20
    teacher_forcing_ratio = .5
    model_path = "checkpoints/epoch-9.pth"

    gpu = True


opt = Config()