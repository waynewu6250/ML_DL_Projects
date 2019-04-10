import torch

class Config:

    # for dataset
    data_path = "data/movie_lines.tsv"
    conversation_path = "data/movie_conversations.txt"
    results_path = "data/data.bin"
    prev_sent = 2

    # for training
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3

    # for model
    char_dim = 50
    latent_dim = 50
    mxlen= 20
    teacher_forcing_ratio = .5
    model_path = None #"checkpoints/epoch-0.pth"

    attn = False

opt = Config()