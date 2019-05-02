import torch

class Config:

    # for english dataset
    data_path = "data/movie_lines.tsv"
    conversation_path = "data/movie_conversations.txt"
    results_path = "data/data.bin"
    prev_sent = 2

    # for chinese dataset
    chinese_data_path = "data/new_data.conv"
    chinese_results_path = "data/chinese_data.bin"

    # for training
    epochs = 120
    batch_size = 256
    learning_rate = 1e-4

    # for model
    char_dim = 300
    latent_dim = 500
    mxlen= 20
    teacher_forcing_ratio = .5
    model_path = "checkpoints/memory.pth"
    chinese_model_path = "checkpoints/memory_chinese.pth"

    attn = False
    chinese = True

opt = Config()