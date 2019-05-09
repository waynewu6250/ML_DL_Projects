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
    model_path = "checkpoints/memory.pth"
    chinese_model_path = "checkpoints/memory_chinese.pth"
    model_attention_path = None #"checkpoints/epoch-89.pth"
    model_rl_path = None

    attn = False
    chinese = False

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

opt = Config()