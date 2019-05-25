import torch

class Config:

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    attn = False
    rl = False
    chinese = False
    fb = False

    # for english dataset
    data_path = "data/movie_lines.tsv"
    conversation_path = "data/movie_conversations.txt"
    results_path = "data/data_new.bin"
    prev_sent = 2

    # for chinese dataset
    chinese_data_path = "data/only_messages.txt" #"data/new_data.conv"
    chinese_results_path = "data/fb_data.bin" if fb else "data/chinese_data.bin"

    # for training
    epochs = 120
    batch_size = 256
    learning_rate = 1e-4

    # for model
    char_dim = 300
    latent_dim = 500
    mxlen= 20
    model_path = "checkpoints/memory_rl.pth" #checkpoints/memory.pth" "checkpoints/memory_new.pth"
    chinese_model_path = "checkpoints/memory_fb.pth" if fb else "checkpoints/memory_chinese.pth"
    model_attention_path = "checkpoints/memory_attention.pth"
    model_rl_path = "checkpoints/memory_rl.pth"

opt = Config()