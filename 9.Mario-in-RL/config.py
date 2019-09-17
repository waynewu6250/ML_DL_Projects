class Config:
    
    checkpoint = 1999
    path = None #'./checkpoints/model.ckpt-{}'.format(checkpoint)

    num_global_steps = 40000
    num_local_steps = 50
    gamma = 0.9
    beta = 0.001
    lr = 2e-4

opt = Config()
