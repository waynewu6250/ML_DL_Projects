
# 7. Recurrent Neural Network for Taiwanese song's lyrics generation
This is the deep learning project completely done by Ting-Wei Wu. <br>
Basically, it will generate some intriguing lyrics based on the major taiwanese songs' style. <br>
The networks are based on simple RNN model with major three singers. <br>

To start, I extract the lyrics from the website [魔境歌詞網](https://mojim.com/) of the four popular chinese singers: 周杰倫 (Jay Chou), 陳奕迅 (Eason Chen), 林俊傑 (JJ Lin)., 五月天 (Mayday) <br>
It could be automated by using the following command inside data/ folder:
```
python scraping.py
```
And thd files are stored as follows:
 ```
 data/
└── files/
    ├── chou.txt
    ├── eason.txt
    ├── jj.txt
    ├── mayday.txt
    ├── all.txt
    ...
 ```
 
 - To use:
 1. To allow visdom for visualization, please run `python -m visdom.server` on the terminal first.
 
 2. Train:
 train with dataset chou.txt+eason.txt
 ```
 python main.py train --gpu=True \
                      --vis=False \
                      --pickle_path = '12-24.npz' \
                      --model_path = 'checkpoints/12-24.pth'
 ```
 train with dataset all.txt (All four singers)
 ```
 python main.py train --gpu=True \
                      --vis=False \
                      --filename = 'data/files/all.txt' \
                      --pickle_path = '12-26-all.npz' \
                      --model_path = 'checkpoints/12-26-all.pth'
 ```
 It will generate the numpy file for storage `all.npz, chou.npz, chou-eason.npz...`
 And also train with defined epochs.
 
 3. Generate lyrics:
 ```
 python main.py generate --start-words = '我的歌' \
                         --prefix-words = '愛情' \
                         --pickle_path = '12-24.npz' \
                         --model_path = 'checkpoints/12-24.pth'
 ```
 start words mean the start tokens you would like to be at the first of the lyrics.
 prefix words mean it will generate the style associated with the given words.
 load the pre-trained model and pickle_path to generate words.
 
 # Results:
 

