<div id="part_5"></div>

# 5. Recurrent Neural Network for Taiwanese song's lyrics generation
This is the deep learning project completely done by Ting-Wei Wu. <br>
Basically, it will generate some intriguing lyrics based on the major taiwanese songs' style. <br>
The networks are based on simple RNN model with major four singers. <br>

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
 
 <Data are preprocessed to feed into dataloader **directly** and be made use at training stage>
 
 - To use:
 1. To allow visdom for visualization, please run `python -m visdom.server` on the terminal first.
 
 2. Train:
 train with dataset chou.txt+eason.txt
 ```
 python main.py train --gpu=True \
                      --vis=False \
                      --pickle_path = 'data/data_files/12-24.npz' \
                      --model_path = 'checkpoints/12-24.pth'
 ```
 train with dataset all.txt (All four singers)
 ```
 python main.py train --gpu=True \
                      --vis=False \
                      --filename = 'data/files/all.txt' \
                      --pickle_path = 'data/data_files/12-26-all.npz' \
                      --model_path = 'checkpoints/12-26-all.pth'
 ```
 It will generate the numpy file for storage `all.npz, chou.npz, chou-eason.npz...`in data/datafiles/ folder. <br>
 And also train with defined epochs.
 
 3. Generate lyrics:
 ```
 python main.py generate --start-words = '我的歌' \
                         --prefix-words = '愛情' \
                         --pickle_path = 'data/data_files/12-26-all.npz' \
                         --model_path = 'checkpoints/12-26-all.pth'
 ```
 start words mean the start tokens you would like to be at the first of the lyrics.
 prefix words mean it will generate the style associated with the given words.
 load the pre-trained model and pickle_path to generate words.
 
 # Results:
 I use start-words "黑色幽默" as inputs: <br>
 The outputs become
 ```
 黑色幽默
 當黃昏色有形狀 有雨有多有多美夢
 我想起清醒的孤單 像你的美麗
 當我在身旁 心中的我 靜靜聆聽
 我是我 的心情 只有妳說
 我有用 你拉開 才會找到我更多
 一生唯一 一聲的傷心
 我 想起 或許是 那一年 是那麼冬季
 心中那麼 那麼多什麼 不能說也有風
 夢中 我已不停離開
 夢不醒是 黑色的世界
 你的風景 也不斷
 那愛的我 依然不需要 微微笑
 是我 想不到越來 已經我依賴
 為何人們為何都在 原來
 這個孤獨獨處是你 擁抱我的美
 一生一年 一千年
 我一定一千年一年 我為了我的生命
 我和我的倔強 一個人看外面
 是一個人的寂寞 ~~~
 那一年 一個人 看得見 我們的傷悲
 那是最是 我的寂寞 很多
 ```

 Then I use start-words "黑色幽默" and add prefix-words "擁抱" as inputs: <br>
 The outputs then become
 ```
 黑色幽默的很美
 我發現你還在還有 我的心中 你的溫暖 我的臉
 我想 我相信 也許你 永遠
 我都放棄自由 時刻告訴一點一句
 我們在等待 等到一生驕傲的路印
 我想要看 我的愛才不會乾淨
 那裡像是誰的浪漫 線索在彼此的以前
 你的靈魂 不是放手的消息
 我不懂 不懂 妳的痛呢 只是有妳在我活的
 我在等待 離開的希望是最後 在消失的時間
 我不能忘記 不斷一點心的
 一路有一天 我們都變成了
 多麼多麼能還能 沒多想你
 也許你怎麼竟有的愛
 不能感覺 沒有你的溫柔
 我只想你的明白
 不對我永遠不能再等 不如不想太多
 你說我 該怎麼樣
 我說了分手說不出來
 你說我有人陪我到哪裡也不能讓我們一個歌
 其實我們一起守護了
 就算我這個人為
 努力簡單就算簡單簡單
 我的腦袋
 你就不要失望
 我的心 有沒有一個眼裡
 你穿的時間咆哮
 我想你 不會懂
 我這麼了我的生活得我
 還能不完痛
 那一個小心情 的哭
 我們相愛的時候
 無論是一種一切
 ```

