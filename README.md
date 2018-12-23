# Machine learning and Deep learning related projects by Ting-Wei Wu
My side project gallery for collections and refinement

### 1. AI robots on telegram
This is the nlp project modified by the final project of coursera course: advanced machine learning specialization-Natural Language Processing.  
It will be put on the telegram messenger supported by amazon web service (AWS) EC2 server and named wayne-bot (waynewu86Bot). <br>
Basically it will do the simple conversations. <br> 
it will also serve as a stackoverflow assistant where you can ask it about code questions and it will return stackoverflow related links.

---

### 2. Image captioning
This is the CV-related project modified by the final project of coursera course: advanced machine learning specialization-Introduction to Deep Learning.   
It can be also referred to the following respositories: <br>
```
1) Coursera-advanced-machine-learning-specialization/ 1.Introduction-to-Deep-Learning/ 10. with_pic_week6_final_project_image_captioning_clean.ipynb
```

It will caption the given image and return a descriptive sentence that depicts the graph.  

---

### 3. Twitter hashtags
This is the pyspark project that will analyze the most frequent hashtags used by individual users.   
It can be also referred to the following respositories: <br>
```
1) My-Sample-Projects/ 1.Machine_Learning/ 5.pyspark/ Twitter_hashtags.ipynb
2) My-Sample-Projects/ 3.NLP/ 10.hashtag_bilstm.ipynb
```
---

### 4. Movie Bot with keras
This is the nlp project completely done by Ting-Wei Wu with data preprocessing and keras seq2seq model establishment. <br>

Run `python train.py` to train the model on dataset preferably with gpu. <br>
Run `python test.py` could interact with the bot with simple conversations which could be trained better with more computational sources. <br>

Training data are extracted from movie_lines.tsv. Detailed descriptions are shown in the jupyter notebook: LHW4-Short-base.ipynb.

---

### 5. Movie Bot with pytorch
This is the nlp project completely done by Ting-Wei Wu with data preprocessing and pytorch seq2seq model establishment. <br>
It is same with the project Movie-bot-keras except new implementation with pytorch model.

Training data are extracted from movie_lines.tsv. Detailed descriptions are shown in the README.md in the subfolder.

-To begin with, for the **data extraction** part:
```
python load.py
```
will load the data from movie_lines.tsv and print out the first set of training data.
```
python batch.py
```
will generate the minibatches for the model.
-  Next, for the **model trainig and testing** part:
```
python train.py
```
will start training the data based on the fed minibatches and core unit used. <br>
(Computation are limited on personal pc so that the performance could be much improved by running on gpu-based server)
```
python test.py
```
will start a trained chatbot with basic conversational functions to interact with the user.

---

### 6. Deep Convolutional Generative Adversarial Networks for Naruto character generation
This is the deep learning project completely done by Ting-Wei Wu. Basically, it simulated the style drawing from Naruto figures to construct new naruto characters by artificial intelligence. <br>

Here, we implemented convolutional neural network based generator & discriminator structures with pytorch framework to adversarially compete against each other to generate new stype figures that match with existing characters. <br>
(Thanks to pytorch book: [chenyuntc github](https://github.com/chenyuntc/pytorch-book/tree/master/chapter7-GAN%E7%94%9F%E6%88%90%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F))

- To start, I download the images from internet: [Zerochan: Naruto](https://www.zerochan.net/NARUTO) and [IMDB](https://www.imdb.com/title/tt6342474/mediaindex?page={}&ref_=ttmi_mi_sm) <br>
  It could be extracted by using the following command inside data/ folder:
  ```
  python scraping.py
  ```
  And the images are stored as follows:
 ```
 data/
└── imgs/
    ├── Akatsuki.%28NARUTO%29.240.92787.jpg
    ├── Akatsuki.%28NARUTO%29.240.202153.jpg
    ├── Akatsuki.%28NARUTO%29.240.241011.jpg
    ...
 ```

- Face detection:
 Then we do the face detection to extract the faces inside the images that we have downloaded by using opencv package:
 ```
 python detect_face.py
 ```

- To use:
 1. To allow vidom for visualization, please run `python2 -m visdom.server` on the terminal first.
 2. Train:
 ```
 python main.py train --gpu --vis=False
 ```
 3. Generate images:
 ```
 python main.py generate --vis=False \
            --netd-path=checkpoints/netd_250.pth \
            --netg-path=checkpoints/netg_250.pth \
            --gen-img=result.png \
            --gen-num=64
 ```


