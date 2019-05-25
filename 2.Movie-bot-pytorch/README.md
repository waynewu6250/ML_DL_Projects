<div id="part_2"></div>

# 2. Movie-bot-with-pytorch
<p align="center">
    <img src="chickbot.png" height=300px center>
</p>

<div align="center">

**Seq2seq** | **Attention** | **Policy Gradient**

This is the nlp project completely done by Ting-Wei Wu with data preprocessing, pytorch seq2seq model establishment with attention mechanism and policy gradient reinforcement.
It is established based on the project [Movie-bot-keras](https://github.com/waynewu6250/ML_DL_Projects/tree/master/3.Movie-bot-keras) except new implementation and model specification with pytorch. <br>



(Please also refer to [StackBoxer Project](https://github.com/waynewu6250/StackBoxer) for full visualization of this project.) <br>
(Thanks to the model and data references: [RL-Chatbot](https://github.com/pochih/RL-Chatbot) and [Dialogue-Corpus](https://github.com/candlewill/Dialog_Corpus))


<div align="left">

# Introduction
There are three modes for the robot: Moviebot, Chickbot (雞寶) and YourFBbot

1. **Moviebot**: A robot trained with [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, which acts for daily english conversations. There are 4 modes for you to choose :)

2. **Chickbot (雞寶)**: A robot trained with dataset modified from [小黄鸡 语料](https://github.com/fate233/dgk_lost_conv/tree/master/results), which basically acts for daily chinese conversations and has consciousness that it's a chicken-like robot.

3. **YourFBbot**: A robot trained by my own fb messages, where you can download [Here](https://www.facebook.com/help/1701730696756992), which basically mimics how you chat with your friends.


# To USE
## 1. **Moviebot**
It will start a trained chatbot called moviebot with basic conversational functions to interact with the user, please feed it with english. <br>

There are four models for you to choose: <br>

### **1) RL Mode**
The best robot is trained and reinforced with policy gradient approach by setting up reward function to improve the outputs. To use, please type in terminal:
>
    bash run.sh test english

* To continue training it:

Please first go to [config.py](https://github.com/waynewu6250/ML_DL_Projects/blob/master/2.Movie-bot-pytorch/config.py) to set model_rl_path as `memory_new.pth` to load pretrained model or you can use existing model. So you won't train for a really long time. <br>

Use following command:
>
    bash run.sh train english

It will be stored as `rl-epoch-x.pth` <br>
The trained model is also stored in checkpoints/ as `model_rl.pth`.

---
### **2) Normal Mode**
The normal robot is trained with pytorch customized two-layer seq2seq model. Here we extract the words in raw data with different word frequency threshold,which sets the criteria how we disregards the word that appears too little times.

To test model with word threshold as **20**, use the following command:
>
    bash run.sh test english normal

* To continue training it:
>
    bash run.sh train english normal

will start training the english data based on the fed minibatches and core unit used. <br>

---
### **3) Normal Mode with a different threshold**
Now we are using dictionary only for word that appears 20 times and more in the dataset. 

To test model with word threshold as **5**, use the following command:
>
    bash run.sh test english normal-low

* To continue training it:

You can minimize it to 5 by changing the self.word_count_threshold parameter in [data/load.py](https://github.com/waynewu6250/ML_DL_Projects/blob/master/2.Movie-bot-pytorch/data/load.py).
Then the use following command:
>   
    python load.py
To create a new data.bin to use. Then replace it in config.py.
>
    bash run.sh train english normal-low

It will be stored as `epoch-x.pth`
The trained model is also stored in checkpoints/ as `memory.pth`.

### **4) Normal Mode with a different threshold**
The robot is also reinforced with the attention mechanism. To use, please type in terminal:
>
    bash run.sh test english attention

* To continue training it:

>
    bash run.sh train english attention

It will be stored as `epoch-x.pth`
The trained model is also stored in checkpoints/ as `model_attention.pth`.


## 2. **Chickbot**
```
bash run.sh test chinese
```
will start a trained chatbot called chickbot with basic conversational functions to interact with the user, please feed it with traditional chinese.

* To continue training it:
```
bash run.sh train chinese
```
will start training the chinese data based on the fed minibatches and core unit used. <br>





## 3. **YourFBbot**

First go to [config.py](https://github.com/waynewu6250/ML_DL_Projects/blob/master/2.Movie-bot-pytorch/config.py) and change fb parameter to True. Then, run the following commands:
```
bash run.sh test chinese
```
will start a trained chatbot called yourFBbot with similar tongues as you.

* To continue training it:
```
bash run.sh train chinese
```
will start training the chinese data based on the fed minibatches and core unit used. <br>


-------------------------------------------------

## Data Preprocessing Reference

Data are preprocessed directly with **python generator** to be made use at training stage

- To extract the data, go to data folder: <br>
1. **Moviebot**
```
python load.py
```
will load the data from `movie_lines.tsv`, `movie_conversations.txt` and print out the first set of training data, and save the data in `data_new.bin`.

2. **Chickbot**
```
python load_chinese.py
```
will load the data from `new_data.conv` and print out the first set of training data, and save the data in `chinese_data.bin`.

2. **YourFBbot**
```
python load_fb.py
```
will load the data from `only_messages.txt` and print out the first set of training data, and save the data in `fb_data.bin`.

- To generate the minibatches for the model.
```
python batch.py
```




