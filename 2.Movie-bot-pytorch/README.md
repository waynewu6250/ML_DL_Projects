<div id="part_5"></div>

# 5.Movie-bot-with-pytorch
This is the nlp project completely done by Ting-Wei Wu with data preprocessing and pytorch seq2seq model establishment.
It is established based on the project Movie-bot-keras except new implementation and model specification with pytorch. <br>

* Please also refer to [StackBoxer Project](https://github.com/waynewu6250/StackBoxer) for full visualization of this project. <br>
(Thanks to the model and data references: [RL-Chatbot](https://github.com/pochih/RL-Chatbot) and [Dialogue-Corpus](https://github.com/candlewill/Dialog_Corpus))

## Introduction
There are two modes for the robot: Moviebot and Chickbot (雞寶)
1. **Moviebot**: A robot trained with [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, which acts for daily english conversations.
2. **Chickbot (雞寶)**: A robot trained with dataset modified from [小黄鸡 语料](https://github.com/fate233/dgk_lost_conv/tree/master/results), which basically acts for daily chinese conversations and has consciousness that it's a chicken-like robot.

-------------------------------------------------

## To USE
1. **Moviebot**
```
bash run.sh test english
```
will start a trained chatbot called moviebot with basic conversational functions to interact with the user, please feed it with english.

2. **Chickbot**
```
bash run.sh test chinese
```
will start a trained chatbot called chickbot with basic conversational functions to interact with the user, please feed it with traditional chinese.

-------------------------------------------------

## Data Preprocessing Reference

Data are preprocessed directly with **python generator** to be made use at training stage

- To extract the data, go to data folder: <br>
1. **Moviebot**
```
python load.py
```
will load the data from `movie_lines.tsv`, `movie_conversations.txt` and print out the first set of training data, and save the data in `data.bin`.

2. **Chickbot**
```
python load_chinese.py
```
will load the data from `new_data.conv` and print out the first set of training data, and save the data in `chinese_data.bin`.

- To generate the minibatches for the model.
```
python batch.py
```

-------------------------------------------------

## Training
1. **Moviebot**
```
bash run.sh train english
```
will start training the english data based on the fed minibatches and core unit used. <br>

2. **Chickbot**
```
bash run.sh train chinese
```
will start training the chinese data based on the fed minibatches and core unit used. <br>


