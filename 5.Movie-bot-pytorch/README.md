# 5.Movie-bot-with-pytorch
This is the nlp project completely done by Ting-Wei Wu with data preprocessing and pytorch seq2seq model establishment.
It is same with the project Movie-bot-keras except new implementation with pytorch model.

<Data are preprocessed directly with **python generator** to be made use at training stage>

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

