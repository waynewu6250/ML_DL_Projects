<div id="part_9"></div>

# 9. DeepEyeNet

This is a collaborative deep learning project called DeepEyeNet, originally licensed by [DeepEyeNet repo](https://github.com/huckiyang/DeepEyeNet) and referenced by [Image Captioning repo](https://github.com/yashk2810/Image-Captioning). The task is to generate medical descriptions of a typical retinal image input by using deep learning high level framework: keras. 

* **Dataset**

The dataset is from http://imagebank.asrs.org/ <br>
(Please only access the Images via Retina Image Bank Website. A full credit index has been set in each folder of a specific disease. Please check https://imagebank.asrs.org/terms-of-use#contributors and https://imagebank.asrs.org/terms-of-use#visitors)

* **Notebooks**

There are four jupyter notebooks to illustrate the whole projects:

1. [Image_captioning_VGG16.ipynb](https://github.com/waynewu6250/ML_DL_Projects/blob/master/9.DeepEyeNet/Image_captioning_VGG16.ipynb), [Image_captioning_InceptionV3.ipynb](https://github.com/waynewu6250/ML_DL_Projects/blob/master/9.DeepEyeNet/Image_captioning_Inception_V3.ipynb):
Step-by-step process to process the data and build the model & giving some example outputs.

2. [Image_captioning_evaluation.ipynb](https://github.com/waynewu6250/ML_DL_Projects/blob/master/9.DeepEyeNet/Image_captioning_evaluation.ipynb):
Evaluate the results by common image captioning metrics.

3. [Image_captioning_keyword_model.ipynb](https://github.com/waynewu6250/ML_DL_Projects/blob/master/9.DeepEyeNet/Image_captioning_keyword_model.ipynb):
Variation of different keyword embedded model to test the performances.

You can download all required files (preprocessed data, model checkpoints, evaluate results) in the following link: <br>
https://drive.google.com/open?id=1D9JJ8y7iNdmqYnfdkjAROtJFSApg3l9A


* **At a glance**

Here we use Keras with Tensorflow backend for the code. 
1. VGG16, VGG19, InceptionV3 model are used for extracting the image features. (VGG19 is the same notebook with VGG16)
2. We also preprocess the medical descriptions of each training data.The first, we feed it into LSTM model to get word features. 
3. Construct a custom-RNN model to feed each word and image feature at each time step and predict next word.
4. Here I create a keyword-model to feed each specified keywords in training data for each image. The uncertain number of keywords are averaged to be a word vector and fed simultaneously with image vector into the final model.
5. At prediction stage, I am using Greedy search and Beam search with k=3 for predicting the captions of the images.


# Results
## Evaluation
I train the final model with and without the keyword reinforced to see the difference. For simplicity, I chose four main types of diseases around ~2000 images for training 3 epochs. Pretrained GLOVE word embeddings are used.

The loss value of **1.8489** has been achieved without the keywords reinforced which gives barely satisfactory results.
The loss value of **0.5446** has been achieved with the keywords reinforced and the model converges much faster than the previous model.

For evaluation: we use bleu, CIDEr, Rouge scores to evaluate our results.
The average bleu scores are calculated as follows, as with all the training ~2000 images.

VGG16:

|  Model  | Phase | CIDEr  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE  |
| ------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| Normal  | Train | 6.3607 | 0.8449 | 0.7535 | 0.6352 | 0.5938 | 0.8633 |
| Keyword | Train | 9.0316 | 0.9602 | 0.8950 | 0.7879 | 0.7570 | 0.9672 | 
| Normal  | Test  | 3.5747 | 0.6255 | 0.5162 | 0.3828 | 0.3493 | 0.6532 |
| Keyword | Test  | 4.6886 | 0.6853 | 0.5964 | 0.4654 | 0.4388 | 0.7127 |

VGG19:

|  Model  | Phase | CIDEr  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE  |
| ------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| Normal  | Train | 6.3607 | 0.8410 | 0.7523 | 0.6320 | 0.5882 | 0.8633 |
| Keyword | Train | 8.3280 | 0.9664 | 0.9491 | 0.9394 | 0.8770 | 0.9672 | 
| Normal  | Test  | 3.5747 | 0.6879 | 0.5389 | 0.4871 | 0.4213 | 0.6532 |
| Keyword | Test  | 4.6886 | 0.7387 | 0.6216 | 0.5837 | 0.5267 | 0.7127 |

Inceptoin V3:

|  Model  | Phase | CIDEr  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE  |
| ------- | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| Normal  | Train | 6.3607 | 0.8692 | 0.8026 | 0.7720 | 0.6940 | 0.8633 |
| Keyword | Train | 8.3280 | 0.9664 | 0.9491 | 0.9394 | 0.8770 | 0.9672 | 
| Normal  | Test  | 3.5747 | 0.6879 | 0.5389 | 0.4871 | 0.4213 | 0.6532 |
| Keyword | Test  | 4.6886 | 0.7387 | 0.6216 | 0.5837 | 0.5267 | 0.7127 |

The calculation could be checked in the jupyter notebook `Image_captioning_evaluation.ipynb`.
And the score results are also stored in `results/results.txt`

## Example readouts

You can check out some examples below. The rest of the examples are in the jupyter notebook `Image_captioning_VGG16.ipynb` and `Image_captioning_Inception_V3.ipynb`. You can run the Jupyter Notebook and try out some retinal image examples for the medical description.

**1. Training Image:**<br>
<img src="train_img.png" width="300"><br>
```
* Model without keywords:
Predicted Caption: 71 year old white female. srnv md. re 20/50 le 20/20.
Ground Truth Caption: 67-year-old white male. srnv-md. re 20/40 le 20/15.
* Model with keywords:
Predicted Caption: 67 year old white male. srnv md. re 20/40 le 20/15.
Ground Truth Caption: 67-year-old white male. srnv-md. re 20/40 le 20/15.
```

**2. Testing Image:**<br>
<img src="test_img.png" width="300"><br>
```
* Model without keywords:
Predicted Caption: diabetic retinopathy fundus image.
Ground Truth Caption: diabetic retinopathy fundus image.
* Model with keywords:
Predicted Caption: diabetic retinopathy fundus image.
Ground Truth Caption: diabetic retinopathy fundus image.
```

# Dependencies:
* Keras 1.2.2
* Tensorflow 0.12.1
* tqdm
* numpy
* pandas
* matplotlib
* pickle
* PIL
* glob



