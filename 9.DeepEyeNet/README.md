<div id="part_9"></div>

# 9. DeepEyeNet

This is a collaborative deep learning project called DeepEyeNet, originally licensed by [DeepEyeNet repo](https://github.com/huckiyang/DeepEyeNet) and referenced by [Image Captioning repo](https://github.com/yashk2810/Image-Captioning). The task is to generate medical descriptions of a typical retinal image input by using deep learning high level framework: keras. 

The dataset is from http://imagebank.asrs.org/ <br>
(Please only access the Images via Retina Image Bank Website. A full credit index has been set in each folder of a specific disease. Please check https://imagebank.asrs.org/terms-of-use#contributors and https://imagebank.asrs.org/terms-of-use#visitors)


Here we use Keras with Tensorflow backend for the code. 
1. VGG16 is used for extracting the image features. 
2. We also preprocess the medical descriptions of each training data.The first, we feed it into LSTM model to get word features. 
3. Construct a custom-RNN model to feed each word and image feature at each time step and predict next word.
4. Here I create a keyword-model to feed each specified keywords in training data for each image. The uncertain number of keywords are averaged to be a word vector and fed simultaneously with image vector into the final model.
5. At prediction stage, I am using Greedy search and Beam search with k=3 for predicting the captions of the images.

# Results:
I train the final model with and without the keyword reinforced to see the difference. For simplicity, I chose four main types of diseases around ~2000 images for training 3 epochs. Pretrained GLOVE word embeddings are used.

The loss value of **1.8489** has been achieved without the keywords reinforced which gives barely satisfactory results.
The loss value of **0.5446** has been achieved with the keywords reinforced and the model converges much faster than the previous model.

The average bleu scores are calculated as follows, as with all the training ~2000 images.
```
* Original model with train imgs: 0.7354783252699579
* Original model with test imgs: 0.6450051351681783
* Keyword model with train imgs: 0.9283996395411378
* Keyword model with test imgs: 0.7225211741046165
```


You can check out some examples below. The rest of the examples are in the jupyter notebook `Image_captioning_VGG16.ipynb`. You can run the Jupyter Notebook and try out some retinal image examples for the medical description.

## Example readouts:

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



