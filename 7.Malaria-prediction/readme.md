<div id="part_7"></div>

# 7. Malaria Cell Image Prediction

This is the deep learning project with CNN Resnet models from pytorch. Image data are provided via Kaggle dataset: [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/home). <br> The task is to classify cell images into two categories: parasitized or uninfected.

Also thanks to the kernels provided to this dataset. <br>
[malaria-detection-with-fastai-v1](https://www.kaggle.com/ingbiodanielh/malaria-detection-with-fastai-v1) <br>
[malaria-detection-with-pytorch](https://www.kaggle.com/devilsknight/malaria-detection-with-pytorch)

<img src="https://www.asianscientist.com/wp-content/uploads/bfi_thumb/Malaria-Parasite-Is-Driving-Human-Evolution-In-Asia-Pacific-2srft49tu93vzoqhwlircw.jpg" height="220" width="330">

<Data are put into torchvision dataset: **ImageFolder** and fed into dataloader to be made use at training stage>

* To use:
1. Train: train with dataset at image folder cell_images/. (skipped)

```
python main.py train --img_size = 224 \
                     --batch_size = 64 \
                     --max_epoch = 10 \
                     --lr = 1e-3 \
                     --model_path = None \
                     --train_save_model = 10
```

It will put the dataset inside the torchvision's Imagefolder container and make split into training, validation and testing data. Then they are trained with torchvision resnet50 model or self-defined model in `resnet.py`.

2. Test: 
I have trained the file on gpu and save it as 9.pth.
```
python main.py train --img_size = 224 \
                     --batch_size = 64 \
                     --max_epoch = 10 \
                     --lr = 1e-3 \
                     --model_path = "checkpoints/9.pth" \
                     --train_save_model = 10
```

3. Visualization:

The visualization is done in `Visualization.ipynb` <br>
First we will show the raw images with respectively parasitized and uninfected <br>
Then we will show the results that the model predicted.
Finally the roc curve and its area under curve performing in testing data set.





