# Land-Cover-Classification-using-CNN
This repo consist of custom Convolutional neural network to classify land cover. After the hyperparameter optimization over %90 accuracy obtained.

## Dataset
The dataset which is used in this project is EuroSAT. EuroSAT is based on Sentinel-2 satellite images and consists of 27,000 labeled images with a total of 10 different classes. The patches measure 64x64 pixels. 

![image](https://user-images.githubusercontent.com/86148100/191494250-70323f3b-c815-4045-b572-42be4bc0db36.png)

## Hyperparamters
Batch size which is number of images that are used for calculating gradients at each step is equal to 128.
Epoch which is Number of times we will go through all the training images is equal to 20.
Learning rate which determines intensity of interaction between model and dataset equal to 1e-3.
Momentum which determines momentum for the gradient descent is equal to 0.9.
Weight decay which determines regularization factor to reduce overfitting is equal to 1e-4.

## Model 
The CNN model consists of 2 convolutions and these convolutions are followed by batch normalization, ReLu and max pooling respectively. First convolution output has 32 channels and the second convolution output has 64 channels. Both of the max pooling kernels are 2x2.

## Results
![image](https://user-images.githubusercontent.com/86148100/191497528-a1e81373-da45-4625-8df9-e98fd46440fc.png)

![image](https://user-images.githubusercontent.com/86148100/191497588-9e9ff769-e7bf-4b2d-b28e-f08d8d6242d0.png)


