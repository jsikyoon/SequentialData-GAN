### Sequential Data GAN
-----

This module is to generate sequential data with GAN implemented by LSTM.

The basic structure is followed ckmarkoh's git https://github.com/ckmarkoh/GAN-tensorflow

## Used Data
----
For making sequential data, MNIST 0-3 data are used (0->1->2->3).

## Modeling
----
Each modules, genrator and discreminator are designed with 2 layer LSTM and 1 layer Fully Connected Network.
Generator is designed as one-to-many model, which get one random vector as input, and generates sequential images.
Discriminator is designed as many-to-one model, which get sequential images, and decides that is real or fake ones.


## Results
----
Epoch 10
![alt tag](https://github.wdf.sap.corp/i338425/time-series_GAN/blob/master/figures/iter10.gif)

Epoch 30
![alt tag](https://github.wdf.sap.corp/i338425/time-series_GAN/blob/master/figures/iter30.gif)

Epoch 50
![alt tag](https://github.wdf.sap.corp/i338425/time-series_GAN/blob/master/figures/iter50.gif)

Epoch 400
![alt tag](https://github.wdf.sap.corp/i338425/time-series_GAN/blob/master/figures/iter400.gif)
