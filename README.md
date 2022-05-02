# BirdCallClassifier

**A convolutional artifical neural network that takes in a audio file of birdcalls and determines the species of the bird.** 

*Inspired by BirdCLEF 2022*

## Index
1. [Approach](#approach)
2. [Preprocessing the data](#preprocessing-the-data)
3. [Creating the model](#creating-the-model)
4. [Training and evaulation](#training-and-evaluation)

### Approach
Using the approach used by Magdalena Kortas in https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b, we will be building a similar model that converts bird audio files into mel-spectrogram images that will then be run through a convolutional neural network to detect bird species. Why a convolutional neural network? Many of the competitors in previous years of BirdCLEF have found that they have found best results using a convolutional neural network classifier that takes a visual representation of bird calls as the input. 

The logic behind this approach is that a convolutional neural network will be able to pick up on the correlation between the frequency patterns of the bird calls (due to the spectrogram mesuring frequncy of the audio file on the veritcal axis) at as well as the the correlation in time (the x-axis of the spectrogram). 

Why a 

### Preprocessing the data
First, we got our data from the BirdCLEF competition here: https://www.kaggle.com/competitions/birdclef-2022/data. This data gives you a variety of data, from a folder of over 150 different bird calls to csv files that provide general information about bird names, species, scientific name, etc. For this neural network, we will only be utilizing the sound files (.ogg format, but most sound formats should work). 





### Creating the model

### Training and evaluation
