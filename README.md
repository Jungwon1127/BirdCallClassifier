# BirdCallClassifier

## Overview of Competition
As the “extinction capital of the world,” Hawai'i has lost 68% of its bird species, the consequences of which can harm entire food chains. Researchers use population monitoring to understand how native birds react to changes in the environment and conservation efforts. But many of the remaining birds across the islands are isolated in difficult-to-access, high-elevation habitats. With physical monitoring difficult, scientists have turned to sound recordings. Known as bioacoustic monitoring, this approach could provide a passive, low labor, and cost-effective strategy for studying endangered bird populations.



Current methods for processing large bioacoustic datasets involve manual annotation of each recording. This requires specialized training and prohibitively large amounts of time. Thankfully, recent advances in machine learning have made it possible to automatically identify bird songs for common species with ample training data. However, it remains challenging to develop such tools for rare and endangered species, such as those in Hawai'i.

The Cornell Lab of Ornithology's K. Lisa Yang Center for Conservation Bioacoustics (KLY-CCB) develops and applies innovative conservation technologies across multiple ecological scales to inspire and inform the conservation of wildlife and habitats. KLY-CCB does this by collecting and interpreting sounds in nature and they've joined forces with Google Bioacoustics Group, LifeCLEF, Listening Observatory for Hawaiian Ecosystems (LOHE) Bioacoustics Lab at the University of Hawai'i at Hilo, and Xeno-Canto for this competition.

In this competition, you’ll use your machine learning skills to identify bird species by sound. Specifically, you'll develop a model that can process continuous audio data and then acoustically recognize the species. The best entries will be able to train reliable classifiers with limited training data.

If successful, you'll help advance the science of bioacoustics and support ongoing research to protect endangered Hawaiian birds. Thanks to your innovations, it will be easier for researchers and conservation practitioners to accurately survey population trends. They'll be able to regularly and more effectively evaluate threats and adjust their conservation actions.

*Inspired by BirdCLEF 2022*

## Index
1. [Approach](#approach)
2. [Preprocessing the data](#preprocessing-the-data)
3. [Creating, training and evaluating the model](#creating-training-and-evaluating-the-model)

### Approach
Using the approach used by Magdalena Kortas in https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b, we will be building a similar model that converts bird audio files into melfrequency spectrogram images that will then be run through a convolutional neural network to detect bird species. 

#### Definitions
**Convolutional Neural Network**

A convolutional neural network is a class of artificial neural network that is most commonly used for image recognition. The most significant feature and why it is so powerful for image recognition is it ability to "successfully capture the Spatial and Temporal dependencies" of that images you pass into it (Saha). Many of the competitors in previous years of BirdCLEF have found that they have found best results using a convolutional neural network classifier that takes a visual representation of bird calls as the input as opposed to other classes of neural networks.

Learn more about them here: <br>
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

**Spectrogram vs Mel Frequency Spectrogram**

This is what a spectrogram looks like:
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c5/Spectrogram-19thC.png"
<p>
  </p>
<p align = "center">
Fig.1 - Spectrogram
</p>

The Pacific Northwest Seismic Network explains it like this: "A spectrogram is a visual way of representing the signal strength, or 'loudness', of a signal over time at various frequencies present in a particular waveform.  Not only can one see whether there is more or less energy at, for example, 2 Hz vs 10 Hz, but one can also see how energy levels vary over time." To generate a spectrogram, one of the approaches you can take is to do a fourier transform. In this case, a fourier transform would "decompose" (change) the audio signal into its constituent frequencies (simply put, it shows which pitches are the most significant of loud over a period of time). 

<p align="center">
  <img src="https://librosa.org/doc/0.7.2/_images/librosa-feature-melspectrogram-1.png"
<p>
  
</p>
<p align = "center">
Fig.2 - Mel Frequency Spectrogram
</p>

A mel frequency spectrogram is a variant of the spectrogram where the frequencies (y-axis) are converted the mel scale.

The significance of the mel scale is that it is a logarithmic unit of pitch that is meant to emulate the way humans percieve sound as people, while quite good at diffentiating lower pitch, struggle as the frequency of the pitch increases. Since humans are the ones labeling and listing to the data, it would follow that we would also want the neural network to "learn" the patterns that human listeners noticed.

The logic behind this approach is that a convolutional neural network will be able to pick up on the correlation between the frequency patterns of the bird calls (due to the spectrogram mesuring frequncy of the audio file on the veritcal axis) as well as the the correlation in time (the x-axis of the spectrogram).

### Preprocessing the data
First, we got our data from the BirdCLEF competition here: https://www.kaggle.com/competitions/birdclef-2022/data. This data gives you a variety of data, from a folder of over 150 different bird calls to csv files that provide general information about bird names, species, scientific name, etc. For this neural network, we will only be utilizing the sound files (.ogg format, but most sound formats should work). 

Using a pandas dataframe, we loaded in all the audio files with their respective descriptions. Then we filtered the dataframe to return to us the 6 birds with the highest number of individual data points we can work with (each class had 500 audio files). 

We took the top 6 birds with the most individual data points (500 each) and it came out to be: 
1. Barn Owl (brnowl) 
2. Common Sandpiper (comsan) 
3. House Sparrow (houspa) 
4. Mallard (mallar3)
5. Northern Cardinal (norcar)
6. Eurasian Skylark (skylar)

Utilizing the librosa python package, we quite simply directly convert the audio files into their respective melspectrogram representations. 

Here are a few examples:
<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166244222-f659b55c-a1e8-4d08-b7cc-8dd866666189.png"
<p>
</p>
<p align = "center">
Fig.4 - Barn Owl Mel Frequency Spectrogram
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166244342-4c0c8903-a5af-455a-a6b9-646b9a656902.png"
<p>

</p>
<p align = "center">
Fig.5 - Common Sandpiper Mel Frequency Spectrogram
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166244383-d4a30a89-99ce-492a-80af-9ff36625ccfc.png"
<p>

</p>
<p align = "center">
Fig.6 - House Sparrow Mel Frequency Spectrogram
</p>


### Creating, training, and evaluating the model

Using a 80, 10, 10 split of training, validation, and testing data, we created a convolution neural network using the Keras library. Using this initial model we obtained the following results:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166316819-535114df-3aa5-472c-9fe2-698383096c5f.png"
<p>
</p>
<p align = "center">
Fig.7 - Novel Convolutional Neural Network
</p>

Wanting to further increase our accuracy, we decided to use MobileNetV2, a pretrained image classification model provided by Keras. The results are as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166317764-b18c86e2-a379-4be8-b2fe-28790fad7d62.png"
<p>
</p>
<p align = "center">
Fig.8 - MobileNetV2 Transfered Neural Network
</p>

Heatmap of MobileNetV2 edition of the CNN:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66310121/166317975-2c6e3fcc-abcc-4c80-8082-149c9dc73450.png"
<p>


Sources: <br>
https://ebird.org/home <br>
https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53 <br>
https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b <br>
https://www.kaggle.com/competitions/birdclef-2022/data <br>
https://pnsn.org/spectrograms/what-is-a-spectrogram <br>
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
