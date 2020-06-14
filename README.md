# AIML Online Capstone -Pneumonia Detection Challenge
Capstone Project from Group 9 - Computer Vision  , as part of the Post Graduate Program in Artificial Intelligence and Machine Learning - July 2019 batch (Group 16)



## Problem Statement
Chest X-rays are used as the quickest and most accessible method for the diagnosis of Pneumonia in patients who present with clinical symptoms like fever, cough and shortness of breath. All common forms of pneumonia like viral, bacterial and fungal pneumonia can be diagnosed from Chest X-Rays, which is an economical choice as well. 

In this Capstone project, we aim to predict the presence of pneumonia from Chest X-rays by detecting the presence of infection based on the lung opacity of the patient. Pneumonia detection models have a major role to play especially in places where the presence of an expert radiologist is not available, like in scenarios we experience currently, where the healthcare system is short-staffed due to the presence of a Pandemic like Covid-19.

Essentially, pneumonia detection is a binary classification problem, where the input is a chest X-Ray image 'X' and the output is a binary label 'y' which belongs to {0,1} indicating the absence or presence of pneumonia respectively. 

## Data
Data Source : (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)

The data provided includes 26684 Dicom images in the training dataset (from 26684 unique patients ) and 3000 Dicom Images in the test dataset. 
Dicom is the standard used in medical imaging ,to enable the transfer of images between different imaging equipments. It includes metadata about the image along with the image itself . There are two csv's which provide the detailed classification of the patients into 3 categories- Lung opacity, No Lung Opacity / Not Normal and Normal and also the broader classification of which patients were diagnosed with Pneumonia. For the patients who were diagnosed with pneumonia, bounding boxes are provided with details of X and Y co-ordinates and Height and Width of the boxes.

## Summary of Tasks Performed
### Step 1: Exploratory Data Analysis 
* Read and Extracted the dicom images using pydicom library
* Merged the data from images with the detailed class and pneumonia labelling dataframes
* Did a study of how the images in each class differed by displaying a sample from each class 
* Identified how the classes are distibuted and also did a bar plot to understand the split of the target classes
* Identified the distribution of age of patients in the dataset 
* Used various visualization tools to derive insights from the training dataset

#### Result of the EDA : [EDA and Dicom Metadata Extractor Jupyter Notebook](https://github.com/meashu31/AIML-Capstone-CV9/blob/master/merged_EDA.ipynb)


### Step 2 : Data Preparation and Preprocessing
* Converted the dicom images to .png format. (.png was selected as it results in reduced storage size for images, than storing into an npy array )
* Used the ImageDataGenerator class from Keras to create the image array and mask

#### Results of Data Preparation
* Custom module to convert dicom images to png : [Png Converter Module](https://github.com/meashu31/AIML-Capstone-CV9/blob/master/generate_png_data.ipynb)
* Data generator and mask preparation : [ Data Preprocessing Jupyter Notebook](https://github.com/meashu31/AIML-Capstone-CV9/blob/master/generator_with_images_and_masks.ipynb)


### Step 3 : Model Building

* Two models with MobileNet as backbone were explored
* [SegNet](https://arxiv.org/pdf/1511.00561.pdf) was explored by moving the encoder to Level 3, to optimize the resource utilization
* [U-Net](https://arxiv.org/pdf/1505.04597.pdf) ,which is a common architecture for biomedical imaging architecture, was explored 

#### Results 
* [Pneumonia Prediction Based on SegNet](https://github.com/meashu31/AIML-Capstone-CV9/blob/master/Segnet_Model_Pneumonia.ipynb)
* [Pneumonia Prediction Based on U-Net](link to be added). Currently this is based on MobileNet backbone

#### Summary of Models
<to be filled>

## Challenges Faced
* One of the challenges was to load and process large number of high resolution dicom images and further processing over them. Loading dicom images directly from the drive to code and iterating over each one of them resulted in exhaustive consumption of available RAM on google colab. To overcome this problem, images were firstly converted to .png format and instead of loading these images in bulk, a custom image generator was developed for a much better memory and time efficient iteration.
* Another challenge with available memory resources was unaviability of enough memory to load few models while processing over GPU. To overcome the problem instead of loading the full base model as encoder, we took up only some layers as encoder.

## Further Improvements
* Image Augmentation to further enhance the images
* Hyperparameter tuning will be attempted to increase the performance of the models
* U-Net will be explored with various backbones like VGG16, ResNet34 etc, to choose the best fitting model
* [CheXNet](https://arxiv.org/pdf/1711.05225.pdf), which is a widely accepted pneumonia detection algorithm, will be explored



