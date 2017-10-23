# EMNIST_classifiers
R code to apply classifiers to EMNIST dataset

## Problem description
Your company is developing a software to digitize handwritten text. Your
team has already developed code to extract the image of each one of the
characters in a given text. You are given the task of developing a
machine learning model capable of reliably translating those images into
digital characters. After some research, you find the [EMNIST dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset),
which seems perfect for the task.
Orginal link [here](https://github.com/deepxhq/rd-challenge/blob/master/challenges/ml-handwritten-character-recognition.md).

## Project overview
- Objective: To develop a handwritten character digits classifier
- Responsable: Rodrigo de Moraes
- Programming language: R
- Time spent: ~  hours (including the documentation)

## The dataset
The EMNIST dataset is a set of handwritten character digits derived from the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19)  and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset (link is external) . Further information on the dataset contents and conversion process can be found in the paper available [here](https://arxiv.org/pdf/1702.05373v1.pdf). Orginal link [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset).

The orginal dataset MNIST, more information about it and reference works, including classification, results can be found [here](http://yann.lecun.com/exdb/mnist/).

For this project I choose the EMNIST Balanced dataset that are compoused by 131.000 samples and 47 classes (letters and numbers). As the name tell, the samples are equally distributed among the classes being 3.000 samples for each class and the Train and Test sets are already defined. This dataset are choosed because it was maked to facilitate the classification task once the calsses are balanced as discuting in this [paper](https://arxiv.org/pdf/1702.05373v1.pdf).

## Applied techiniques 
Based on the Knowledge Discovery in Databases (KDD) process, I used techiniques to clean, select and transform the data before to aplly Machine Learning classifiers. The first task in KDD was made by selection the dataset as described previously. Following It will described the other apllied techiniques.

### Data cleaning
To avoid simple noises, I analysed the train dataset to identify low pixels levels can be ignored change them to zero. Therefore I plot a histogram with all the pixels and their values and it was possible to conclude that pixel values less than ~17% of the max value (255) woudl be zeroed, since aroud this percentage ends the first "coomb" on the histogram.

### Feature Selection and Tranformation
As all the predictors columns (features) are numerical and continuous, it was used two techniques to reduce the data dimensions and improuve the data representation. 
Following the literature, it was used the PCA transformation to select just the best features, thoese represent the most of variability of the data. To analyze it, was plotted a graph with the cumulative variance by the number of the features (PC) and it was possible to conclude that with the 100 firsts components represent 80% of the variance of the data and the 200 firsts represent 90%.
Another technique tryed was the LDA. It's a technique similar to PCA but it consideres the data classes to try to improve the data representation. The application of LDA results in a dataset with the number os predictors columns iquals the number of different classes minus one.

References:
- PCA: 
  - https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
- LDA: 
	- https://tgmstat.wordpress.com/2014/01/15/computing-and-visualizing-lda-in-r/
 	- https://rstudio-pubs-static.s3.amazonaws.com/35817_2552e05f1d4e4db8ba87b334101a43da.html
 	- https://medium.com/towards-data-science/is-lda-a-dimensionality-reduction-technique-or-a-classifier-algorithm-eeed4de9953  
	- https://medium.com/towards-data-science/is-lda-a-dimensionality-reduction-technique-or-a-classifier-algorithm-eeed4de9953a
	 

### Normalizaton
To apply the calssifiers after the tranformations, it was used a comum MAX-MIN normalization in the dataset.

### Classification
The classification problem is a classical supervised classification problem since we have a train dataset labled. To do this task, it was tryed to aplly four differents classifiers. It adopted the KNN and SVM classifiers looking some papers of litarature. The ANN classifier was tested too even though the data size is huge. Lestly, a DeepNeuralNetworks was used since it has a good potential in data sequencing as images data.

References:
- KNN: 
  - https://www.analyticsvidhya.com/blog/2015/08/learning-concept-knn-algorithms-programming/
  - https://www.datacamp.com/community/tutorials/machine-learning-in-r
  - https://www.r-bloggers.com/using-knn-classifier-to-predict-whether-the-price-of-stock-will-increase/
- SVM:
  - https://stackoverflow.com/questions/22009871/how-to-perform-multi-class-classification-using-svm-of-e1071-package-in-r
  - https://stackoverflow.com/questions/34328105/how-to-build-multiclass-svm-in-r	 
- ANN: 
  - https://www.kdnuggets.com/2016/08/begineers-guide-neural-networks-r.html 
  - https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/
  - https://www.r-bloggers.com/multilabel-classification-with-neuralnet-package/
- DNN:
  - https://www.r-bloggers.com/deep-learning-in-r-2/
  - http://docs.h2o.ai/h2o/latest-stable/h2o-docs/booklets/DeepLearningBooklet.pdf
	- https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning
	 
## Classifiers training
For the application of the classifiers where defined subdatasets from the original with a % of the number of samples of teh original set. It was necessarary because the computional coast of the techinique applyed is high and the computer used was just a notebook woth a Corei5 and 8gb RAM memory. Analysis performed on theses subsets showed that the distribution and the variance of the data was minimally impacted.

For the preparation of the data, it where used two aproaches. One of them applied the PCA while the other applied the LDA tranormation. To the PCA technique, was selected the firsts 200 principal components to defined teh dataset train and for LDA, all the 46 predictors columns created was used.

The sequence of the applied techiniques on the dataset was:
1 - Selection a fraction of the data;
2 - Data Cleaning;
3 - Feature selection and tranforamation;
4 - Normalization,

With the data ready, it was pssible to apply the classifiers. For this step, a lot of combinations of the % of the amouth of samples of the original set and classifiers parameter (as K for KNN; cost and kernel for SVM; number of nerons, learning rate and activation function for ANN; and number of nerons, regularization and activation function for DNN) where tested.

The 


