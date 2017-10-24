# EMNIST_classifiers
R code to apply classifiers to EMNIST dataset

## Problem description
Your company is developing a software to digitize handwritten text. Your team has already developed code to extract the image of each one of the characters in a given text. You are given the task of developing a machine learning model capable of reliably translating those images into digital characters. After some research, you find the [EMNIST dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset), which seems perfect for the task. Orginal link [here](https://github.com/deepxhq/rd-challenge/blob/master/challenges/ml-handwritten-character-recognition.md).

## Project overview
- Objective: To develop a handwritten character digits classifier
- Responsable: Rodrigo de Moraes
- Programming language: R
- Time spent: ~48 hours (including the documentation)

## How to run the code
	1 - download all files in this directory;
	2 - open the DeepX.r file;
	3 - replace the mainDirectory variable value to the path were the files where donwloaded;
	4 - run the code;

Obs. the original code runs the best classifier founded. But it is possible to change parameters,to uncomment lines to get other classifiers and results. The steps 2 and 4.3.1 are analysis steps an can be ignored to run the classifier.

## The dataset
The EMNIST dataset is a set of handwritten character digits derived from the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19)  and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset (link is external) . Further information on the dataset contents and conversion process can be found in the paper available [here](https://arxiv.org/pdf/1702.05373v1.pdf). Orginal link [here](https://www.nist.gov/itl/iad/image-group/emnist-dataset).

The orginal dataset MNIST, more information about it and reference works, including classification, results can be found [here](http://yann.lecun.com/exdb/mnist/).

For this project I choose the EMNIST Balanced dataset that are compoused by 131.000 samples and 47 classes (letters and numbers). As the name tell, the samples are equally distributed among the classes being 3.000 samples for each class and the Train and Test sets are already defined. This dataset are choosed because it was maked to facilitate the classification task once the calsses are balanced as discuting in this [paper](https://arxiv.org/pdf/1702.05373v1.pdf).

## Applied techiniques 
Based on the Knowledge Discovery in Databases (KDD) process, I used techiniques to clean, select and transform the data before to aplly Machine Learning classifiers. The first task in KDD was made by selection the dataset as described previously. Following It will described the other apllied techiniques.

### Data cleaning
To avoid simple noises, the train dataset was analysed to identify low pixels levels can be ignored change them to zero. To do this, a histogram with all the pixels and their values was made and it was possible to conclude that pixel values less than ~17% of the max value (255) would be zeroed, since around this percentage ends the first "coomb" on the histogram.

### Features Selection and Tranformation
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
The classification problem is a classical supervised classification problem since we have a train dataset labled. To do this task, it was tryed four differents classifiers. The KNN and SVM classifiers were adopted by looking some papers of litarature. The ANN classifier was tested too even though the data size is huge. Lestly, a DeepNeuralNetworks was used since it has a good potential in data sequencing as images data.

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
For the application of the classifiers where defined subdatasets from the original with a % of the number of samples of teh original set. It was necessarary because the computional coast of the techinique applyed is high and the computer used was just a notebook with a Core i5 and 8gb RAM memory. Analysis performed on theses subsets showed that the distribution and the variance of the data was minimally impacted.

For the preparation of the data, it where used two aproaches. One of them applied the PCA while the other applied the LDA transformation. To the PCA technique, was selected the firsts 200 principal components to defined the dataset train and for LDA, all the 46 predictors columns created was used. 

The sequence of the applied techiniques on the dataset was:
	1 - Selection a fraction of the data;
	2 - Data Cleaning;
	3 - Features selection and tranforamation;
	4 - Normalization.

With the data ready, it was pssible to apply the classifiers. For this step, a lot of combinations of the % of the number of samples of the original dataset and classifiers parameter (as K for KNN; cost and kernel for SVM; number of nerons, learning rate and activation function for ANN; and number of nerons, regularization and activation function for DNN) were tested.

For the application of KNN, SVM and ANN was used the R packages "class", "e1071" and "neuralnet" respectly. All of them does not provide funtions to antomaticly test many parameters values. So, to obtain preliminary results, the variation of parameters values of these three classifiers was mede manually. Unlike that, for the application of Deep Learning it was used the "h2o" package that provides a function able to test a lot of values and combinations of the parameters.

The preliminary test results showed that ANN and SVM are very costly and to try to find the bests parameters values would take a lot of time. The KNN revealed a good computional cost, but the test acurracy whit the approches applyed reached just around 60%. Finally, the Deep Neural Network was the classifier with the best test and time results. The differentiated architecture of the technique and the multi-thread process provedes by the h2o peackage allowed to reach ~80% od accuracy in the test set.

The final (and the best) classifier was obtained by the use the following approach:
	- Select a subset of the original train dataset (1/3);
	- Data cleaning removing possible noisy data;
	- Selection junt predictors with variance (remove images edges);
	- Natural normalization pixels values (divide by 255);
	- Train and test a Deep Neural Networks.

## Points to improve

As improvments would be do:

	- Apllay a under or over sampling techinique on the original imbalanced dataset to define a dataset with less number of samples but with a better representation of all data instead of a random selection;
	- Use other cleaning techniques specify to images data; 
	- Try more different sets of parameters for all classifiers;
	- A better results analysis to try to indentify critic classes; 
	- Use a hadoop plataform and the Mahout Samsara Environment to run the techniques with all dataset.



