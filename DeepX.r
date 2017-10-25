
#rm(list = setdiff(ls(), lsf.str()));

mainDirectory = "C:/RODRIGO/DeepX/";

## load the functions
source(file.path(mainDirectory,"functions.r"))

########################################################################################
##------ 1 - load data
########################################################################################
# Getting a random and balanced sub data set. I have to do this to can to run the classifiers. If it's needed, it's possible to define as 1 the proportion of the original data set to get. 
train_data_set = loadSuSetCsvData(file.path(mainDirectory,"train_dataset.csv"),1/3);


########################################################################################
##------ 2 - Analysis whit a histogram of all pixels (this step can be ignored to run the classifier) 
########################################################################################
# Transforming the DF in a unique vector. 
temp_analysis = as.vector(train_data_set[,1]);
for (col in 2:(length(train_data_set[1,])-1)){
  temp_analysis=c(temp_analysis,as.vector(train_data_set[,col]))
}
# Ploting the histograms of original data:
qplot(temp_analysis[temp_analysis>0],geom="histogram",bins = length(unique(temp_analysis)))

#min_pixel_value was defined around 17% of the max valeu to try eliminated noisy pixels
min_pixel_value = 45;

# Ploting the histograms with just the values keeped:
qplot(temp_analysis[temp_analysis>min_pixel_value],geom="histogram",bins = length(unique(temp_analysis[temp_analysis>min_pixel_value])));
rm(temp_analysis);


########################################################################################
##------ 3 - replace noisy values based on step 2
########################################################################################
min_pixel_value = 45;
x = subset(train_data_set, select = -c(class));
y = train_data_set$class;
x[x<min_pixel_value] = 0;
rm(train_data_set);


########################################################################################
##------ 4 - Feature Selection/Tranformation
########################################################################################
# 4.1 - select just the pixels that don't have zero variance.
good_indexes = getIndexesWithVariance(x);
x_selected = x[,good_indexes];
rm(x);

# 4.2 - Normalization, just to make a standard data set for data mining. This is not necessary in this data since the pixels have a natural scale.
x_selected_norm=x_selected/255;

## 4.3 - appling the PCA
#print("PCA");
#system.time(PCA_model <- prcomp(x_selected_norm, 
#                          scale. = T));
#
### 4.3.1 - Analysing the PCA results (this step can be ignored to run the classifier) 
##compute standard deviation of each principal component
#std_dev <- PCA_model$sdev;
##compute variance
#pr_var <- std_dev^2;
##proportion of variance explained
#prop_varex <- pr_var/sum(pr_var);
##Plot of the cumulative variance from PCA components
#plot(cumsum(prop_varex), xlab = "Principal Component",
#     ylab = "Cumulative Proportion of Variance Explained",
#     type = "b")
#yValues = seq(from = 0, to = 1, by = 0.1);
#xValues = seq(from = 0, to = 700, by = 25);
#axis(2,at=yValues,labels=yValues)
#axis(1,at=xValues,labels=xValues)
## We can see that with the 100 firsts components of de PCA we have 80% of the variance of the data and with the 200 firsts, we have 90%.
#rm(yValues,xValues,prop_varex,pr_var,std_dev);
#
### 4.3.2 Selecting just the N top PCA vectors
#number_of_PCA_vectors = 200;
#x_PCA_selected = as.data.frame(PCA_model$x[,1:number_of_PCA_vectors]);
#
###4.4 - LDA
#system.time(LDA_ <- LDAFeatureTransformation(x_selected,y));
#x_LDA = LDA_$newData;
#LDA_model = LDA_$LDA_model;

########################################################################################
##------ 5 - Data normalization
########################################################################################
#x_PCA_selected = normalization(x_PCA_selected)$newData;
#x_LDA = normalization(x_LDA)$newData;



########################################################################################
##------ 6 - Train the classifiers
########################################################################################

#### SVM
#system.time(SVM_PCA_Model <- trainSVMClassifier(x=x_PCA_selected,y=y));
#system.time(SVM_LDA_Model <- trainSVMClassifier(x=x_LDA,y=y));
#
#### ANN
#system.time(ANN_LDA_Model <- trainANNClassifier(x=x_LDA,y=y,hiddenNeurons_param=c(55,35),stepmax_param=2000));
#system.time(ANN_PCA_Model <- trainANNClassifier(x=x_PCA_selected,y=y,hiddenNeurons_param=c(55,35),stepmax_param=2000));

## DNN 
library(h2o);
h2o.init(nthreads=3);
system.time(DNN_model <- trainDNNClassifier(x=x_selected_norm,y=y,directoryPath=mainDirectory));


########################################################################################
##------ 7 - Testing the classifiers
########################################################################################
print("TESTING......");

## 1 - reading the test data
test_set = read.csv(file.path(mainDirectory,"test_dataset.csv"));
test_set$class = paste("L_",test_set$class,sep="");
x_test = subset(test_set, select = -c(class));
y_test = test_set$class;
y_test_df = as.data.frame(y_test);
colnames(y_test_df) = "y";

## 3 - appling data cleaning
x_test[x_test<min_pixel_value] = 0;
rm(test_set);

## 4 - selecting the same predictors of train data
x_test_selected = x_test[,good_indexes];
rm(x_test);

## natural normalization
x_test_selected_norm=x_test_selected/255;

### applying the PCA and LDA transformatio and feature selection
#x_PCA_test =as.data.frame(predict(PCA_model, newdata = x_test_selected_norm))[,1:number_of_PCA_vectors];
#x_LDA_test = as.matrix(x_test_selected) %*% LDA_model$scaling;
#x_LDA_test = as.data.frame(x_LDA_test);

## 5- min-max normalization
#x_LDA_test = normalization(x_LDA_test)$newData;
#x_PCA_test = normalization(x_PCA_test)$newData;

##training and testing the KNN calssifier
#print("KNN");
#system.time(KNN_PCA_Model <- trainKNNClassifier(x=x_PCA_selected,y=y,x_test=x_PCA_test));
#round(mean(KNN_PCA_Model == y_test),digits = 2)
#system.time(KNN_LDA_Model <- trainKNNClassifier(x=x_LDA,y=y,x_test=x_LDA_test,k_param =50));
#round(mean(KNN_LDA_Model == y_test),digits = 2)
#
###test SVM
#print("SVM");
#system.time(acc_svm_PCA <- testSVMClassifier(x_PCA_test,y_test,SVM_PCA_Model))
#system.time(acc_svm_LDA <- testSVMClassifier(x_LDA_test,y_test,SVM_LDA_Model))
#acc_svm_PCA
#acc_svm_LDA
#
###test SVM
#print("ANN");
#system.time(acc_svm_PCA <- testANNClassifier(x=x_PCA_test,y=y_test,ANN_PCA_Model))
#system.time(acc_svm_LDA <- testANNClassifier(x=x_LDA_test,y=y_test,ANN_LDA_Model))
#acc_svm_PCA
#acc_svm_LDA 

##test DNN
write.csv(cbind(x_test_selected_norm,y_test_df),file = file.path(mainDirectory,"test_dataset_dnn.csv"));
dnn_test_set = h2o.uploadFile(file.path(mainDirectory,"test_dataset_dnn.csv"));
h2o.performance(DNN_model,newdata =dnn_test_set );

h2o.shutdown(prompt=FALSE)