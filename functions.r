

# THE FUNCTIONS

## a funtion to load a calssification dataset from a path and get just a balanced subset defined by a fraction
loadSuSetCsvData <- function(path,percentageSubDataSet){
  data = read.csv(path);
  data$class = paste("L_",data$class,sep="");
  
  # Getting a random and balanced sub data set. 
  library(caret)
  indexes_SubDatasetRamdomBalanced = createDataPartition(data$class, p = percentageSubDataSet, list = FALSE);
  newData=data[indexes_SubDatasetRamdomBalanced,];
  rm(indexes_SubDatasetRamdomBalanced,data)
  return(newData)
}


## a funtion to get just the indexes os a dataset with variance >0
getIndexesWithVariance <- function(x){
  variance_analysis = nearZeroVar(x, saveMetrics = TRUE);
  good_indexes = as.vector(variance_analysis[,"zeroVar"] ==FALSE);
  rm(variance_analysis);
  return(good_indexes)
}


## a funtion to apply LDA transformation and get new data and the LDA model
LDAFeatureTransformation <- function(x,y){
  library(MASS);
  print("LDA");
  system.time(LDA_model <- lda(formula = y ~ .,
                               data = cbind(x,y),  
                               prior = rep(1,times=(length(unique(y))))/length(unique(y))));
  
  newData = as.matrix(x) %*% LDA_model$scaling;
  newData = as.data.frame(newData);
  return(list(newData=newData,LDA_model=LDA_model)) 
}


# a funtion to apply MAX-MIN normalization
normalization <- function(x){
  maxs = apply(x, 2, max);
  mins = apply(x, 2, min);
  newData = as.data.frame(scale(x,center = mins, scale = maxs - mins));
  r = list(maxs=maxs,mins=mins,newData=newData);
  return(r)
}


## a funtion to train the KNN classifier and to get the model trined
trainKNNClassifier <- function(x,y,x_test,k_param=round(sqrt(length(x[,1])),digits=0)){
  # as deafault, I am using the K parameter as the SQRT;
  library(class);
  knn_model = knn(train = x, 
                  test = x_test,
                  cl = y, 
                  k=k_param);
  return(knn_model)
}


## a funtion to train the SVM classifier and to get the model trined
trainSVMClassifier <- function(x,y){
  library(e1071)
  svm_model = svm(x, 
                  as.data.frame(y)$y,
                  kernel = "radial",
                  cost = 5,
                  probability = TRUE);
  return(svm_model) 
}


## a funtion to train the ANN classifier and to get the model trined
trainANNClassifier <- function(x,y,hiddenNeurons_param,learningrate_param=0.01,stepmax_param=1000){
  # To apply ANN, each label needs to be tranformed in a new comlumn in the DF data. The nnet package has a funtion to do this.
  library(nnet)
  dataTrain = cbind(x,class.ind(as.factor(y)));
  
  predictors_names = paste(names(x),collapse=' + ');
  labels_names = paste(names(dataTrain)[(length(x[1,])+1):length(dataTrain[1,])],collapse=' + ');
  f = paste(labels_names,'~',predictors_names);
  f = as.formula(f);
  
  library(neuralnet)
  ann_model = neuralnet(f,
                        dataTrain,
                        algorithm = "backprop",
                        act.fct = "logistic",
                        lifesign = "minimal",
                        learningrate = learningrate_param,
                        hidden=hiddenNeurons_param,
                        stepmax = stepmax_param,
                        linear.output = FALSE);
  return(ann_model)
}


## a funtion to train the DNN classifier with the combinations of various parameters and to get the model trined and the validation accuracy
trainDNNClassifier <- function (x,y,directoryPath){
  
  y=as.data.frame(y);
  y_name <- names(y);
  x_names <- names(x);
  
  path_ = file.path(directoryPath,"train_dataset_dnn.csv");
  write.csv(cbind((x),y),file = path_);
  dann_train_set = h2o.uploadFile(path_);
  rm(x,y)
  
  splits <- h2o.splitFrame(dann_train_set, 0.75, seed=1234);
  train  <- h2o.assign(splits[[1]], "train.hex"); #75%
  valid  <- h2o.assign(splits[[2]], "valid.hex"); #25%
  rm(dann_train_set)
  
  hyper_params <- list(
    activation=c("Tanh","Rectifier","TanhWithDropout"),
    hidden=list(c(350,250,150),c(250,150,100),c(300,150,75)),
    l1=c(1e-20,1e-10,1e-15)
    #l2=c(1e-20,1e-10,1e-15),
    #input_dropout_ratio=c(0,0.05)
  )
  
  model_grid <- h2o.grid("deeplearning",                        
                         hyper_params=hyper_params,                        
                         x = x_names,
                         y = y_name,                        
                         distribution="multinomial",                        
                         training_frame=train,                        
                         validation_frame=valid);
  
  model_grid <- h2o.getGrid(grid_id = model_grid@grid_id, 
                              sort_by = "accuracy",
                              decreasing = TRUE);
  
  D_ANN_model <- h2o.getModel(model_grid@model_ids[[1]]);

  return(D_ANN_model)
  
}


## a funtion to test the ANN classifier
testANNClassifier <- function(x,y,model){
  y = class.ind(as.factor(y));
  predictions = compute(model,x);
  results = predictions$net.result;
  original_values = max.col(y);
  results_ = max.col(results);
  acc=round(mean(results_ == original_values),digits=2);
  return(acc)
}


## a funtion to test the SVM classifier
testSVMClassifier <- function(x,y,model){
  library(nnet)
  y = class.ind(as.factor(y));
  results = predict(model, x, decision.values = TRUE, probability = TRUE);
  results = as.data.frame(attr(results, "probabilities"));
  results = results[,order(names(results))];
  original_values = max.col(y);
  results = max.col(results);
  acc=round(mean(results == original_values),digits=2);
  return(acc)
}
