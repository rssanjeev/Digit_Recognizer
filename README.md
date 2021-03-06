# Digit_Recognizer
The goal is to recognize digits 0 to 9 in handwriting images.
---
title: "IST_707_HW3"
author: "Sanjeev Ramasamy"
date: "March 31, 2019"
output:
  word_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#Loading the required packages:
```{r,eval=TRUE, results='hide', message=FALSE, warning=FALSE}
library(ggplot2)
library(caret)
library(ElemStatLearn)
library(klaR)
library(e1071)
library(mlbench)
```

#Data Load:
```{r}
data <- read.csv("sample_data.csv", header = TRUE)
test <- read.csv("sample-testdata.csv", header = TRUE)
data[,2:785] <- data[,2:785]/255.0
test[,2:785] <- test[,2:785]/255.0
#As the last entry in the test data is a blank row, we get rid of the row
test <- test[1:nrow(test)-1,]
test$label <- NULL
```

#Label exploration
```{r}
qplot(data$label,
      geom="histogram",
      binwidth = 0.5,  
      main = "Histogram for Digits", 
      xlab = "Digits",  
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2),
      xlim=c(0,12))
```

#Dependant Variable:
The dependant variable 'label' is in the string format, this has to be converted into 'factor' so that the classification algorithm will be able to perform hassle free.
```{r}
data$label <- as.factor(data$label)
```

#Data Split:
Let is now split this data into train & test.
```{r}
train_index <- createDataPartition(data$label, p = 0.8, list = FALSE)

data_train <- data[train_index, ]
data_test <- data[-train_index, ]
```

#Naive Bayes Classifier
Let us now run the naive bayes algorithm on the training data with the parameters set to default.
```{r}
start1 <- Sys.time()
#default_model <- train(label ~ ., data = data_train, method = "nb")
Sys.time() - start1
```

Let us have a look at the default model:
```{r}
#default_model
```

Predicting the labels on the testing data set:
```{r}
#pred <- predict(default_model, data_test)
```

Calculating the accuracy of the model:
```{r}
#confusionMatrix(pred, data_test$label)
```

Fine Tuned MOdel: Now, let us fine tune the model with additional parameters.
```{r}
start3 <- Sys.time()
nb_tuned <- train(label ~ ., data = data_train, method = "nb",
                trControl = trainControl(method = "none"),
                tuneGrid = expand.grid(fL = 1, usekernel = T, adjust = 1))
Sys.time() - start3
```

Let us have a look at the fine tuned model.
```{r}
nb_tuned
```

Predicting the labels on the testing data set:
```{r}
predict_nb_tuned <- predict(nb_tuned, newdata = data_test, type = "raw")
```

Calculating the accuracy of the model:
```{r}
confusionMatrix(predict_nb_tuned, data_test$label)
```

#KNN
K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining and intrusion detection.

It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).

Model Training
```{r}
start1 <- Sys.time()
model_knn2 <- train(label ~ ., data = data_train, method = "knn",
                    tuneGrid = data.frame(k = seq(1, 25)),
                    trControl = trainControl(method = "repeatedcv",
                                           number = 10, repeats = 3))
Sys.time() - start1
```

Model review
```{r}
model_knn2
```

Predicting the labels on the testing data set:
```{r}
predict_knn <- predict(model_knn2, newdata = data_test, type = "raw")
```

Confusiont Matrix for checking accuracy
```{r}
confusionMatrix(predict_knn, data_test$label, positive = "pos")
```

#SUPPORT VECTOR MACHINE
A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both classification and regression purposes. SVMs are more commonly used in classification problems. So, lets go ahead and train the model on the trianing data:

```{r message=FALSE, warning=FALSE}
start1 <- Sys.time()
model_svm_linear <- train(label ~ ., data = data,
                          method = "svmLinear",
                          preProcess = c("center", "scale"),
                          trControl = trainControl(method = "cv", number = 3),
                          tuneGrid = expand.grid(C = seq(0, 1, 0.05)))
Sys.time()-start1
```

Model review:
```{r}
model_svm_linear
```

Predict the lable on the test data:
```{r}
predict_svm_linear <- predict(model_svm_linear, newdata = test)
```

Accuracy check for the model:
```{r}
confusionMatrix(predict_svm_linear, test$label)
```

As we can see that the fine tunes KNN model has the highest accuracy, let us run the model on the test data and try to predict the labels.
Predicting the label on test data
```{r}
predict_knn2 <- predict(model_knn2, newdata = test)
test['Label']=predict_knn2
```

The above steps were performed on the Kaggle's train dataset and the predicted labels were uploaded the test data prediction to Kaggle's website. Due to the size constraint we stay with the 3-fold cross-validation method.

```{r}
#train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)
#train[,2:785] <- train[,2:785]/255.0
test[,1:784] <- test[,1:784]/255.0
#train$label <- as.factor(train$label)

start1 <- Sys.time()
model_SVM_k <- train(label ~ ., data = train, method = "svmLinear",
                     preProcess = c("center", "scale"),
                    tuneGrid = expand.grid(C = seq(0, 1, 0.05)),
                    trControl = trainControl(method = "repeatedcv",
                                           number = 10, repeats = 3))
Sys.time() - start1


predict_knn2 <- predict(model_svm_linear, newdata = test)
test['Label']=predict_knn2

nrow(test)
sub <- data.frame(test$Label)
sub["ImageId"]<- seq.int(nrow(sub))
names(sub) <- c("Label", "ImageId")
# reorder by column name
sub <- subset(sub, select=c("ImageId", "Label"))
head(sub)
write.csv(sub, file="submission.csv",row.names=FALSE)
```
