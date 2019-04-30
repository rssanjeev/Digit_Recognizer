library(ggplot2)
library(caret)
library(ElemStatLearn)
library(klaR)
library(e1071)
library(mlbench)

data <- read.csv("sample-data.csv", header = TRUE)
testdata <- read.csv("sample-testdata.csv", header = TRUE)
data[,2:785] <- data[,2:785]/255.0


#Label explorationDude

hist(data$label)

ggplot(data=data, aes(data$label)) + geom_histogram()


qplot(data$label,
      geom="histogram",
      binwidth = 0.5,  
      main = "Histogram for Digits", 
      xlab = "Digits",  
      fill=I("blue"), 
      col=I("red"), 
      alpha=I(.2),
      xlim=c(0,12))

data$label <- as.factor(data$label)
#testdata$label <- as.factor(testdata$label)

train_index <- createDataPartition(data$label, p = 0.8, list = FALSE)

data_train <- data[train_index, ]
data_test <- data[-train_index, ]

#Predict using naive Baye's classifier Model 


start1 <- Sys.time()
default_model <- train(label ~ ., data = data_train, method = "nb")
nb_defaiult_time = Sys.time() - start1
nb_defaiult_time

pred <- predict(default_model, data_test)
confusionMatrix(pred, data_test$label)


start3 <- Sys.time()
model3 <- train(label ~ ., data = data_train, method = "nb",
                trControl = trainControl(method = "none"),
                tuneGrid = expand.grid(fL = 1, usekernel = T, adjust = 1))
Sys.time() - start3
predict_nb3 <- predict(model3, newdata = data_test, type = "raw")
confusionMatrix(predict_nb3, data_test$label)

#KNN
start1 <- Sys.time()
model_knn2 <- train(label ~ ., data = data_train, method = "knn",
                    tuneGrid = data.frame(k = seq(1, 25)),
                    trControl = trainControl(method = "repeatedcv",
                                           number = 10, repeats = 3))
Sys.time() - start1
predict_knn2 <- predict(model_knn2, newdata = data_test)

confusionMatrix(predict_knn2, data_test$label, positive = "pos")

#SVM
start1 <- Sys.time()
model_svm_linear <- train(label ~ ., data = data_train,
                          method = "svmLinear",
                          preProcess = c("center", "scale"),
                          trControl = trainControl(method = "boot", number = 25),
                          tuneGrid = expand.grid(C = seq(0, 1, 0.05)))


start1 <- Sys.time()predict_svm_linear <- predict(model_svm_linear, newdata = data_test)
confusionMatrix(predict_svm_linear, data_test$label)

Sys.time() - start1
model_svm_rbf <- train(label ~ ., data = data_train,
                       preProcess = c("center", "scale"),
                       tuneGrid = expand.grid(sigma = seq(0, 1, 0.1),
                                              C = seq(0, 1, 0.1)),
                       method = "svmRadial",
                       trControl = trainControl(method = "boot",
                                                number = 25))
Sys.time() - start1
predict_svm_rbf <- predict(model_svm_rbf, newdata = data_test)
confusionMatrix(model_svm_rbf, data_test$label)
plot(model_svm_rbf)

start1 <- Sys.time()
model_knn2 <- train(label ~ ., data = train, method = "knn",
                    tuneGrid = data.frame(k = seq(1, 25)),
                    trControl = trainControl(method = "cv",
                                             number = 3))




The above generated CSV file has been submitted in the Kaggle.

