---
title: "JHU - Practical Machine Learning - Course Project"
author: "Myr Balada"
date: "December 06, 2020"
output: html_document
---
## Overview
Human Activity Recognition - HAR - has emerged as a key research area in the last years and is gaining increasing attention by the pervasive computing research community (see picture below, that illustrates the increasing number of publications in HAR with wearable accelerometers), especially for the development of context-aware systems. There are many potential applications for HAR, like: elderly monitoring, life log systems for monitoring energy expenditure and for supporting weight-loss programs, and digital assistants for weight lifting exercises.

## Introduction  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  

## Data Preprocessing  
```{r, cache = T}
rm(list=ls())                # free up memory for the download of the data sets
library(knitr)
library(caret)
library(lattice)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
library(tibble)
library(bitops)
library(randomForest)
library(corrplot)
set.seed(12345)
```
### Download the Data
```{r, cache = T}
# set the URL for the download
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}
```  
### Read the Data
After downloading the data from the data source, we can read the two csv files into two data frames.  
```{r, cache = T}
# download the datasets
training <- read.csv("./data/pml-training.csv")
testing <- read.csv("./data/pml-testing.csv")

# create a partition with the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
```

```{r, cache = T}
# Data for TrainSet 
dim(TrainSet)
```

```{r, cache = T}
# Data for TestSet 
dim(TestSet)
```

The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict. 

### Clean the data
In this step, we will clean the data and get rid of observations with missing values as well as some meaningless variables.
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. The Near Zero variance (NZV) variables are also removed and the ID variables as well.
```{r, cache = T}
# remove variables with Nearly Zero Variance
NZV <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -NZV]
TestSet  <- TestSet[, -NZV]
```
```{r, cache = T}
# Data for TrainSet 
dim(TrainSet)
```
```{r, cache = T}
# Data for TestSet 
dim(TestSet)
```

```{r, cache = T}
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
```  
Next, we get rid of some columns that do not contribute much to the accelerometer measurements.
```{r, cache = T}
# remove identification only variables (columns 1 to 5)
TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
```
```{r, cache = T}
# Data for TrainSet 
dim(TrainSet)
```
```{r, cache = T}
# Data for TestSet 
dim(TestSet)
```

Now, the cleaned training data set contains 19622 observations and 54 variables, while the testing data set contains 20 observations and 53 variables. The "class" variable is still in the cleaned training set.

### Correlation Analysis

A correlation among variables is analyzed before proceeding to the modeling procedures.  
```{r, cache = T}
corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", outline = T, addgrid.col = "darkgray", method = "color", type = "lower", cl.pos = "r", cl.cex = 0.5, tl.cex = 0.4, mar = c(0.5,0,0.5,0), tl.col = rgb(0, 0, 0))
```
The highly correlated variables are shown in dark colors in the graph above. To make an evem more compact analysis, a PCA (Principal Components Analysis) could be performed as pre-processing step to the datasets. Nevertheless, as the correlations are quite few, this step will not be applied for this assignment.

# Prediction Model Building

Three methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below.
A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

### Method 1: Random Forest
```{r, cache = T}
# model fit
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

```{r, cache = T}
# prediction on Test dataset
predictRandForest <- predict(modFitRandForest, newdata=TestSet)
confMatRandForest <- confusionMatrix(table(predictRandForest, TestSet$classe))
confMatRandForest
```

```{r, cache = T}
# plot matrix results
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```


### Method 2: Decision Trees
```{r, cache = T}
# model fit
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)
```

```{r, cache = T}
# prediction on Test dataset
predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(table(predictDecTree, TestSet$classe))
confMatDecTree
```

```{r, cache = T}
# plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```

### Method 3: Generalized Boosted Model

```{r, cache = T}
# model fit
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
```

```{r, cache = T}
# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(table(predictGBM, TestSet$classe))
confMatGBM
```

```{r, cache = T}
# plot matrix results
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```

# Applying the Selected Model to Test Data

The accuracy of the 3 regression modeling methods above are:

- Random Forest : **0.999**
- Decision Tree : **0.7342**
- GBM : **0.9871**

The Random Forest model will be applied to predict the 20 quiz results (testing dataset) as shown below.


```{r, cache = T}
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```

# Conclusion
About the analysis developed in all these tests, it was possible to fit the model to obtain a high degree of precision in the sample observations.
There is a lot of contaminating data that in subsequent analysis, it should not be included (at least in my case), since it confuses being part of the decision tree. All those variables that are very close to zero or that do not contribute quality of information to the results should be eliminated.

Despite these remaining questions on missing data in the samples, the random forest model with cross-validation produces a surprisingly accurate model that is sufficient for predictive analytics.