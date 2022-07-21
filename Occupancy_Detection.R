#OCCUPANCY DETECTION

#DATA PREPARATION

#Install and load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)


#Occupancy Detection Data sets
#https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#
#https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip


#Create temp file
temp = tempfile()

#Download zip file from internet to temp file
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip", temp)

#Extract the the 3 text files out of zip 
file_one <- unzip(temp, "datatest.txt")
file_two <- unzip(temp, "datatest2.txt")
file_three <- unzip(temp, "datatraining.txt")

#Load the data sets in R from the objects above
test_one <- read.csv(file_one)
test_two <- read.csv(file_two)
training <- read.csv(file_three)

#Join the three data sets sort the data by date
data <- full_join(training, test_one) %>% 
  full_join(test_two) %>% arrange(date)

rm(temp, file_one, file_two, file_three)


#DATA EXPLORATION

#Dimension of data
dim(data)

#data structure
str(data)

#unique values of occupancy status
unique(data$Occupancy)

#Convert Occupancy variable to factor data type
data$Occupancy <- as.factor(data$Occupancy)

#first six rows of the data set
head(data)

#summary statistics
summary(data)

#Percentage of occupancy
data %>% group_by(Occupancy) %>% 
  summarise(n = n()) %>% 
  mutate(percent = (n*100)/sum(n))


#Distribution of Temperature
p1 <- data %>% ggplot(aes(Temperature)) +
      geom_histogram(bins = 30, fill = "royal blue", color = "black") +
      ggtitle("Temperature")

#Distribution of Humidity         
p2 <- data %>% ggplot(aes(Humidity)) + 
       geom_histogram(bins = 30, fill = "royal blue", color = "black") +   
       ggtitle("Humidity")

#Distribution of Light
p3 <- data %>% ggplot(aes(Light)) + 
      geom_histogram(bins = 30, fill = "royal blue", color = "black") +
      ggtitle("Light")

#Distribution of CO2
p4 <- data %>% ggplot(aes(CO2)) + 
      geom_histogram(bins = 30, fill = "royal blue", color = "black") +
      ggtitle("CO2")

#Distribution of HumidityRatio  
p5 <- data %>% ggplot(aes(HumidityRatio)) + 
       geom_histogram(bins = 30, fill = "royal blue", color = "black") +
       ggtitle("HumidityRatio")

#Distribution of Occupancy Status
p6 <- data %>% ggplot(aes(Occupancy)) + 
      geom_bar(width = 0.5, fill = "royal blue", color = "black") + 
      ggtitle("Occupancy Status")

gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, nrow = 3, ncol = 2)

#compute correlation
cor_mat <- data %>% select(Temperature, Humidity, Light, CO2, HumidityRatio) %>% cor()
corrplot::corrplot(cor_mat, type = "lower", method = "number")


#MODELING

data1 <- data %>% select(Temperature, Humidity, Light, CO2, Occupancy)
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(data1$Occupancy, times = 1, p = 0.2, list = FALSE)
train_set <- data1[-test_index,]
test_set <- data1[test_index,]

#Logistics Regression

#fit a logistic model on the train set
glm_fit <- glm(Occupancy ~ ., data = train_set, family = binomial)
summary(glm_fit)


#Prediction using the test set
p_hat <- predict(glm_fit, newdata = test_set, type = "response")
y_hat <- factor(ifelse(p_hat > 0.5, 1, 0))

#Confusion matrix
accuracy_glm <- confusionMatrix(data = y_hat, reference = test_set$Occupancy)$overall["Accuracy"]
accuracy_glm


#K-Nearest Neighbors

#Using cross validation to select k that maximizes the accuracy

#Define a set of ks
ks <- seq(3, 251, 2)

#compute accuracies for values of ks
accuracy <- sapply(ks, function(k){
  
  knn_fit <- knn3(Occupancy ~ ., data = train_set, k = k)
  
  y_hat_knn <- predict(knn_fit, newdata = test_set, type = "class")
  
  cm_test <- confusionMatrix(data = y_hat_knn, reference = test_set$Occupancy)
  
  test_accuracy <- cm_test$overall["Accuracy"]
  
  return(test_accuracy)
})

#maximum accuracy
accuracy_knn <- max(accuracy)

#select k with the maximum accuracy
ks[which.max(accuracy)]
accuracy_knn


#Decision Trees

#Using cross validation to select complexity parameter that maximizes the accuracy

#Fit model using train set
train_rpart <- train(Occupancy ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 30)),
                    data = train_set)

#Prediction using the test_set
y_hat_rpart <- predict(train_rpart, test_set)

#Compute the accuracy
accuracy_rpart <- confusionMatrix(y_hat_rpart, test_set$Occupancy)$overall["Accuracy"]
accuracy_rpart


#RESULT
accuracy_results <- tibble(Methods = c("logistics regression",
                                       "KNN",
                                       "Decision Trees"),
                           Accuracy = c(accuracy_glm,
                                        accuracy_knn,
                                        accuracy_rpart))
accuracy_results

