#Movielens Capstone Project

#RECOMMENDATION SYSTEM

#DATA PREPARATION

# Create edx set, validation set (final hold-out test set)
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#METHOD AND ANALYSIS
#Data Preparation

#Convert to tibble and explore
edx %>% as_tibble()

#Unique rating
unique(edx$rating)

#Number of unique rating
length(unique(edx$rating))

#Unique rating count
edx %>% group_by(rating) %>% count %>% arrange(desc(n))

#Summary of unique users and unique movies 
edx %>% summarize(n_users = n_distinct(userId), 
                  n_movies = n_distinct(movieId))
#movies that have greater  rating in descending order of magnitude
num_movies <- edx %>% count(movieId) %>% arrange(desc(n)) %>% top_n(10) %>% pull(movieId)

#Data set of 15 users and 7 movies
edx %>% 
  filter(movieId%in%num_movies) %>% 
  filter(userId %in% c(1:10)) %>% 
  select(userId, title, rating) %>%
  mutate(title = str_remove(title, ", The"),
         title = str_remove(title, ":.*"),
         title = str_remove_all(title, "[(13459)]+")) %>%
  pivot_wider(names_from = title, values_from = rating)

#count distribution by movie
edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bin = 30, fill = "royal blue", color = "black") +
  scale_x_log10()+
  ggtitle("Movies")

#count distribution by userId
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bin = 30, fill = "royal blue", color = "black") +
  scale_x_log10() +
  ggtitle("userId")

#function that computes the RMSE for vectors of rating and their corresponding predictors
RMSE <- function(true_ratings, predicted_rating){
  sqrt(mean((true_ratings - predicted_rating)^2))
}

#train and test sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index]
test_set <- edx[test_index]

#Simplest Model
#Average of all ratings of the training set
mu <- mean(train_set$rating)
mu

#RMSE for the mean of all ratings
rmse_mu <- RMSE(test_set$rating, mu)
rmse_mu


#Movie Effects

#Compute movie effects, bi
bi_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
head(bi_avgs)

#Distribution of bi
bi_avgs %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bin = 10, fill ="Royal blue", color = "black") +
  ggtitle("Movie effects")

#Merge test_set with bi__avgs and drop NAs
test_set_one <- test_set %>% 
  left_join(bi_avgs, by = "movieId") %>% drop_na()

#Prediction and RMSE for movie effect model
predicted_ratings <- mu + test_set_one$b_i
rmse_bi <- RMSE(test_set_one$rating, predicted_ratings)
rmse_bi



#Adding user effects

#average rating for users
train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bin = 30, fill = "royal blue", color = "black") +
  ggtitle("UserId")

#Compute user effects, b_u
bu_avgs <- train_set %>% 
  left_join(bi_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
head(bu_avgs)

#Compute predicted values on test set
test_set_two <- test_set %>% 
  left_join(bi_avgs, by='movieId') %>%
  left_join(bu_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u ) %>% drop_na()

#Compute RMSE results
rmse_bu <- RMSE(test_set_two$rating, test_set_two$pred)
rmse_bu


#Regularization

#Distinct movieId and title from edx
movie_titles <- edx %>% 
  select(movieId, title) %>% 
  distinct()
head(movie_titles)

#10 best movies with bi and rating
train_set %>% count(movieId) %>% 
  left_join(bi_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) 

#10 worst movies with bi and rating
train_set %>% count(movieId) %>% 
  left_join(bi_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) 

#Regularized movie effects model

#Define a set of lambdas
lambdas <- seq(0, 10, 0.25)

#Compute RMSEs for regularized movie effect model
rmses_bi_reg <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)     
  
  bi_reg  <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  test_data <- test_set %>%
    left_join(bi_reg, by = "movieId") %>%
    mutate(pred = mu + b_i) %>% drop_na() 
  
  return(RMSE(test_data$rating , test_data$pred))
})

#Plot lambdas vs RMSEs
qplot(lambdas, rmses_bi_reg, main = "Regularized movie effect model")

#Select lambda with the least RMSE
lambda <- lambdas[which.min(rmses_bi_reg)]
lambda
min(rmses_bi_reg)


#Regularized Movie and User effects model

#Define a set of lambdas
lambdas <- seq(0, 10, 0.25)

#Compute the RMSEs for regularized movie and user effects models
rmses_bu_reg <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  bi_reg <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  bu_reg <- train_set %>% 
    left_join(bi_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  test_data_one <- 
    test_set %>% 
    left_join(bi_reg, by = "movieId") %>%
    left_join(bu_reg, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    drop_na()
  
  return(RMSE(test_data_one$rating, test_data_one$pred))
})

qplot(lambdas, rmses_bu_reg, main = "Regularized movie and user effects model") 

#Select lambda with least RMSE
min(rmses_bu_reg)
lambda <- lambdas[which.min(rmses_bu_reg)]
lambda


#Matrix Factorization using train, test and validation sets

#Install and load the recosystem
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

#Specifies train,testing and validation sets from R objects into recosystem input formats
train_reco <- data_memory(user_index = train_set$userId, item_index = train_set$movieId, rating = train_set$rating, index1 = TRUE)
test_reco <- data_memory(user_index = test_set$userId, item_index = test_set$movieId, rating = test_set$rating, index1 = TRUE)
validation_reco <- data_memory(user_index = validation$userId, item_index = validation$movieId, rating = validation$rating, index1 = TRUE)

#Create a model Object
r <- Reco()
set.seed(1234, sample.kind = "Rounding")

#Call the $tune() method to select best tuning parameters
opts <-  r$tune(train_reco, opts = list(dim = c(10, 20, 30),
                                        costp_l1 = 0, costq_l1 = 0,
                                        lrate = c(0.05, 0.1, 0.2), nthread = 1))
#show the best tuning parameters
opts

#Train the model
r$train(train_reco, opts = opts$min)

#Predict using the test set
pred_rating_reco <- r$predict(test_reco, out_memory())

#Evaluate the  RMSE using test set
rmse_test_reco <- RMSE(test_set$rating, pred_rating_reco)
rmse_test_reco

#Predict using validation set
pred_validation_ratings_reco <- r$predict(validation_reco, out_memory())

#Evaluate the RMSE using validation set
rmse_validation_reco <- RMSE(validation$rating, pred_validation_ratings_reco)
rmse_validation_reco

#Result of models
rmse_results <- tibble(method = c("Just the average", 
                                  "Movie Effect Model", 
                                  "Movie + User Efects Model",
                                  "Regularized Movie Effect Model",
                                  "Regularized Movie + User Effects Model",
                                  "Matrix factorization using test set",
                                  "Matrix factorization using Validation set"), 
                       RMSE = c(rmse_mu,
                                rmse_bi,
                                rmse_bu,
                                min(rmses_bi_reg),
                                min(rmses_bu_reg),
                                rmse_test_reco,
                                rmse_validation_reco))
rmse_results 



