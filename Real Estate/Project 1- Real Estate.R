# Load required libraries
library(dplyr)
library(car)
library(randomForest)
library(rpart) # For decision tree
if (!require(Metrics)) install.packages("Metrics", dependencies = TRUE)
library(Metrics) # For evaluation metrics


# Load data
test = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Real Estate\\housing_test.csv")
train = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Real Estate\\housing_train.csv")

# Combine datasets for preprocessing
test$Price = NA
train$data = 'train'
test$data = 'test'
all = rbind(train, test)

# Convert Postcode to character
all$Postcode = as.character(all$Postcode)


# Create dummy variables function
CreateDummies = function(data, var, freq_cutoff = 100) {
  t = table(data[, var])
  t = t[t > freq_cutoff]
  categories = names(t)[-1]
  
  for (cat in categories) {
    name = paste(var, cat, sep = "_")
    name = gsub("[^A-Za-z0-9_]", "_", name) # Sanitize variable names
    data[, name] = as.numeric(data[, var] == cat)
  }
  data[, var] = NULL
  return(data)
}


# Drop unnecessary columns
all = all %>% select(-SellerG, -Address, -Suburb)


# Create dummy variables
for_dummy_vars = c('Postcode', 'CouncilArea', 'Method', 'Type')
for (var in for_dummy_vars) {
  all = CreateDummies(all, var, 100)
}


# Impute missing values with mean
for (col in names(all)) {
  if (sum(is.na(all[, col])) > 0 & !(col %in% c("data", "Price"))) {
    all[is.na(all[, col]), col] = mean(all[all$data == 'train', col], na.rm = TRUE)
  }
}


# Split data back into train and test
trainf = all %>% filter(data == 'train') %>% select(-data)
testf = all %>% filter(data == 'test') %>% select(-Price, -data)


# Random Forest Model
rf_model = randomForest(Price ~ ., data = trainf)
rf_train_preds = predict(rf_model, newdata = trainf)


# Decision Tree Model
dt_model = rpart(Price ~ ., data = trainf, method = "anova")
dt_train_preds = predict(dt_model, newdata = trainf)


# Function to calculate Adjusted R²
adjusted_r2 = function(actual, predicted, num_predictors) {
  n = length(actual)
  r2 = cor(actual, predicted)^2
  adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_predictors - 1))
  return(adj_r2)
}


# Number of predictors (excluding target variable)
num_predictors_rf = ncol(trainf) - 1
num_predictors_dt = ncol(trainf) - 1


# Evaluation Metrics for Random Forest
rf_mae = Metrics::mae(trainf$Price, rf_train_preds)
rf_r2 = cor(trainf$Price, rf_train_preds)^2
rf_adj_r2 = adjusted_r2(trainf$Price, rf_train_preds, num_predictors_rf)


# Evaluation Metrics for Decision Tree
dt_mae = Metrics::mae(trainf$Price, dt_train_preds)
dt_r2 = cor(trainf$Price, dt_train_preds)^2
dt_adj_r2 = adjusted_r2(trainf$Price, dt_train_preds, num_predictors_dt)


# Display results
cat("Random Forest - MAE:", rf_mae, "R²:", rf_r2, "Adjusted R²:", rf_adj_r2, "\n")
cat("Decision Tree - MAE:", dt_mae, "R²:", dt_r2, "Adjusted R²:", dt_adj_r2, "\n")


# Compare models based on R² and Adjusted R²
if (rf_adj_r2 > dt_adj_r2) {
  cat("The Random Forest model performs better based on Adjusted R².\n")
} else {
  cat("The Decision Tree model performs better based on Adjusted R².\n")
}


# Predictions on test data
rf_test_preds = predict(rf_model, newdata = testf)
write.csv(rf_test_preds, file = "rf_submission.csv", row.names = FALSE)


dt_test_preds = predict(dt_model, newdata = testf)
write.csv(dt_test_preds, file = "dt_submission.csv", row.names = FALSE)
