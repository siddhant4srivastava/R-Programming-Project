library(tidyverse)
library(dplyr)
library(tidymodels)

hr_test = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Human Resource\\hr_test.csv", stringsAsFactors = F)
hr_train = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Human Resource\\hr_train.csv", stringsAsFactors = F)

glimpse(hr_test)
glimpse(hr_train)

hr_train$left = as.factor(hr_train$left)
hr_train$sales = as.factor(hr_train$sales)
hr_train$salary = as.factor(hr_train$salary)
hr_test$sales = as.factor(hr_test$sales)
hr_test$salary = as.factor(hr_test$salary)


# Load necessary libraries

library(pROC)
library(caret)
library(randomForest)

# Split hr_train into training and validation sets (e.g., 80% train, 20% validation)

set.seed(123)  # For reproducibility
train_index = createDataPartition(hr_train$left, p = 0.8, list = FALSE)
train_data = hr_train[train_index, ]
validation_data = hr_train[-train_index, ]

# Logistic Regression Model

logistic_model = glm(left ~ ., data = train_data, family = binomial)

# Predict probabilities on validation set for logistic regression

logistic_pred = predict(logistic_model, validation_data, type = "response")
logistic_pred

# Calculate ROC-AUC for logistic regression

roc_obj_log = roc(validation_data$left, logistic_pred)
auc_value_log = auc(roc_obj_log)
print(paste("Logistic Regression ROC-AUC:", auc_value_log))

# Random Forest Model

rf_model = randomForest(left ~ ., data = train_data, ntree = 100, mtry = 3, importance = TRUE)

# Predict probabilities on validation set for Random Forest

rf_pred_prob = predict(rf_model, validation_data, type = "prob")[, 2]  # Probability for 'left' class (1)
rf_pred_prob

# Calculate ROC-AUC for Random Forest

roc_obj_rf = roc(validation_data$left, rf_pred_prob)
auc_value_rf = auc(roc_obj_rf)
print(paste("Random Forest ROC-AUC:", auc_value_rf))

# Find optimal cutoff scores using Youden's J statistic and F1 score for Random Forest

cutoffs = seq(0.1, 0.9, by = 0.01)  # Range of cutoff thresholds

# Youden's J statistic for RF

youden_index_rf = sapply(cutoffs, function(cutoff) {
  predicted_classes = ifelse(rf_pred_prob > cutoff, 1, 0)
  cm = confusionMatrix(factor(predicted_classes), factor(validation_data$left))
  sensitivity = cm$byClass["Sensitivity"]
  specificity = cm$byClass["Specificity"]
  sensitivity + specificity - 1  # Youden's J statistic
})

optimal_cutoff_rf = cutoffs[which.max(youden_index_rf)]
print(paste("Optimal Cutoff for Random Forest (Youden's J):", optimal_cutoff_rf))

# F1 Score for RF

f1_scores_rf = sapply(cutoffs, function(cutoff) {
  predicted_classes = ifelse(rf_pred_prob > cutoff, 1, 0)
  cm = confusionMatrix(factor(predicted_classes), factor(validation_data$left))
  precision = cm$byClass["Pos Pred Value"]
  recall = cm$byClass["Sensitivity"]
  if (is.na(precision) || is.na(recall)) return(NA)  # Handle cases with no positives
  (2 * precision * recall) / (precision + recall)  # F1 score
})

optimal_f1_cutoff_rf = cutoffs[which.max(f1_scores_rf)]
print(paste("Optimal Cutoff for Random Forest (F1 Score):", optimal_f1_cutoff_rf))

# Predictions on Test Data

rf_prob_scores = predict(rf_model, hr_test, type = "prob")[, 2]  # Probability of 'left' class
rf_prob_scores

# Save only the 'Probability_Left' column to a CSV file

write.csv(data.frame(Probability_Left = rf_prob_scores),
          "predicted_probabilities_rf.csv", row.names = FALSE)

# Print summary of optimal cutoffs for Random Forest

print(paste("Best Cutoff based on Youden's J:", optimal_cutoff_rf))
print(paste("Best Cutoff based on F1 Score:", optimal_f1_cutoff_rf))














