# Load necessary libraries
library(pROC)
library(caret)
library(randomForest)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(ggplot2)
library(corrplot)
library(reshape2)

hr_test = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Human Resource\\hr_test.csv", stringsAsFactors = F)
hr_train = read.csv("D:\\PDFs\\Edvancer Eduventures\\Certified Business Analytics using R\\Projects\\Human Resource\\hr_train.csv", stringsAsFactors = F)

glimpse(hr_test)
glimpse(hr_train)

str(hr_train)

hr_train$left = as.factor(hr_train$left)
hr_train$sales = as.factor(hr_train$sales)
hr_train$salary = as.factor(hr_train$salary)
hr_test$sales = as.factor(hr_test$sales)
hr_test$salary = as.factor(hr_test$salary)


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


# 1. Attrition Distribution

ggplot(hr_train, aes(x = as.factor(left))) +
  geom_bar(fill = "tomato") +
  labs(title = "Employee Attrition Count", x = "Left (1 = Yes, 0 = No)", y = "Count")

# 2. Satisfaction vs Last Evaluation with Attrition

ggplot(hr_train, aes(x = satisfaction_level, y = last_evaluation, color = as.factor(left))) +
  geom_point(alpha = 0.5) +
  labs(title = "Satisfaction vs Evaluation Colored by Attrition", color = "Left")

# 3. Attrition by Department

ggplot(hr_train, aes(x = sales, fill = as.factor(left))) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Rate by Department", x = "Department", y = "Proportion", fill = "Left") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 4. Salary Level vs Attrition

ggplot(hr_train, aes(x = salary, fill = as.factor(left))) +
  geom_bar(position = "fill") +
  labs(title = "Attrition by Salary Level", x = "Salary Level", y = "Proportion", fill = "Left")

# 5. Correlation Heatmap for Numeric Variables

numeric_vars = hr_train %>%
  select_if(is.numeric)

cor_matrix = cor(numeric_vars)
corrplot(cor_matrix, method = "color", addCoef.col = "black", number.cex = 0.7, tl.cex = 0.8,
         tl.col = "black", type = "lower", title = "Correlation Heatmap", mar = c(0,0,2,0))

# 6. Boxplot: Satisfaction Level by Attrition

ggplot(hr_train, aes(x = as.factor(left), y = satisfaction_level, fill = as.factor(left))) +
  geom_boxplot() +
  labs(title = "Satisfaction Level by Attrition", x = "Left", y = "Satisfaction Level")











