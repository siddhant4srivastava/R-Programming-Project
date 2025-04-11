library(caret)
library(pROC)
library(randomForest)
library(rpart)
library(rpart.plot)
library(e1071)
library(dplyr)
library(ggplot2)


# Load datasets
train_data = read.csv("D://PDFs//Edvancer Eduventures//Certified Business Analytics using R//Projects//Banking//bank-full_train.csv", stringsAsFactors = TRUE)
test_data = read.csv("D://PDFs//Edvancer Eduventures//Certified Business Analytics using R//Projects//Banking//bank-full_test.csv", stringsAsFactors = TRUE)

head(test_data)
head(train_data)

variable.names(train_data)

str(train_data)

train_data$y = as.factor(train_data$y)

# Set seed for reproducibility
set.seed(123)

# Split data 80-20
train_index = createDataPartition(train_data$y, p = 0.8, list = FALSE)
train_split = train_data[train_index, ]
test_split = train_data[-train_index, ]


log_model = glm(y ~ . -ID, data = train_split, family = "binomial")

# Predict probabilities
log_pred_prob = predict(log_model, newdata = test_split, type = "response")
log_pred = ifelse(log_pred_prob > 0.5, "yes", "no")
log_pred = as.factor(log_pred)

tree_model = rpart(y ~ . -ID, data = train_split, method = "class")
rpart.plot(tree_model)

tree_pred = predict(tree_model, newdata = test_split, type = "class")

rf_model = randomForest(y ~ . -ID, data = train_split, ntree = 100, importance = TRUE)

rf_pred = predict(rf_model, newdata = test_split)
rf_prob = predict(rf_model, newdata = test_split, type = "prob")[,2]


control = trainControl(
  method = "cv",             # Cross-validation
  number = 5,                # 5-fold
  classProbs = TRUE,         # Needed for ROC
  summaryFunction = twoClassSummary, # For using ROC as metric
  savePredictions = "final"
)


# Define grid
grid_tree = expand.grid(cp = seq(0.001, 
                                 0.05, 
                                 by = 0.005))

# Train model with tuning
set.seed(123)
tree_tuned = train(y ~ . -ID, data = train_split,
                    method = "rpart",
                    trControl = control,
                    tuneGrid = grid_tree,
                    metric = "ROC")

# Predictions
tree_tuned_pred = predict(tree_tuned, newdata = test_split)
tree_tuned_prob = predict(tree_tuned, newdata = test_split, type = "prob")[,2]

# View best parameters
print(tree_tuned$bestTune)

grid_rf = expand.grid(mtry = c(2, 4, 6, 8, 10))

# Train Random Forest
set.seed(123)
rf_tuned = train(
  y ~ . -ID,
  data = train_split,
  method = "rf",
  trControl = control,
  tuneGrid = grid_rf,
  ntree = 100,
  metric = "ROC"
)

rf_tuned_pred = predict(rf_tuned, newdata = test_split)
rf_tuned_prob = predict(rf_tuned, newdata = test_split, type = "prob")[,2]

# View best parameters
print(rf_tuned$bestTune)

get_metrics = function(true, pred, prob) {
  cm = confusionMatrix(pred, true, positive = "yes")
  roc_obj = roc(as.numeric(true) ~ prob)
  
  list(
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"],
    AUC = auc(roc_obj)
  )
}

# Calculate metrics
log_metrics = get_metrics(test_split$y, log_pred, log_pred_prob)
tree_metrics = get_metrics(test_split$y, tree_pred, as.numeric(tree_pred == "yes"))
rf_metrics = get_metrics(test_split$y, rf_pred, rf_prob)

# Show results
print("Logistic Regression:")
print(log_metrics)

print("Decision Tree:")
print(tree_metrics)

print("Random Forest:")
print(rf_metrics)

# Assuming Random Forest performed best
final_predictions = predict(rf_model, newdata = test_data)
submission = data.frame(ID = test_data$ID, y = final_predictions)

# Save submission
write.csv(submission, "siddhant_srivastava_P5_part2.csv", row.names = FALSE)

#Data Visualization

# Class distribution
ggplot(train_data, aes(x = y, fill = y)) +
  geom_bar(width = 0.5) +
  labs(title = "Class Distribution (Subscribed vs Not Subscribed)",
       x = "Subscription to Term Deposit",
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("#F8766D", "#00BFC4"))


# Get importance from trained RF
rf_imp = varImp(rf_tuned)
imp_df = data.frame(Feature = rownames(rf_imp$importance),
                     Importance = rf_imp$importance$Overall)

# Plot top 15 features
imp_df = imp_df %>% arrange(desc(Importance)) %>% head(15)

ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#00BFC4") +
  coord_flip() +
  labs(title = "Top 15 Important Features (Random Forest)",
       x = "Features", y = "Importance") +
  theme_minimal()

# Compute ROC curves

roc_log = roc(test_split$y, log_pred_prob)
roc_tree = roc(test_split$y, tree_tuned_prob)
roc_rf = roc(test_split$y, rf_tuned_prob)

# Plot ROC
plot(roc_log, col = "blue", legacy.axes = TRUE, print.auc = TRUE,
     main = "ROC Curve Comparison", lwd = 2)
plot(roc_tree, col = "red", add = TRUE, print.auc = TRUE, print.auc.y = 0.4)
plot(roc_rf, col = "green", add = TRUE, print.auc = TRUE, print.auc.y = 0.3)
legend("bottomright", legend = c("Logistic", "Decision Tree", "Random Forest"),
       col = c("blue", "red", "green"), lwd = 2)



mean_age = mean(train_data$age)
round(mean_age, 2)

# Get Q1 and Q3
q1 = quantile(train_data$balance, 0.25)
q3 = quantile(train_data$balance, 0.75)
iqr = q3 - q1

# Define outlier thresholds
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

# Count of outliers
outlier_count <- sum(train_data$balance < lower_limit | train_data$balance > upper_limit)

# Print result
outlier_count

variance_balance = var(train_data$balance)
variance_balance

library(car)

# Assuming a model
model = lm(y ~ ., data = train_data)

# Check multicollinearity
vif(model)

remove_high_vif = function(data, threshold = 5) {
  data = data[, sapply(data, is.numeric)]  # Only numeric vars
  model = lm(as.formula(paste("balance ~", paste(names(data)[-1], collapse = "+"))), data = data)
  vif_values = vif(model)
  return(vif_values[vif_values > threshold])
}