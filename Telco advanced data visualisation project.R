install.packages(c("caret", "randomForest"))
install.packages("treemap")
library(tidyverse)
library(caret)
library(randomForest)
library(ggdendro)
library(gridExtra)


#Load Telco_customer_churn_analysis  Dataset 
Telco_customer_churn_analysis<- read.csv("C:/Users/magst/Downloads/Advanced data visualisation project/WA_Fn-UseC_-Telco-Customer-Churn.csv")
view(Telco_customer_churn_analysis)


#Exploratory Data Analysis (EDA)
# Churn Rate Visualization
ggplot(Telco_customer_churn_analysis, aes(x = Churn, fill = Churn)) + 
  geom_bar() + 
#Data preprocessing of the Telco_customer_churn_analysis
#1. checking missing values  and removing them if any 
sum(is.na(Telco_customer_churn_analysis))
Telco_customer_churn_analysis <- na.omit(Telco_customer_churn_analysis)

#2. Convert Categorical Variables to Factors
Telco_customer_churn_analysis$Churn <- as.factor(Telco_customer_churn_analysis$Churn)
Telco_customer_churn_analysis$gender <- as.factor(Telco_customer_churn_analysis$gender)
Telco_customer_churn_analysis$Contract <- as.factor(Telco_customer_churn_analysis$Contract)

#3.Split Data into Training and Testing Sets
set.seed(123)
trainIndex <- createDataPartition(Telco_customer_churn_analysis$Churn, p = 0.8, list = FALSE)
train <- Telco_customer_churn_analysis[trainIndex, ]
test <- Telco_customer_churn_analysis[-trainIndex, ]
  ggtitle("Churn Distribution")

# Churn vs Contract Type
ggplot(Telco_customer_churn_analysis, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  geom_text(stat = "count", aes(label = scales::percent(..count../sum(..count..), accuracy = 1)), 
            position = position_fill(vjust = 0.5)) +
  ggtitle("Churn Rate by Contract Type") +
  ylab("Proportion of Customers") +
  xlab("Contract Type") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "red"))


#Monthly Charges vs. Total Charges (Scatter Plot)
ggplot(Telco_customer_churn_analysis, aes(x = MonthlyCharges, y = TotalCharges, color = Churn)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "black") +
  ggtitle("Monthly Charges vs Total Charges") +
  xlab("Monthly Charges (USD)") +
  ylab("Total Charges (USD)") +
  theme_minimal() +
  scale_color_manual(values = c("No" = "steelblue", "Yes" = "red"))


#Churn by Tenure (Histogram)
ggplot(Telco_customer_churn_analysis, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 5, alpha = 0.6, position = "identity", color = "black") +
  geom_density(aes(y = ..count.. * 5), alpha = 0.3) +  # Density curve scaled to match histogram
  ggtitle("Customer Churn by Tenure") +
  xlab("Tenure (Months)") +
  ylab("Count of Customers") +
  theme_minimal() +
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "red"))

#Dendrogram (Clustering Customers)
# Selecting relevant numeric variables
df_numeric <- head(Telco_customer_churn_analysis,25) %>% select(MonthlyCharges, tenure)

# Compute distance matrix
dist_matrix <- dist(df_numeric, method = "euclidean")

# Create hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D")

# Convert into dendrogram format
dendro_data <- as.dendrogram(hc)

# Plot dendrogram
ggdendrogram(dendro_data, rotate = TRUE, theme_dendro = FALSE) +
  ggtitle("Dendrogram of Customer Clusters")
install.packages("ggalluvial")
library(ggalluvial)
library(reshape2)

# Pie Chart - Churn Proportion
churn_counts <- Telco_customer_churn_analysis %>%
  count(Churn)

p1 <- ggplot(churn_counts, aes(x = "", y = n, fill = Churn)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  theme_void() +
  ggtitle("Customer Churn Distribution") +
  scale_fill_manual(values = c("Yes" = "red", "No" = "green"))

# Heatmap - Churn by Contract Type and Internet Service
heatmap_data <- Telco_customer_churn_analysis %>%
  count(Contract, InternetService, Churn) %>%
  spread(Churn, n, fill = 0)

heatmap_melted <- melt(heatmap_data, id.vars = c("Contract", "InternetService"))

p2 <- ggplot(heatmap_melted, aes(x = Contract, y = InternetService, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  ggtitle("Heatmap: Churn by Contract & Internet Service") +
  theme_minimal()

# Waterfall Chart - Cumulative Churn by Contract Type
contract_churn <- Telco_customer_churn_analysis %>%
  count(Contract, Churn) %>%
  spread(Churn, n, fill = 0) %>%
  mutate(Difference = Yes - No)

p3 <- ggplot(contract_churn, aes(x = Contract, y = Difference, fill = Difference > 0)) +
  geom_bar(stat = "identity") +
  ggtitle("Waterfall Chart: Churn by Contract Type") +
  scale_fill_manual(values = c("TRUE" = "red", "FALSE" = "green")) +
  theme_minimal()

# Arrange the plots
gridExtra::grid.arrange(p1, p2, p3, nrow = 2)


 
# 1.Logistic Regression
model_log <- glm(Churn ~ tenure +
                   MonthlyCharges +
                   TotalCharges, data = train, family = binomial)
# 2.Random Forest Model
model_rf <- randomForest(Churn ~ ., data = train, ntree = 100)

#Model Evaluation
#Make Predictions
pred_log <- predict(model_log, test, type = "response")
pred_rf <- predict(model_rf, test, type = "class")

#Convert Predictions to Binary
pred_log <- ifelse(pred_log > 0.5, "Yes", "No")

#Calculate Accuracy
conf_matrix_log <- confusionMatrix(as.factor(pred_log), test$Churn)
conf_matrix_rf <- confusionMatrix(pred_rf, test$Churn)

print(conf_matrix_log$overall["Accuracy"])
print(conf_matrix_rf$overall["Accuracy"])

#Step 7: Feature Importance (Random Forest)
importance(model_rf)
varImpPlot(model_rf)
#Step 8: Save Model for Future Use
saveRDS(model_rf, "churn_model.rds")

# Create the clustered bar plot  
ggplot(Telco_customer_churn_analysis, aes(x = factor(tenure), fill = Churn)) +  
  geom_bar(position = "dodge", alpha = 0.6, color = "black") +   
  ggtitle("Clustered Bar Plot of Customer Churn by Tenure") +  
  ylab("Count of Customers") +  
  xlab("Tenure (Months)") +  
  theme_minimal() +   
  scale_fill_manual(values = c("No" = "steelblue", "Yes" = "red")) +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))     
 
