# ---------------------------------------------------------------------------
# E-Commerce Linear Regression Analysis - Combined Version
# Author: Kushagra Shukla
# ---------------------------------------------------------------------------

# Install necessary packages if not already installed
if (!require("car")) install.packages("car")       # For VIF (Variance Inflation Factor)
if (!require("cluster")) install.packages("cluster") # For K-means clustering
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("caret")) install.packages("caret")
if (!require("corrplot")) install.packages("corrplot")
if (!require("glmnet")) install.packages("glmnet")
if (!require("MASS")) install.packages("MASS")
if (!require("leaps")) install.packages("leaps")

# Load all libraries
library(ggplot2)
library(caret)
library(corrplot)
library(glmnet)
library(MASS)
library(leaps)
library(car)       # Contains vif() function
library(cluster)   # For K-means clustering

# ---------------------------------------------------------------------------
# IMPORT DATA AND BASIC EXPLORATION
# ---------------------------------------------------------------------------
ecomdata <- read.csv("./data/ecomdata")  # Replace with your file path if needed

# Basic Exploration
str(ecomdata)
summary(ecomdata)

# ---------------------------------------------------------------------------
# VISUALIZATION AND CORRELATION ANALYSIS
# ---------------------------------------------------------------------------

# Scatter plots
ggplot(ecomdata, aes(x = Time.on.Website, y = Yearly.Amount.Spent)) + 
  geom_point(colour = "orange") + 
  ggtitle("Time on Website vs Yearly Amount Spent") + 
  xlab("Time on Website") +
  ylab("Yearly Amount Spent")

ggplot(ecomdata, aes(x = Avg..Session.Length, y = Yearly.Amount.Spent)) + 
  geom_point(colour = "orange") +
  ggtitle("Average Session Length vs Yearly Amount Spent") + 
  xlab("Average Session Length") +
  ylab("Yearly Amount Spent")

# Pairplot
pairs(ecomdata[c("Avg..Session.Length", "Time.on.App", "Time.on.Website", 
                 "Length.of.Membership", "Yearly.Amount.Spent")],
      col = "orange", pch = 16,
      labels = c("Avg Session Length", "Time on App", "Time on Website",
                 "Length of Membership", "Yearly Spent"),
      main = "Pairplot of Variables")

# Histogram and Boxplot of Length of Membership
ggplot(ecomdata, aes(x = Length.of.Membership)) + 
  geom_histogram(color = "white", fill = "orange", binwidth = 0.5)

ggplot(ecomdata, aes(x = Length.of.Membership)) + 
  geom_boxplot(fill = "orange")

# ---------------------------------------------------------------------------
# SIMPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------
lm.fit1 <- lm(Yearly.Amount.Spent ~ Length.of.Membership, data = ecomdata)
summary(lm.fit1)

# Plot regression line
plot(Yearly.Amount.Spent ~ Length.of.Membership, data = ecomdata)
abline(lm.fit1, col = "red")

# Residual diagnostics
qqnorm(residuals(lm.fit1)); qqline(residuals(lm.fit1), col = "red")
shapiro.test(residuals(lm.fit1))

# ---------------------------------------------------------------------------
# TRAIN-TEST SPLIT AND MODEL EVALUATION (SIMPLE LINEAR)
# ---------------------------------------------------------------------------
set.seed(1)
train_idx <- sample(1:nrow(ecomdata), 0.8 * nrow(ecomdata))
train <- ecomdata[train_idx, ]
test <- ecomdata[-train_idx, ]

lm.fit0.8 <- lm(Yearly.Amount.Spent ~ Length.of.Membership, data = train)
summary(lm.fit0.8)

prediction0.8 <- predict(lm.fit0.8, newdata = test)
err0.8 <- prediction0.8 - test$Yearly.Amount.Spent
rmse <- sqrt(mean(err0.8^2))
mape <- mean(abs(err0.8 / test$Yearly.Amount.Spent))

c(RMSE = rmse, MAPE = mape, R2 = summary(lm.fit0.8)$r.squared)

# ---------------------------------------------------------------------------
# MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------
multi.lm.fit <- lm(Yearly.Amount.Spent ~ Avg..Session.Length + 
                     Time.on.App + Time.on.Website + 
                     Length.of.Membership, data = ecomdata)
summary(multi.lm.fit)

# Multicollinearity Check (Variance Inflation Factor)
vif(multi.lm.fit)

# Train-test split for multiple regression
set.seed(1)
train_idx <- sample(1:nrow(ecomdata), 0.8 * nrow(ecomdata))
train <- ecomdata[train_idx, ]
test <- ecomdata[-train_idx, ]

multi.lm.fit0.8 <- lm(Yearly.Amount.Spent ~ Avg..Session.Length + 
                        Time.on.App + Time.on.Website + 
                        Length.of.Membership, data = train)
summary(multi.lm.fit0.8)

prediction.multi0.8 <- predict(multi.lm.fit0.8, newdata = test)
err.multi <- prediction.multi0.8 - test$Yearly.Amount.Spent
rmse.multi <- sqrt(mean(err.multi^2))
mape.multi <- mean(abs(err.multi / test$Yearly.Amount.Spent))

c(RMSE = rmse.multi, MAPE = mape.multi, R2 = summary(multi.lm.fit0.8)$r.squared)

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING AND ADVANCED CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
ecomdata$App_Web_Ratio <- ecomdata$Time.on.App / (ecomdata$Time.on.Website + 1)
ecomdata$Engagement_Score <- ecomdata$Avg..Session.Length * ecomdata$Length.of.Membership

# Correlation matrix visualization
num_data <- ecomdata[, sapply(ecomdata, is.numeric)]
cor_matrix <- cor(num_data, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

# ---------------------------------------------------------------------------
# CUSTOMER SEGMENTATION: K-MEANS CLUSTERING
# ---------------------------------------------------------------------------
set.seed(2)
kmeans_result <- kmeans(ecomdata[, c("Avg..Session.Length", "Time.on.App", 
                                     "Time.on.Website", "Length.of.Membership")], 
                        centers = 3, nstart = 20)

# Add cluster assignment to the data
ecomdata$Cluster <- as.factor(kmeans_result$cluster)

# Plot clusters
ggplot(ecomdata, aes(x = Avg..Session.Length, y = Time.on.App, color = Cluster)) + 
  geom_point() + 
  ggtitle("Customer Segmentation using K-means Clustering") +
  xlab("Avg. Session Length") + 
  ylab("Time on App")

# ---------------------------------------------------------------------------
# RESIDUAL ANALYSIS FOR MULTIPLE REGRESSION
# ---------------------------------------------------------------------------
# Plot residuals vs fitted values to check for homoscedasticity
plot(multi.lm.fit0.8, which = 1)

# QQ plot to check for normality of residuals
qqnorm(residuals(multi.lm.fit0.8))
qqline(residuals(multi.lm.fit0.8), col = "red")

# ---------------------------------------------------------------------------
# END OF ANALYSIS
# ---------------------------------------------------------------------------

