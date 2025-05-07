# E-Commerce Linear Regression Analysis

This project provides a comprehensive analysis of an e-commerce customer dataset using linear regression techniques. It includes exploratory data analysis (EDA), feature selection, model building (both simple and multiple linear regression), residual diagnostics, and customer segmentation using K-means clustering.
## Live Report

Access the rendered HTML report here:  
[View the full analysis report](https://kushagrashukla30.github.io/EcomRegression2/)


## Summary of Analysis

- **Dataset**: Contains details of customer behavior and spending patterns.
- **Exploratory Data Analysis**:
  - Visual and statistical summary of key features
  - Correlation heatmap and distribution plots
- **Simple Linear Regression**:
  - Predicted customer spending using single features like `Time_on_App` and `Time_on_Website`
- **Multiple Linear Regression**:
  - Included multiple predictors to improve accuracy
  - Tested for multicollinearity and removed insignificant variables
- **Model Diagnostics**:
  - Evaluated model fit with residual plots, Q-Q plots, and summary statistics
- **Clustering**:
  - Applied K-Means clustering to segment customers based on behavior
  - Visualized clusters and analyzed cluster characteristics

## Tools & Libraries Used

- R
- Quarto
- ggplot2
- dplyr
- tidyr
- stats
- cluster
- factoextra
- broom

## Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/KushagraShukla30/EcomRegression2.git
   cd EcomRegression2
    ````

2. Open the `ecom_linear_regression_analysis.qmd` file in RStudio or Quarto.

3. Install required R packages:

   ```r
   install.packages(c("ggplot2", "dplyr", "tidyr", "cluster", "factoextra", "broom"))
   ```

4. Render the document to HTML:

   ```r
   quarto::quarto_render("ecom_linear_regression_analysis.qmd")
   ```

