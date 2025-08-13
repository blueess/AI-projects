# AI-projects
This repository contains a collection of AI and machine learning projects, including data analysis, model training, and evaluation. Each project is organized in its own folder.

## Project: Student Performance Prediction
This project analyzes factors affecting student exam scores using regression models. It demonstrates data preprocessing, outlier removal, feature engineering, and model comparison (linear and polynomial regression).

### Features

- Data cleaning and preprocessing
- Outlier detection and removal
- Exploratory data analysis with visualizations
- Regression modeling (Linear & Polynomial)
- Model evaluation with error metrics
  
### Files

- `student_per.ipynb`: Main Jupyter notebook for the student performance analysis
- Data files (CSV) used for training and testing models
- Supporting scripts and functions

### Dataset
The project uses the Student Performance Factors dataset from Kaggle.
Link: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

## Project: Forest Cover Type Prediction
This project uses machine learning techniques to predict forest cover types based on cartographic variables. The dataset includes features such as elevation, soil type, and wilderness area, and the goal is to classify the type of forest cover present.

### Features

- Data preprocessing and cleaning
- Exploratory data analysis and visualization
- Feature engineering
- Model training and evaluation (e.g., Random Forest, Decision Tree, etc.)
- Performance metrics and comparison

### Files

- `forest_model.ipynb`: Main Jupyter notebook containing the analysis, modeling, and results
- Data files (CSV) used for training and testing models
- Supporting scripts and utility functions

### Dataset
The project uses the Forest CoverType dataset from the UCI Machine Learning Repository.
Link: https://archive.ics.uci.edu/dataset/31/covertype

## Project: Loan Approval Prediction
This project uses machine learning to predict loan approval status based on applicant and financial features. It demonstrates data preprocessing, outlier removal, encoding, model training, and evaluation using both Logistic Regression and Decision Tree classifiers. SMOTE is used to address class imbalance.

### Features

- Data cleaning and preprocessing
- Outlier detection and removal
- Label encoding for categorical variables
- Feature scaling
- Model training (Logistic Regression, Decision Tree)
- Handling class imbalance with SMOTE
- Model evaluation with classification reports and accuracy metrics
- Overfitting checks

### Files

- `loan_pred_model.ipynb`: Main Jupyter notebook for the loan approval prediction workflow
- Data files (CSV) used for training and testing models
- Supporting scripts and utility functions

### Dataset
The project uses a loan approval dataset containing applicant information and loan status.
Link: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data

### Additional requirements for this project
-imblearn

Install it with:
```bash
pip install imblearn
```

## Requirements for any of the projects

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn
  
Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
