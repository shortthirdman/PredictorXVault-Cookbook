# PredictorXVault-Cookbook

Master Machine Learning: A Collection of Classification &amp; Regression Algorithms in Action

> Machine Learning Techniques Explained: A Visual Guide with Code Examples

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)	![GitHub License](https://img.shields.io/github/license/shortthirdman/PredictorXVault-Cookbook?style=for-the-badge)	[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/shortthirdman/PredictorXVault-Cookbook/main)


![GitHub repo file or directory count](https://img.shields.io/github/directory-file-count/shortthirdman/PredictorXVault-Cookbook?type=file&extension=ipynb&style=for-the-badge&logo=jupyter&label=Jupyter%20Notebooks)


### **Machine Learning - Classification and Regression**

Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions without explicit programming. Two of the most fundamental types of ML problems are **classification** and **regression**. Both are supervised learning tasks, meaning they rely on labeled datasets to train models.

#### **Classification**
Classification is the task of predicting a discrete label or category for an input. The goal is to assign each instance of data to one of a predefined set of classes. Common classification algorithms include:

- **Logistic Regression**: Despite its name, it's a linear model used for binary classification.
- **Decision Trees**: A tree-like structure where internal nodes represent features, branches represent decision rules, and leaf nodes represent class labels.
- **Random Forest**: An ensemble of decision trees used to improve predictive accuracy by averaging results.
- **Support Vector Machines (SVM)**: Finds the hyperplane that best separates classes in a high-dimensional space.
- **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies data points based on the majority class of their nearest neighbors.
- **Neural Networks**: Inspired by the human brain, these models consist of layers of nodes that process data through activations.

**Applications**:
- Email spam detection
- Image recognition
- Disease diagnosis
- Sentiment analysis

#### **Regression**
Regression involves predicting a continuous value based on input data. Unlike classification, which outputs discrete labels, regression tasks aim to model relationships between variables and predict numerical outcomes. Common regression algorithms include:

- **Linear Regression**: The simplest form of regression, which models a linear relationship between the dependent and independent variables.
- **Ridge and Lasso Regression**: Variants of linear regression that incorporate regularization to prevent overfitting.
- **Decision Trees for Regression**: A regression tree that splits data based on feature values, but the prediction is a continuous value rather than a category.
- **Random Forest Regression**: An ensemble method that averages the results of multiple regression trees.
- **Support Vector Regression (SVR)**: Uses the principles of SVM to fit a regression model, aiming to find a function that approximates the data with a small margin of error.
- **Neural Networks for Regression**: Can model complex, non-linear relationships between input variables and output predictions.

**Applications**:
- Predicting house prices
- Stock market forecasting
- Sales prediction
- Energy consumption forecasting

---

**Key Differences**:
- **Output Type**: Classification predicts discrete labels, while regression predicts continuous values.
- **Algorithms**: Many algorithms overlap between classification and regression, such as decision trees and neural networks, but are adapted for the type of output they are trying to predict.

Both classification and regression play a vital role in real-world applications, and mastering these algorithms is essential for anyone working with machine learning to unlock predictive power from data.


## Development

  - Create a Python virtual environment and activate
	
	```shell
	python -m venv dev
	```
	
	```shell
	.\dev\Scripts\activate
	```

  - Install the packages and dependencies as listed in requirements file
	
	```shell
	pip install -U -r requirements.txt --no-cache-dir --disable-pip-version-check
	```

  - Start your development `Jupyter Notebook` or `Jupyter Lab` server
	
	```shell
	jupyter lab --notebook-dir=.\notebooks --no-browser
	```
	
	```shell
	jupyter notebook
	```

## Code Examples

* [Gradient Boosting Regressor](https://nbviewer.org/github/shortthirdman/PredictorXVault-Cookbook/blob/main/notebooks/regression/Gradient_Boosting_Regressor.ipynb)

* [Decision Tree Regressor](https://nbviewer.org/github/shortthirdman/PredictorXVault-Cookbook/blob/main/notebooks/regression/Decision_Tree_Regressor.ipynb)

* [Dummy Regressor](https://nbviewer.org/github/shortthirdman/PredictorXVault-Cookbook/blob/main/notebooks/regression/Dummy_Regressor.ipynb)

* [Decision Tree Classifier](https://nbviewer.org/github/shortthirdman/PredictorXVault-Cookbook/blob/main/notebooks/classification/Decision_Tree_Classifier.ipynb)

## References

- [Gradient Boosting Regressor - Towards Data Science](https://towardsdatascience.com/gradient-boosting-regressor-explained-a-visual-guide-with-code-examples-c098d1ae425c)

- [Decision Tree Regressor - Towards Data Science](https://towardsdatascience.com/decision-tree-regressor-explained-a-visual-guide-with-code-examples-fbd2836c3bef)

- [Dummy Regressor - Towards Data Science](https://towardsdatascience.com/dummy-regressor-explained-a-visual-guide-with-code-examples-for-beginners-4007c3d16629)

- [Decision Tree Classifier - Towards Data Science](https://towardsdatascience.com/decision-tree-classifier-explained-a-visual-guide-with-code-examples-for-beginners-7c863f06a71e)
