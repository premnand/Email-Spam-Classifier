# Email Spam Classifier

## Project Overview
The Email Spam Classifier is a machine learning project aimed at identifying and classifying emails as either spam or ham (non-spam). The model uses various machine learning algorithms to achieve high accuracy in classifying emails based on their content.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Conclusion](#conclusion)
- [License](#license)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TfidfVectorizer (from Scikit-learn)
- Matplotlib (for visualization)

## Dataset
The dataset used for training and testing the classifier is the UCI Spambase dataset, which contains a collection of emails labeled as spam or ham. The dataset includes features derived from the email text, allowing for effective classification.

## Installation
To run this project, ensure you have Python and the required libraries installed. You can set up a virtual environment and install the necessary libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib
```
## Usage
- **Load the Dataset:** Load the dataset using pandas to create a DataFrame.
- **Data Preprocessing:** Clean the text data and transform it using TfidfVectorizer.
- **Split the Data:** Split the dataset into training and testing sets using train_test_split.
- **Train the Model:** Train the logistic regression model and/or Naive Bayes classifier using the training data.
- **Evaluate the Model:** Use cross-validation and performance metrics (precision, recall, f1-score) to evaluate model performance.
- **Predict New Emails:** The trained model can be used to classify new email inputs by preprocessing them in the same way as the training data.

## Model Evaluation
The model achieved the following average accuracies during cross-validation:

* **Logistic Regression:** 97.94%
* **Naive Bayes:** 98.36%
  
## Classification Report (Logistic Regression)
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       965
           1       0.98      0.85      0.91       150

    accuracy                           0.98      1115
   macro avg       0.98      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```
## Classification Report (Naive Bayes)
```
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       965
           1       0.99      0.89      0.94       150

    accuracy                           0.98      1115
   macro avg       0.98      0.95      0.96      1115
weighted avg       0.98      0.98      0.98      1115
```
## Hyperparameter Tuning

The following hyperparameters were tuned to improve model performance:

- **Logistic Regression:** ```[C: 10, penalty: 'l2']```
- **Naive Bayes:** ```[alpha: 0.1, force_alpha: True]```

## Model Performance After Tuning

- **Accuracy for Tuned Naive Bayes: 98.39%**
```
            precision    recall  f1-score   support

         0       0.98      1.00      0.99       965
         1       0.99      0.89      0.94       150

  accuracy                           0.98      1115
 macro avg       0.98      0.95      0.96      1115
weighted avg     0.98      0.98      0.98      1115
```

- **Accuracy for Tuned Logistic Regression: 97.67%**
```
          precision    recall  f1-score   support

       0       0.98      1.00      0.99       965
       1       0.98      0.85      0.91       150

  accuracy                         0.98      1115
 macro avg     0.98      0.92      0.95      1115 
weighted avg   0.98      0.98      0.98      1115
```

## Conclusion
The Email Spam Classifier demonstrates the ability to accurately classify emails as spam or ham using machine learning techniques. The project showcases effective data preprocessing, model training, and evaluation.

## License
This project is licensed under the MIT License. 
