# Heart Disease Prediction using KNN Classifier

## Objective
This project employs the K-Nearest Neighbors (KNN) classification method to predict heart disease presence. It highlights the impact of feature scaling on the performance of the model and identifies the optimal number of neighbors (K value) for the best classification results.

## Project Overview
We implement the KNN classifier to distinguish patients with or without heart disease based on their medical measurements, emphasizing the model's sensitivity to the scale of input features and the importance of hyperparameter tuning.

## Requirements
To run this project, you will need:

- Python 3.x
- Pandas: For data handling and manipulation.
- Matplotlib & Seaborn: For visualizing data.
- Numpy: For numerical computations.
- Scikit-learn: For implementing machine learning algorithms.

## Data Preprocessing Steps
- **Data Cleaning**: Tidying up the dataset by addressing missing values and anomalies.
- **Exploratory Data Analysis (EDA)**: Utilizing statistics and visualization to understand the data deeply.
- **Train-Test Split**: Separating data into training and testing sets for model validation.
- **Feature Scaling**: Standardizing data to ensure each feature contributes appropriately to the outcome.

## Key Insights from Data Analysis
- **Feature Scaling**: Standardization of features significantly enhances model accuracy.
- **Hyperparameter Tuning**: Determining the optimal K value is crucial, with K=4 providing the best results for scaled features.

## Model Selection and Justification
- **KNN Classifier**: Chosen for its effectiveness in binary classification problems and its non-parametric nature.
- **Reasoning**: KNN's adaptability makes it suitable for complex datasets where the relationship between variables is not straightforward.

## Performance Metrics and Comparison
- **Accuracy (Scaled Data)**: Achieved a maximum of ~86.81% accuracy with K=4, indicating high predictive performance.
- **Accuracy (Unscaled Data)**: Reached up to ~69.24% accuracy with K=9, showing less predictive capability.
- **Error Rate (Scaled Data)**: Observed a minimum error rate of ~13.17%, which is considerably low.
- **Error Rate (Unscaled Data)**: Noted a minimum error rate of ~30.77%, which is higher than the scaled data approach.
- **Comparison**: The model with scaled data outperforms the unscaled version, highlighting the importance of preprocessing. The lower K value for the optimal result in scaled data suggests a more concise and generalizable model.

## Instructions for Use
1. Clone the repository.
2. Install required dependencies from `requirements.txt`.
3. Run the Jupyter notebook to train and evaluate the KNN classifier.
4. Experiment with different K values to observe variations in performance.

## Conclusion
Feature scaling proved to be a decisive factor in enhancing the KNN classifier's accuracy. The comprehensive analysis underscores the necessity of meticulous preprocessing and model tuning to achieve optimal predictive performance in machine learning tasks.
