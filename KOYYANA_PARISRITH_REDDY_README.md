# Student Performance Analysis and Prediction

**Minor Project — Boosting**  
**Student:** KOYYANA PARISRITH REDDY

## Problem statement
Explore student exam-score patterns and compare regression models that estimate math scores from other available student-record features.

## Dataset
**Students Performance in Exams** — [Kaggle dataset page](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams). The commonly distributed dataset contains math, reading and writing scores and categorical fields. Verify the current dataset description and license on Kaggle before submitting.

## Objectives and workflow
1. Load the CSV and inspect rows, columns, data types and summary statistics.
2. Check missing values, duplicate rows and score ranges.
3. Remove exact duplicates and treat exam scores outside 0–100 as invalid.
4. Prepare features using a train/test split and a preprocessing pipeline.
5. Create at least four meaningful visualizations and observations.
6. Compare a mean baseline, Linear Regression and Ridge Regression.
7. Evaluate with MAE, RMSE and R²; explain the actual values from the notebook output.
8. Summarize findings, limitations and future improvements.

## How to run
Upload the `.ipynb` file to Google Colab or open it in Jupyter, then run all cells from top to bottom. If automatic download fails, download the CSV from Kaggle and upload `StudentsPerformance.csv` when prompted.

## Metrics
- **MAE:** average absolute prediction error in score points.
- **RMSE:** gives larger prediction errors more weight.
- **R²:** compares model fit with a mean-prediction baseline; it can be negative.

## Limitations
The dataset may not represent all schools or student populations and omits many influences on learning. Associations do not prove causation. Predictions should not be used to label or restrict students.

## Final preparation
After running the notebook, add two observations from your actual charts and the best model's actual MAE, RMSE and R² to your conclusion. Confirm the dataset's current license and be ready to explain why you chose regression and a train/test split.
