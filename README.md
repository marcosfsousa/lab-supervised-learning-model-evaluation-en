![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Supervised Learning Model Evaluation

## Goal

Practice calculating and interpreting the model evaluation metrics used for both regression and classification problems: fit a model on each type of task, generate predictions, and compare performance on training vs. testing data.

## Datasets

- **Boston Housing** (`housing.csv`, 506 rows, 13 features) — regression task, predicting `MEDV`, the median value of owner-occupied homes (in $1000s), from features like crime rate, rooms per dwelling, and pupil-teacher ratio.
- **Iris** (`sklearn.datasets.load_iris`, 150 rows, 4 features) — classification task, predicting one of 3 flower species (setosa, versicolor, virginica) from sepal/petal length and width.

## What Was Done

### Regression (Boston Housing)
1. Loaded `housing.csv` and split it into an 80/20 train/test set (`random_state=42`).
2. Trained a `LinearRegression` model on the training features and generated predictions on both training and testing sets.
3. Calculated and compared **R²**, **MSE/RMSE**, and **MAE** between the train and test predictions.

### Classification (Iris)
1. Loaded the Iris dataset from scikit-learn and split it 80/20, using `stratify=y` to keep class proportions balanced across the split.
2. Trained a `LogisticRegression` model and generated predictions on both training and testing sets.
3. Calculated and compared **accuracy**, **balanced accuracy**, **precision**, **recall**, and **F1 score** (all weighted-averaged across the 3 classes) between train and test.
4. Generated **confusion matrices** for both sets to see exactly which classes were being confused.

## Results

**Regression (Linear Regression on Boston Housing):**

| Metric | Train | Test |
|--------|-------|------|
| R²     | 0.751 | 0.669 |
| MSE    | 21.64 | 24.29 |
| MAE    | 3.31  | 3.19  |

**Classification (Logistic Regression on Iris):**

| Metric            | Train  | Test   |
|-------------------|--------|--------|
| Accuracy          | 0.975  | 0.967  |
| Balanced Accuracy | 0.975  | 0.967  |
| Precision         | 0.975  | 0.970  |
| Recall            | 0.975  | 0.967  |
| F1                | 0.975  | 0.967  |

Test confusion matrix: only 1 misclassification out of 30 (a versicolor predicted as virginica); the training confusion matrix shows 3 misclassifications out of 120, all between versicolor and virginica — the two classes that are not linearly separable from each other.

## Key Learnings

- Comparing a metric on train **vs.** test is what actually tells you something about generalization — a single number in isolation doesn't reveal overfitting or underfitting. Here, the regression model's R² drop (0.751 → 0.669) is a mild sign of overfitting, while the classification metrics stay close together, indicating a good, generalizable fit.
- Different regression metrics answer different questions: R² tells you the proportion of variance explained, MSE penalizes large errors more heavily (and its square root, RMSE, converts the error back into the target's original units), and MAE gives a simple average error.
- For multiclass classification, metrics like precision/recall/F1 need an averaging strategy (`average='weighted'` here) to collapse per-class scores into one number — weighting by class support to reflect real class balance.
- `stratify=y` in `train_test_split` matters most for small or imbalanced classification datasets: it keeps each class proportionally represented in both splits, avoiding a train or test set that's accidentally skewed.
- Confusion matrices reveal *which* classes get mixed up, not just how often — here, all Iris misclassifications happen between versicolor and virginica, consistent with those two species being harder to separate than setosa.
- High and closely-matched train/test scores (as seen on Iris) are a good sign, but they also depend on the dataset being well-balanced and relatively easy to separate — the same model/metric pairing won't always produce results this clean on messier real-world data.

## Deliverables

- Completed `your-code/main.ipynb` notebook with all exercise cells filled in and executed.

## Submission

Upon completion, add your deliverables to git. Then commit git and push your branch to the remote.

## Resources

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)
- [Model evaluation: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
