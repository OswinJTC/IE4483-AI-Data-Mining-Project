"""IE4483 sentiment analysis: four traditional machine-learning method families.

Run from the project root (the directory containing data/processed/):
    python 01_Traditional_ML_results/Traditional_ML_baselines.py

Compares four classifier families on the shared train/validation split:
    1. TF-IDF + Logistic Regression
    2. TF-IDF + Linear SVM
    3. TF-IDF + Multinomial Naive Bayes
    4. TF-IDF + Truncated SVD (LSA) + Logistic Regression

Also compares default vs tuned Naive Bayes using a small validation grid.

Output: confusion-matrix PNG for each configuration and one model_comparison.csv.
No individual validation prediction files or Naive Bayes tuning CSV are saved.

TF-IDF, SVD and the classifiers are fitted on the training split.
No test-set labels or scores are inferred.
"""
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')  # Works in terminals and notebooks without a display.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


DATA_DIR = Path('data/processed')
OUTPUT_DIR = Path('01_Traditional_ML_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the shared train and validation splits.
train = pd.read_json(DATA_DIR / 'train_split.json')
val = pd.read_json(DATA_DIR / 'val_split.json')
train_reviews = train['reviews'].fillna('').astype(str)
val_reviews = val['reviews'].fillna('').astype(str)
y_train = train['sentiments'].astype(int)
y_val = val['sentiments'].astype(int)
print(f'Train: {len(train):,} reviews | Validation: {len(val):,} reviews')
print('Class counts (training):', y_train.value_counts().sort_index().to_dict())

# Fit the shared TF-IDF vectorizer on the training reviews.
tfidf = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=20_000,
    sublinear_tf=True,
)
start = perf_counter()
X_train = tfidf.fit_transform(train_reviews)
tfidf_fit_seconds = perf_counter() - start
X_val = tfidf.transform(val_reviews)
print('TF-IDF train matrix:', X_train.shape)
print(f'TF-IDF fit/transform training time: {tfidf_fit_seconds:.3f} seconds')

# Tune Naive Bayes by validation Negative F1.
# Learned class priors can favor the majority class. The tuned model's validation
# score can be optimistic because the same split is used for selection.
nb_search_rows = []
for fit_prior in [True, False]:
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
        trial = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        trial.fit(X_train, y_train)
        trial_pred = trial.predict(X_val)
        nb_search_rows.append({
            'alpha': alpha,
            'fit_prior': fit_prior,
            'Accuracy': accuracy_score(y_val, trial_pred),
            'Negative F1': f1_score(y_val, trial_pred, pos_label=0, zero_division=0),
            'Macro F1': f1_score(y_val, trial_pred, average='macro', zero_division=0),
        })
nb_tuning = pd.DataFrame(nb_search_rows).sort_values('Negative F1', ascending=False)
nb_best = nb_tuning.iloc[0]
nb_best_alpha = float(nb_best['alpha'])
nb_best_fit_prior = bool(nb_best['fit_prior'])
print(f'NB best on validation: alpha={nb_best_alpha}, fit_prior={nb_best_fit_prior}')

# MultinomialNB uses TF-IDF directly because SVD can produce negative features.
models = {
    'Logistic Regression': LogisticRegression(
        C=10, class_weight='balanced', max_iter=2000, random_state=42,
    ),
    'Linear SVM': LinearSVC(
        C=10, class_weight='balanced', max_iter=5000, random_state=42,
    ),
    'Naive Bayes (default)': MultinomialNB(alpha=0.1),
    'Naive Bayes (tuned)': MultinomialNB(alpha=nb_best_alpha, fit_prior=nb_best_fit_prior),
    'SVD + Logistic Regression': make_pipeline(
        TruncatedSVD(n_components=200, random_state=42),
        Normalizer(),
        LogisticRegression(
            C=10, class_weight='balanced', max_iter=2000, random_state=42,
        ),
    ),
}

# Evaluate on the shared validation split.
results = []
for model_name, model in models.items():
    start = perf_counter()
    model.fit(X_train, y_train)
    model_fit_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict(X_val)
    prediction_seconds = perf_counter() - start

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_val, pred),
        'Negative F1': f1_score(y_val, pred, pos_label=0, zero_division=0),
        'Macro F1': f1_score(y_val, pred, average='macro', zero_division=0),
        'Fit time (s)': round(model_fit_seconds, 3),
        'Predict time (s)': round(prediction_seconds, 3),
    }
    results.append(metrics)
    filename = model_name.lower().replace(' ', '_').replace('+', 'plus')
    ConfusionMatrixDisplay.from_predictions(
        y_val, pred, labels=[0, 1], display_labels=['Negative', 'Positive'],
        cmap='Blues', values_format='d', colorbar=False,
    )
    plt.title(f'{model_name} — validation set')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f'Saved confusion matrix: {filename}_confusion_matrix.png')

# Save validation metrics sorted by Negative F1.
comparison = pd.DataFrame(results).sort_values('Negative F1', ascending=False)
comparison.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
print('\n=== All model configurations — validation comparison ===')
print(comparison.to_string(index=False, float_format=lambda n: f'{n:.4f}'))
print('\nFit times exclude the shared TF-IDF fit and NB hyperparameter search; SVD fit time includes SVD.')
print(f'Results saved to: {OUTPUT_DIR.resolve()}')
print('Saved files: one model_comparison.csv and five confusion-matrix PNGs.')
