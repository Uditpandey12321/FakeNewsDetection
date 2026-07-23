import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from preprocess import preprocess_dataframe, clean_text
from setup_dataset import main as ensure_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
FAKE_CSV = os.path.join(DATASET_DIR, 'Fake.csv')
TRUE_CSV = os.path.join(DATASET_DIR, 'True.csv')


def load_and_merge_data():
    """
    Load Fake.csv and True.csv datasets.
    Merge both datasets and create target column:
    Fake = 0, True = 1
    """
    ensure_dataset()
    
    print("Loading dataset files...")
    df_fake = pd.read_csv(FAKE_CSV)
    df_true = pd.read_csv(TRUE_CSV)

    df_fake['label'] = 0  # 0 for Fake
    df_true['label'] = 1  # 1 for True / Real

    print(f"Loaded {len(df_fake)} Fake news records and {len(df_true)} Real news records.")

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("Running NLP text preprocessing...")
    df = preprocess_dataframe(df, text_column='text', title_column='title')
    print(f"Preprocessed dataset ready with {len(df)} total clean samples.")

    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train 3 classifiers and evaluate across metrics:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Random Forest
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    }

    results = []
    trained_models = {}

    print("\n" + "="*70)
    print(" STEP 4 & 5: MODEL TRAINING & EVALUATION ")
    print("="*70)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': auc,
            'Confusion Matrix': cm,
            'Classification Report': classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
        })

    results_df = pd.DataFrame(results)
    
    print("\n--- MODEL COMPARISON TABLE ---")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].to_string(index=False))

    return results_df, trained_models


def perform_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Perform GridSearchCV hyperparameter tuning on Logistic Regression and Random Forest
    """
    print("\n" + "="*70)
    print(" STEP 6: HYPERPARAMETER TUNING (GridSearchCV) ")
    print("="*70)

    # 1. Logistic Regression Tuning
    print("\nTuning Logistic Regression...")
    lr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    print(f"Best Logistic Regression Parameters: {grid_lr.best_params_}")
    print(f"Best Logistic Regression CV Accuracy: {grid_lr.best_score_:.4f}")

    # 2. Random Forest Tuning
    print("\nTuning Random Forest Classifier...")
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [15, 30, None],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    print(f"Best Random Forest Parameters: {grid_rf.best_params_}")
    print(f"Best Random Forest CV Accuracy: {grid_rf.best_score_:.4f}")

    # Determine top tuned model
    best_tuned_model = grid_lr.best_estimator_ if grid_lr.best_score_ >= grid_rf.best_score_ else grid_rf.best_estimator_
    best_tuned_name = 'Tuned Logistic Regression' if grid_lr.best_score_ >= grid_rf.best_score_ else 'Tuned Random Forest'

    y_pred = best_tuned_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTop Tuned Model: {best_tuned_name} with Test Accuracy: {test_acc*100:.2f}%")

    return best_tuned_model, best_tuned_name, test_acc


def save_artifacts(model, vectorizer):
    """
    Save best_model.pkl and tfidf_vectorizer.pkl using Joblib
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print("\n" + "="*70)
    print(" STEP 7: SAVE MODEL ARTIFACTS ")
    print("="*70)
    print(f"Saved Best Model to       : {model_path}")
    print(f"Saved TF-IDF Vectorizer to: {vec_path}")


def main():
    df = load_and_merge_data()

    print("\n" + "="*70)
    print(" STEP 3: FEATURE ENGINEERING (TF-IDF Vectorization) ")
    print("="*70)

    # Check minimum dataset size to set appropriate min_df
    num_samples = len(df)
    min_df_val = 5 if num_samples >= 100 else 1

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=min_df_val,
        max_df=0.8
    )

    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label'].values

    print(f"TF-IDF Matrix shape: {X.shape}")

    # Step 4: Train-test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} | Testing set size: {X_test.shape[0]}")

    results_df, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    tuned_model, tuned_name, tuned_acc = perform_hyperparameter_tuning(X_train, y_train, X_test, y_test)

    # Find overall best model
    best_row = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]
    best_untuned_name = best_row['Model']
    best_untuned_acc = best_row['Accuracy']

    if tuned_acc >= best_untuned_acc:
        final_best_model = tuned_model
        final_best_name = tuned_name
        final_acc = tuned_acc
    else:
        final_best_model = trained_models[best_untuned_name]
        final_best_name = best_untuned_name
        final_acc = best_untuned_acc

    print("\n" + "*"*70)
    print(f" BEST PERFORMING MODEL: {final_best_name} (Accuracy: {final_acc*100:.2f}%) ")
    print("*"*70)

    save_artifacts(final_best_model, vectorizer)


if __name__ == '__main__':
    main()
