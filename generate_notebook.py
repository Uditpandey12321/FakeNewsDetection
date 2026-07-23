import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📰 Fake News Detection System\n",
    "### End-to-End Natural Language Processing (NLP) & Machine Learning Pipeline\n",
    "\n",
    "This notebook demonstrates the complete process of building, evaluating, and tuning machine learning models to detect **Fake vs. Real News** articles using Natural Language Processing techniques."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries & Set Up Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import re\n",
    "import string\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from wordcloud import WordCloud\n",
    "from collections import Counter\n",
    "import joblib\n",
    "\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, precision_score, recall_score, f1_score,\n",
    "    roc_auc_score, confusion_matrix, classification_report, roc_curve\n",
    ")\n",
    "\n",
    "plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')\n",
    "print('All libraries imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load & Merge Dataset\n",
    "Target mapping:\n",
    "- `Fake = 0` (Fake News)\n",
    "- `True = 1` (Real / True News)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load datasets from dataset directory\n",
    "fake_df = pd.read_csv('../dataset/Fake.csv')\n",
    "true_df = pd.read_csv('../dataset/True.csv')\n",
    "\n",
    "fake_df['label'] = 0\n",
    "true_df['label'] = 1\n",
    "\n",
    "print(f'Fake News Samples: {len(fake_df)}')\n",
    "print(f'True News Samples: {len(true_df)}')\n",
    "\n",
    "# Merge and shuffle\n",
    "df = pd.concat([fake_df, true_df], ignore_index=True)\n",
    "df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "Clean text using NLTK: lowercasing, regex URL/special character removal, stop-word removal, and PorterStemmer stemming."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure NLTK resources\n",
    "nltk.download('punkt', quiet=True)\n",
    "nltk.download('stopwords', quiet=True)\n",
    "\n",
    "stemmer = PorterStemmer()\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def clean_text(text):\n",
    "    if not isinstance(text, str):\n",
    "        return ''\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub(r'<.*?>', '', text)\n",
    "    text = re.sub(r'[^a-zA-Z\\s]', '', text)\n",
    "    tokens = word_tokenize(text)\n",
    "    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]\n",
    "    return ' '.join(cleaned)\n",
    "\n",
    "# Remove nulls & duplicates\n",
    "df = df.dropna(subset=['text']).copy()\n",
    "df['title'] = df['title'].fillna('')\n",
    "df['combined_text'] = df['title'] + ' ' + df['text']\n",
    "df = df.drop_duplicates(subset=['combined_text']).reset_index(drop=True)\n",
    "\n",
    "df['clean_text'] = df['combined_text'].apply(clean_text)\n",
    "print('Preprocessing completed!')\n",
    "df[['title', 'label', 'clean_text']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Distribution of Fake vs Real News\n",
    "plt.figure(figsize=(7, 4))\n",
    "sns.countplot(x='label', data=df, palette=['#e74c3c', '#2ecc71'])\n",
    "plt.xticks([0, 1], ['Fake (0)', 'Real (1)'])\n",
    "plt.title('Distribution of News Class Labels', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Category')\n",
    "plt.ylabel('Count')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Word Clouds for Fake vs Real News\n",
    "fake_text = ' '.join(df[df['label'] == 0]['clean_text'])\n",
    "real_text = ' '.join(df[df['label'] == 1]['clean_text'])\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "\n",
    "wc_fake = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(fake_text)\n",
    "axes[0].imshow(wc_fake, interpolation='bilinear')\n",
    "axes[0].set_title('Word Cloud - Fake News', fontsize=14, fontweight='bold', color='crimson')\n",
    "axes[0].axis('off')\n",
    "\n",
    "wc_real = WordCloud(width=800, height=400, background_color='black', colormap='Greens').generate(real_text)\n",
    "axes[1].imshow(wc_real, interpolation='bilinear')\n",
    "axes[1].set_title('Word Cloud - Real News', fontsize=14, fontweight='bold', color='forestgreen')\n",
    "axes[1].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Top 20 Frequent Words in Dataset\n",
    "all_words = ' '.join(df['clean_text']).split()\n",
    "top_20 = Counter(all_words).most_common(20)\n",
    "top_df = pd.DataFrame(top_20, columns=['Word', 'Frequency'])\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x='Frequency', y='Word', data=top_df, palette='viridis')\n",
    "plt.title('Top 20 Most Frequent Cleaned Words', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Feature Engineering (TF-IDF Vectorization)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8)\n",
    "X = vectorizer.fit_transform(df['clean_text'])\n",
    "y = df['label'].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "print(f'Train shape: {X_train.shape} | Test shape: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Training & Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
    "    'Multinomial Naive Bayes': MultinomialNB(),\n",
    "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)\n",
    "}\n",
    "\n",
    "results = []\n",
    "trained_models = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    trained_models[name] = model\n",
    "    y_pred = model.predict(X_test)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    prec = precision_score(y_test, y_pred)\n",
    "    rec = recall_score(y_test, y_pred)\n",
    "    f1 = f1_score(y_test, y_pred)\n",
    "    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])\n",
    "    \n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Accuracy': acc,\n",
    "        'Precision': prec,\n",
    "        'Recall': rec,\n",
    "        'F1 Score': f1,\n",
    "        'ROC AUC': auc\n",
    "    })\n",
    "\n",
    "res_df = pd.DataFrame(results)\n",
    "display(res_df.style.highlight_max(axis=0, color='lightgreen'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Hyperparameter Tuning (GridSearchCV)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logistic Regression Tuning\n",
    "lr_grid = GridSearchCV(\n",
    "    LogisticRegression(max_iter=1000, random_state=42),\n",
    "    {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']},\n",
    "    cv=3, scoring='accuracy', n_jobs=-1\n",
    ")\n",
    "lr_grid.fit(X_train, y_train)\n",
    "print('Best LR Params:', lr_grid.best_params_)\n",
    "print('Best LR Accuracy:', lr_grid.best_score_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Save Trained Model & Vectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.makedirs('../models', exist_ok=True)\n",
    "best_model = lr_grid.best_estimator_\n",
    "joblib.dump(best_model, '../models/best_model.pkl')\n",
    "joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')\n",
    "print('Model and Vectorizer saved successfully!')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/fake_news_detection.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print('notebooks/fake_news_detection.ipynb generated successfully!')
