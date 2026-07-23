import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required NLTK resources are downloaded silently
def download_nltk_deps():
    for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

download_nltk_deps()

# Initialize NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()


def clean_text(text: str, use_stemming: bool = True, use_lemmatization: bool = False) -> str:
    """
    Complete NLP Preprocessing function following standard pipeline:
    1. Lowercase text
    2. Remove URLs
    3. Remove HTML tags / special characters
    4. Remove numbers & punctuation
    5. Tokenization
    6. Stop-word removal
    7. Stemming (PorterStemmer) and/or Lemmatization
    8. Rejoin tokens into clean text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 6. Filter stopwords and short tokens, apply Stemming / Lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            word = token
            if use_stemming:
                word = stemmer.stem(word)
            if use_lemmatization:
                word = lemmatizer.lemmatize(word)
            cleaned_tokens.append(word)

    # 7. Join cleaned tokens back into text string
    return " ".join(cleaned_tokens)


def preprocess_dataframe(df, text_column='text', title_column='title'):
    """
    Full DataFrame preprocessing pipeline:
    - Drop null values
    - Drop duplicate records
    - Combine title and text for richer NLP features
    - Apply text cleaning
    """
    # Remove null values
    df = df.dropna(subset=[text_column]).copy()
    
    if title_column and title_column in df.columns:
        df[title_column] = df[title_column].fillna('')
        df['combined_text'] = df[title_column] + " " + df[text_column]
    else:
        df['combined_text'] = df[text_column]

    # Remove duplicate records based on combined text
    df = df.drop_duplicates(subset=['combined_text']).reset_index(drop=True)

    # Apply text cleaning
    df['clean_text'] = df['combined_text'].apply(clean_text)

    # Filter out empty clean texts after processing
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

    return df


if __name__ == '__main__':
    sample = "BREAKING: The government announces 100% free electricity for all citizens at http://example.com!"
    print("Original Text:", sample)
    print("Cleaned Text :", clean_text(sample))
