import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_for_knowledge_base(text):
    text = text.lower()
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'[^a-z0-9\s.]', '', text)

    sentences = sent_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        processed_sentences.append(' '.join(lemmatized_words))
    return "\n".join(processed_sentences)
