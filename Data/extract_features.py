import pandas as pd
import numpy as np
import string
import spacy
import nltk
from spellchecker import SpellChecker
from spacy.matcher.matcher import Matcher
from spacy.symbols import nsubj, nsubjpass, dobj, conj, pobj, prep, amod, NOUN, VERB, ADJ, ADV
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize spell checker
spell = SpellChecker()

# Load stopwords
stop_words = set(stopwords.words('english'))


def count_passive_voice(doc):
    """Count passive voice constructions using spaCy."""
    passive_count = 0
    for token in doc:
        # Check for passive voice: auxiliary verb + past participle
        if token.dep_ == "nsubjpass":
            passive_count += 1
    return passive_count


def extract_features(review_text):
    """Extract all 10 features from a review."""
    
    if pd.isna(review_text) or review_text.strip() == "":
        return [0] * 10
    
    # Process with spaCy
    doc = nlp(review_text)
    
    # NST – Number of Sentences
    sentences = list(doc.sents)
    nst = len(sentences)
    
    # Get words (tokens that are not punctuation or spaces)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    
    # NWO – Number of Words
    nwo = len(words)
    
    # AWL – Average Word Length
    if nwo > 0:
        awl = sum(len(token.text) for token in words) / nwo
    else:
        awl = 0
    
    # ASL – Average Sentence Length (words per sentence)
    if nst > 0:
        asl = nwo / nst
    else:
        asl = 0
    
    # NVB – Number of Verbs
    nvb = sum(1 for token in doc if token.pos_ == "VERB")
    
    # NAJ – Number of Adjectives
    naj = sum(1 for token in doc if token.pos_ == "ADJ")
    
    # NPV – Number of Passive Voice Constructions
    npv = count_passive_voice(doc)
    
    # CDV – Content Diversity = |V| / #tokens (ignore punctuation & stopwords)
    content_words = [token.text.lower() for token in words 
                     if token.text.lower() not in stop_words]
    if len(content_words) > 0:
        unique_content_words = len(set(content_words))
        cdv = unique_content_words / len(content_words)
    else:
        cdv = 0
    
    # NTP – Number of Typos
    # Get words without punctuation
    word_list = [token.text.lower() for token in words if token.text.isalpha()]
    misspelled = spell.unknown(word_list)
    ntp = len(misspelled)
    
    # TPR – Typo Ratio
    if len(word_list) > 0:
        tpr = ntp / len(word_list)
    else:
        tpr = 0
    
    return [awl, asl, nwo, nvb, naj, npv, nst, cdv, ntp, tpr]


def main():
    print("Loading data...")
    # Read the original data
    df = pd.read_csv('data.csv')
    
    print(f"Processing {len(df)} reviews...")
    
    # Extract features for each review
    features_list = []
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing review {idx}/{len(df)}...")
        
        features = extract_features(row['Review'])
        features_list.append(features)
    
    # Create features dataframe
    feature_columns = ['AWL', 'ASL', 'NWO', 'NVB', 'NAJ', 'NPV', 'NST', 'CDV', 'NTP', 'TPR']
    features_df = pd.DataFrame(features_list, columns=feature_columns)
    
    # Add the label column
    features_df['Label'] = df['Real=1/Fake=0']
    
    # Save to CSV
    output_file = 'reviewFeatures.csv'
    features_df.to_csv(output_file, index=False)
    
    print(f"\nFeature extraction complete!")
    print(f"Output saved to: {output_file}")
    print(f"\nFeature statistics:")
    print(features_df.describe())
    print(f"\nFirst few rows:")
    print(features_df.head())


if __name__ == "__main__":
    main()
