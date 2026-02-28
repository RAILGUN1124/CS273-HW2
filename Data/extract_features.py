"""
Feature extraction script for restaurant review classification (Fake vs Real).
Extracts 10 linguistic features from each review in data.csv and saves
the results to reviewFeatures.csv.
"""

import pandas as pd
import numpy as np
import string
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from spellchecker import SpellChecker
from spacy.matcher import Matcher
from spacy.symbols import nsubj, nsubjpass, dobj, conj, pobj, prep, amod, NOUN, VERB, ADJ, ADV

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Load spaCy model
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# Increase max length to handle potentially long reviews safely
nlp.max_length = 1_500_000

# ---------------------------------------------------------------------------
# Initialise spell-checker with domain-specific vocabulary
# ---------------------------------------------------------------------------
spell = SpellChecker()

# Indian-food / restaurant domain words that should NOT be counted as typos
DOMAIN_WORDS = {
    # Indian dishes & ingredients
    "tikka", "masala", "naan", "biryani", "biriyani", "biriyanis", "paneer",
    "samosa", "samosas", "korma", "vindaloo", "tandoori", "roti", "paratha",
    "chutney", "lassi", "dal", "dhal", "palak", "pakoda", "pakora",
    "jalfrezi", "karahi", "bhartiya", "aloo", "gobi", "sabzi",
    "kebab", "kebabs", "kabob", "kabobs", "murgh", "mahkani", "makhani",
    "gulab", "jamun", "kulche", "kheer", "biryani",
    "halal", "naan", "chai", "raita", "pappadum", "papadum", "papadums",
    "shahi", "amritsari", "hyderabadi", "punjabi",
    # Restaurant names & places
    "tikka", "shack", "lubbock", "chipotle", "doordash", "walmart",
    "tarka", "pal", "nikunj", "arugula", "chik",
    # Common informal / internet words
    "wifi", "lol", "lolz", "ok", "wouldnt", "didnt", "dont", "wasnt",
    "isnt", "couldnt", "shouldnt", "cant", "wont", "ive", "im", "thats",
    "hes", "shes", "theyre", "youll", "youre", "weve", "theyd",
    "youd", "mustnt", "hadnt", "hasnt", "havent", "arent", "werent",
    "gonna", "gotta", "wanna", "kinda", "sorta", "whatnot",
    "appetizing", "flavorful", "uncommon", "overpriced",
    "americanized", "microwaveable",
    # Food-related English words often flagged
    "entree", "entrée", "decor", "creamy", "saucy",
    "busboys", "waitress", "waiter",
    "hella", "yummy", "delish",
    # Misc terms from reviews
    "cia", "bc", "tx",
}
spell.word_frequency.load_words(DOMAIN_WORDS)

# NLTK stopwords set (used for Content Diversity)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Passive-voice matcher using spaCy's rule-based Matcher
# ---------------------------------------------------------------------------
matcher = Matcher(nlp.vocab)

# Pattern: auxiliary "be" verb + past participle (VBN)
# e.g. "was eaten", "were given", "is made", "been told"
passive_pattern = [
    {"DEP": "auxpass"},                     # auxiliary in passive
    {"DEP": {"IN": ["agent", "ROOT", "attr", "advcl", "relcl", "ccomp",
                     "xcomp", "conj", "pcomp", "acl"]},
     "TAG": {"IN": ["VBN", "VBD"]},         # past participle / past tense
     "OP": "?"},
]

# Simpler but effective: just look for nsubjpass dependency anywhere
# We will count passive constructions via dependency parse instead of matcher
# for higher accuracy.

# ---------------------------------------------------------------------------
# Feature extraction functions
# ---------------------------------------------------------------------------

def average_word_length(tokens):
    """AWL – Average Word Length (alphabetic tokens only)."""
    words = [t for t in tokens if t.isalpha()]
    if not words:
        return 0.0
    return np.mean([len(w) for w in words])


def average_sentence_length(sentences, tokens):
    """ASL – Average Sentence Length (words per sentence)."""
    words = [t for t in tokens if t.isalpha()]
    if not sentences:
        return 0.0
    return len(words) / len(sentences)


def number_of_words(tokens):
    """NWO – Number of Words (alphabetic tokens)."""
    return len([t for t in tokens if t.isalpha()])


def number_of_verbs(doc):
    """NVB – Number of Verbs (spaCy POS)."""
    return sum(1 for token in doc if token.pos_ == "VERB")


def number_of_adjectives(doc):
    """NAJ – Number of Adjectives (spaCy POS)."""
    return sum(1 for token in doc if token.pos_ == "ADJ")


def number_of_passive_voice(doc):
    """NPV – Number of Passive Voice Constructions.
    We count the number of tokens whose dependency label is 'nsubjpass',
    which marks the subject of a passive verb.  Each such occurrence
    corresponds to one passive-voice construction.
    """
    return sum(1 for token in doc if token.dep_ == "nsubjpass")


def number_of_sentences(sentences):
    """NST – Number of Sentences."""
    return len(sentences)


def content_diversity(tokens):
    """CDV – Content Diversity = |Vocabulary| / #tokens.
    Ignore punctuation and stopwords.
    """
    filtered = [t.lower() for t in tokens
                if t.isalpha() and t.lower() not in STOP_WORDS]
    if not filtered:
        return 0.0
    vocab = set(filtered)
    return len(vocab) / len(filtered)


def count_typos(tokens):
    """NTP – Number of Typos.
    Uses pyspellchecker on alphabetic tokens.
    Lowercases before checking; skips very short words (<=2 chars).
    """
    words = [t.lower() for t in tokens if t.isalpha() and len(t) > 2]
    misspelled = spell.unknown(words)
    return len(misspelled)


def typo_ratio(n_typos, tokens):
    """TPR – Typo Ratio = #typos / #words."""
    words = [t for t in tokens if t.isalpha()]
    if not words:
        return 0.0
    return n_typos / len(words)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Normalise whitespace: collapse newlines and multiple spaces into one."""
    import re
    text = str(raw)
    text = re.sub(r"\s+", " ", text)   # collapse all whitespace (incl. \n) → single space
    text = text.strip()
    return text


def extract_features(review_text):
    """Return a dict of all 10 features for a single review."""
    text = _clean_text(review_text)

    # Tokenise with NLTK
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Parse with spaCy
    doc = nlp(text)

    # Compute features
    awl = average_word_length(tokens)
    asl = average_sentence_length(sentences, tokens)
    nwo = number_of_words(tokens)
    nvb = number_of_verbs(doc)
    naj = number_of_adjectives(doc)
    npv = number_of_passive_voice(doc)
    nst = number_of_sentences(sentences)
    cdv = content_diversity(tokens)
    ntp = count_typos(tokens)
    tpr = typo_ratio(ntp, tokens)

    return {
        "AWL": round(awl, 4),
        "ASL": round(asl, 4),
        "NWO": nwo,
        "NVB": nvb,
        "NAJ": naj,
        "NPV": npv,
        "NST": nst,
        "CDV": round(cdv, 4),
        "NTP": ntp,
        "TPR": round(tpr, 4),
    }


def main():
    # ---- Load data ---------------------------------------------------------
    df = pd.read_csv("data.csv")
    print(f"Loaded {len(df)} reviews from data.csv")

    # ---- Extract features for every review ---------------------------------
    feature_rows = []
    for idx, row in df.iterrows():
        review = row["Review"]
        feats = extract_features(review)
        feats["Real=1/Fake=0"] = row["Real=1/Fake=0"]
        feature_rows.append(feats)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processed {idx + 1}/{len(df)} reviews ...")

    # ---- Build output DataFrame (10 features + label) ----------------------
    feature_df = pd.DataFrame(feature_rows)

    # Ensure column order: 10 features then label
    col_order = ["AWL", "ASL", "NWO", "NVB", "NAJ", "NPV", "NST", "CDV",
                 "NTP", "TPR", "Real=1/Fake=0"]
    feature_df = feature_df[col_order]

    # ---- Save to reviewFeatures.csv ----------------------------------------
    feature_df.to_csv("reviewFeatures.csv", index=False)
    print(f"\nSaved {len(feature_df)} rows to reviewFeatures.csv")
    print(feature_df.head())


if __name__ == "__main__":
    main()
