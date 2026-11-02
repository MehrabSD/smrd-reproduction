"""
Text Preprocessing Module for Rumor Detection.
Supports both:
1. Transformer-friendly cleaning (preserves sentence structure, handles, URLs handled, no stopword destruction)
2. Classical/paper cleaning (URL removal, optional lowercasing/tokenization/stopword removal)
"""

import re
import html

# Precompiled regex patterns
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
MENTION_REGEX = re.compile(r'@\w+')
MULTIPLE_SPACES = re.compile(r'\s+')

# Standard English stopwords (self-contained to avoid external nltk download requirements)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

def clean_tweet_for_transformer(text: str, remove_urls: bool = True) -> str:
    """
    Cleans tweet text while preserving grammatical and semantic structure for Transformers.
    Unescapes HTML entities, strips URLs/mentions according to parameters, and normalizes whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)
    
    # 2. Handle URLs
    if remove_urls:
        text = URL_REGEX.sub("", text)
    else:
        text = URL_REGEX.sub("[URL]", text)
        
    # 3. Normalize Twitter mentions
    text = MENTION_REGEX.sub("@user", text)
    
    # 4. Normalize whitespace
    text = MULTIPLE_SPACES.sub(" ", text).strip()
    return text

def clean_tweet_classical(text: str, remove_stopwords: bool = True) -> str:
    """
    Classical NLP preprocessing as described in the paper:
    URL removal, lowercase, alphanumeric/punctuation filtering, stopword removal.
    """
    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    text = URL_REGEX.sub("", text)
    text = MENTION_REGEX.sub("", text)
    text = text.lower()
    
    # Tokenize by word characters
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
        
    return " ".join(tokens)

