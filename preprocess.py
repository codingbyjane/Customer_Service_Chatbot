# Import dependencies
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stopwords_set = set(stopwords.words('english'))

def preproces_text(text):
    text = text.lower() # Convert to lowercase
    tokens = word_tokenize(text) # Tokenize the text