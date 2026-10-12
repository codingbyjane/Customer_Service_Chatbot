# Import dependencies
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string


# Load the Customer Service on Twitter dataset
queries_df = pd.read_csv("data/raw/twcs.csv", usecols=['author_id', 'inbound', 'text'], nrows=100000) # Load the first 100,000 for the CSV file and filter the necessary columns
print(queries_df.head())








nltk.download('punkt')
nltk.download('stopwords')

stopwords_set = set(stopwords.words('english'))
stopwords_set.remove('not')  # Keep 'not' for sentiment analysis

def preproces_text(text):

    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Convert to lowercase
