# Import dependencies
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import string


# Load the Customer Service on Twitter dataset
queries_df = pd.read_csv("data/raw/twcs.csv", usecols=['author_id', 'inbound', 'text'], nrows=100000) # Load the first 100,000 for the CSV file and filter the necessary columns

# Filter the dataframe to incude inbound queries only
queries_df = queries_df[queries_df['inbound'] == True].copy() # Separate the dataframe containing inbound queries by creating an independent copy of the original dataframe
print(f"Total number of inbound queries: {len(queries_df)}")
print(queries_df.head())

# Remove empty rows from the 'text' column and create a separate copy of the dataframe
queries_df = queries_df.dropna(subset=['text']).copy()

# Collapse whitespaces, tabs, newlines and remove leading/trailing spaces by replacing them with a single space
queries_df['text'] = queries_df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove empty rows and whitespace-only rows from the 'text' column 
queries_df = queries_df[queries_df['text'] != '']

# Remove duplicate rows based on the 'text' column & reset indexes
queries_df = queries_df.drop_duplicates(subset=['text']).reset_index(drop=True)

nltk.download('punkt')

def preproces_text(text):

    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Convert to lowercase
