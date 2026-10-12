import pandas as pd
import re

# Load the Customer Service on Twitter dataset
queries_df = pd.read_csv("data/raw/twcs.csv", usecols=['author_id', 'inbound', 'text'], nrows=100000) # Load the first 100,000 for the CSV file and filter the necessary columns

# Filter the dataframe to incude inbound queries only
queries_df = queries_df[queries_df['inbound'] == True].copy() # Separate the dataframe containing inbound queries by creating an independent copy of the original dataframe
print(f"Total number of inbound queries: {len(queries_df)}")

# Remove empty rows from the 'text' column and create a separate copy of the dataframe
queries_df = queries_df.dropna(subset=['text']).copy()

# Collapse whitespaces, tabs, newlines and remove leading/trailing spaces by replacing them with a single space
queries_df['text'] = queries_df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove empty rows and whitespace-only rows from the 'text' column 
queries_df = queries_df[queries_df['text'] != '']

# Remove duplicate rows based on the 'text' column & reset indexes
queries_df = queries_df.drop_duplicates(subset=['text', 'author_id']).reset_index(drop=True) # By filtering 'text' and 'author_id' only the queryduplicates from the same authors will be removed
print(f"Total number of unique inbound queries: {len(queries_df)}")

def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', 'HTTPURL', text, flags=re.IGNORECASE)  # Replace URLs with a "HTTPURL" placeholder
    text = re.sub(r'@\w+', "@SUPPORT", text, flags=re.IGNORECASE)  # Replace mentions with a "@SUPPORT" placeholder
    text = re.sub(r'#', "", text)  # Remove hashtag symbols

    return text

queries_df['cleaned_text'] = queries_df['text'].map(preprocess_text)  # Apply the preprocessing function to the 'text' column and save the results in a new column 'cleaned_text'
print(queries_df[['text', 'cleaned_text']].head(20))

# Create a new dataframe with only the necessary columns for further processing
cleaned_queries_df = queries_df[['author_id', 'inbound', 'cleaned_text']].copy()

# Save the preprocessed dataframe to a CSV file (don't save the pandas indexes)
cleaned_queries_df.to_csv("data/processed/cleaned_twcs.csv", index=False)