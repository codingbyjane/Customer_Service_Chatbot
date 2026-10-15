import pandas as pd
import re

# Load the Customer Service on Twitter dataset
queries_df = pd.read_csv("data/raw/twcs.csv", usecols=['author_id', 'inbound', 'text', 'in_response_to_tweet_id'], nrows=500000) # Load the first 50,000 queries for the CSV file and filter the necessary columns

# Filter the dataframe to incude inbound queries and opening messages only
queries_df = queries_df[(queries_df['inbound'] == True) & (queries_df['in_response_to_tweet_id'].isna())].copy() # Separate the dataframe containing inbound queries and queries that aren't replies to previous tweets (such tweets have ambiguous meaning and lack context) by creating an independent copy of the original dataframe
print(f"Total number of inbound non-reply queries: {len(queries_df)}")

# Remove empty rows from the 'text' column and create a separate copy of the dataframe
queries_df = queries_df.dropna(subset=['text']).copy()

# Collapse whitespaces, tabs, newlines and remove leading/trailing spaces by replacing them with a single space
queries_df['text'] = queries_df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove empty rows and whitespace-only rows from the 'text' column 
queries_df = queries_df[queries_df['text'] != '']

# Remove duplicate rows based on the 'text' column & reset indexes
queries_df = queries_df.drop_duplicates(subset=['text', 'author_id']).reset_index(drop=True) # By filtering 'text' and 'author_id' only the query duplicates from the same authors will be removed
print(f"Total number of unique inbound queries: {len(queries_df)}")

def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text, flags=re.IGNORECASE)  # Replace URLs with a "http" placeholder (aligns with the model's training data)
    text = re.sub(r'@\w+', "@user", text, flags=re.IGNORECASE)  # Replace mentions with a "@user" placeholder (aligns with the model's training data)
    text = re.sub(r'#', "", text)  # Remove hashtag symbols

    return text

queries_df['cleaned_text'] = queries_df['text'].map(preprocess_text) # Map the preprocessing function to the 'text' column and save the results in a new column 'cleaned_text'
print(queries_df[['text', 'cleaned_text']].head(20))

# Create a new dataframe with only the necessary columns for further processing
cleaned_queries_df = queries_df[['author_id', 'cleaned_text']].copy()

# Save the preprocessed dataframe to a CSV file (don't save the pandas indexes)
cleaned_queries_df.to_csv("data/processed/cleaned_twcs.csv", index=False)

# Randomly sample 3,000 rows from the cleaned_queries_df dataframe and save it into a CSV file
sample_queries_df = cleaned_queries_df.sample(n=3000, random_state=42)
sample_queries_df.to_csv("data/processed/cleaned_twcs_sample.csv", index=False)