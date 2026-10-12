# Importing dependencies
import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Define a reusable sentiment analysis function (used by the chatbot)
def get_sentiment(text):
    sentiment_result = sentiment_pipeline(text, truncation=True, max_length=128)[0]
    return sentiment_result['label'].lower(), sentiment_result['score']


# Implement a batch processing function to iterate through the large amount of queries efficiently
def batch_sentiment_analysis(queries, batch_size=32):
    
    query_sentiment_results = [] # Create an empty list to store the sentiment analysis results

    for query_index in range(0, len(queries), batch_size): # Iterate through the queries in batches of the specified batch size (starting from index 0 to the length of the queries list, with a step size of batch_size)
        batch = queries[query_index:query_index + batch_size] # Specify where the batch starts and ends by slicing the queries list from the current index (inclusive) to the current index + batch_size (exclusive) (a batch size of 32 gives start values 0, 32, 64, 96, ...)
        batch_sentiment_results = sentiment_pipeline(batch, batch_size=batch_size, truncation=True, max_length=128) # Perform sentiment analysis on the currect batch of queries using the pipeline function and store the results in a temporary variable (truncate the input text to a maximum length of 128 tokens to avoid errors with long queries)
        query_sentiment_results.extend(batch_sentiment_results) # Add all the current batched results to the query_sentiment_results list using the extend() method

        queries_processed = min(query_index + batch_size, len(queries)) # Adding a process indicator variable (returns the smaller value between the current index + batch_size and the total number of queries)
        print(f"Processed {queries_processed}/{len(queries)} queries")

    return query_sentiment_results # Return the complete list of sentiment analysis results after processing all batches


if __name__ == "__main__": # Everything below this condition only runs when this file is executed directly

    # Check the model's label mapping
    print(sentiment_pipeline.model.config.id2label)

    # Import the cleaned dataset sample
    sample_queries_df = pd.read_csv("data/processed/cleaned_twcs_sample.csv")
    print(f"Total number of queries in the sample dataset: {len(sample_queries_df)}")

    # Convert the 'cleaned_text' column to a list of strings for batch processing
    queries_list = sample_queries_df['cleaned_text'].tolist()

    # Apply the batch sentiment analysis function to the sample list of queries
    sentiment_analysis_results = batch_sentiment_analysis(queries_list, batch_size=32)

    # Convert the sentiment analysis results to a separate DataFrame
    query_sentiment_df = pd.DataFrame({
        "author_id" : sample_queries_df['author_id'].to_numpy(),
        "cleaned_text" : sample_queries_df['cleaned_text'].to_numpy(),
        "sentiment" : [result['label'].lower() for result in sentiment_analysis_results],
        "confidence_score" : [result['score'] for result in sentiment_analysis_results]
    })

    # Display the first 20 rows of the sentiment analysis results DataFrame
    print(query_sentiment_df.head(10))

    # Display the distribution of sentiment labels in the results DataFrame
    print(f"\nSentiment label distribution:\n{query_sentiment_df['sentiment'].value_counts()}")

    # Save the sentiment analysis results to a CSV file (don't save the pandas indexes)
    query_sentiment_df.to_csv("data/processed/twcs_sample_sentiment_results.csv", index=False)