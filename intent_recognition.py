# Importing dependencies
import pandas as pd
from transformers import pipeline

zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["refund", "delivery issue", "order cancellation", "technical support", "product inquiry", "billing issue", "payment issue", "general inquiry", "subscription issue"]

# Define a reusable intent recognition function (used by the chatbot)
def get_intent(text, candidate_labels):
    intent_result = zs_classifier(text, candidate_labels)

    # Create a dictionary with label-score pairs using a dictionary comprehension
    intent_scores = {
        label:score for label,score in zip(intent_result['labels'], intent_result['scores'])
    }

    top_label = max(intent_scores, key=intent_scores.get) # Retrieve the label (key) corresponding to the highest score (value)
    top_score = intent_scores[top_label]

    return top_label, top_score


def batch_intent_recognition(queries, candidate_labels, batch_size=8):

    query_intent_results = []

    for query_index in range(0, len(queries), batch_size):
        batch = queries[query_index:query_index + batch_size]
        batch_intent_results = zs_classifier(batch, candidate_labels, batch_size=batch_size, truncation=True, max_length=128)

        for result in batch_intent_results: # Loop through resulting dictionaries (pipeline outputs a dictionary with 'sequence', 'labels', 'scores' for every query)
            top_label, top_score = max(zip(result['labels'], result['scores']), key=lambda x:x[1]) # Retrieve the highest score and its corresponding label for each query
            query_intent_results.append((top_label, top_score)) # Append label-score tuples for every query to the results list

        queries_processed = min(query_index + batch_size, len(queries))
        print(f"Processed {queries_processed}/{len(queries)} queries")

    return query_intent_results


if __name__ == "__main__":

    ''' example_sentence = "I need help with my order. It hasn't arrived yet."

    example_result = zs_classifier(example_sentence, candidate_labels)
    print(example_result) '''

    # Load the cleaned dataset sample
    sample_queries_df = pd.read_csv("data/processed/cleaned_twcs_sample.csv")

    # Convert the 'cleaned_text' column of the dataframe into a list of strings for batch processing
    queries_list = sample_queries_df['cleaned_text'].tolist()

    '''
    # Apply the batch intent recognition function to the sample list of queries
    intent_recognition_results = batch_intent_recognition(queries_list, candidate_labels)

    query_intent_df = pd.DataFrame({
        "author_id" : sample_queries_df['author_id'].to_numpy(),
        "cleaned_text" : sample_queries_df['cleaned_text'].to_numpy(),
        "intent" : [result[0] for result in intent_recognition_results],
        "confidence score" : [result[1] for result in intent_recognition_results]
        })

    print(query_intent_df.head(10)) '''

    small_sample_df = sample_queries_df.head(100).copy()
    print(small_sample_df.head(10))

    small_sample_list = small_sample_df['cleaned_text'].tolist()
    print(small_sample_list[:5])

    test_results = batch_intent_recognition(small_sample_list, candidate_labels)

    test_results_df = pd.DataFrame({
        "author_id" : small_sample_df["author_id"].to_numpy(),
        "cleaned_text" : small_sample_df["cleaned_text"].to_numpy(),
        "label" : [result[0] for result in test_results],
        "score" : [result[1] for result in test_results]
    })

    print(test_results_df.head(10))