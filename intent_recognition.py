# Importing dependencies
import pandas as pd
from transformers import pipeline

zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels that each user query will be compared against
candidate_labels = ["requesting a refund or return", 
                    "requesting an order cancellation", 
                    "a billing or payment issue",
                    "a delivery or shipping issue", 
                    "service cancellation complaint",
                    "an account or login issue", 
                    "a product inquiry",  
                    "giving thanks or positive feedback", 
                    "poor customer service", 
                    "requesting a status update",
                    "reporting a technical or software problem",
                    "requesting to contact a human agent"
                    ]

# Define a custom hypothesis sentence template that the premise (user query) will be compared against because NLI works best on natural text
hypothesis_template = "This customer message is about {}."

# Define a reusable intent recognition function (used by the chatbot)
def get_intent(text):
    intent_result = zs_classifier(text, candidate_labels, hypothesis_template=hypothesis_template) # BART MNLI compares the user query against a hypothesis iterating through each intent and runs a softmax across all labels, so the highest scoring label is the most likely intent

    # Create a dictionary with label-score pairs using a dictionary comprehension
    intent_scores = {label:score for label,score in zip(intent_result['labels'], intent_result['scores'])}

    top_label = max(intent_scores, key=intent_scores.get) # Retrieve the label (key) corresponding to the highest score (value). Running the max() function on a dictionary returns the key (label) with the highest value.
    top_score = intent_scores[top_label]

    return top_label, top_score


def batch_intent_recognition(queries, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template, batch_size=16):

    query_intent_results = []

    for query_index in range(0, len(queries), batch_size):
        batch = queries[query_index:query_index + batch_size]
        batch_intent_results = zs_classifier(batch, candidate_labels, hypothesis_template=hypothesis_template, batch_size=batch_size)

        for result in batch_intent_results: # Loop through resulting dictionaries (pipeline outputs a dictionary with 'sequence', 'labels', 'scores' for every query)
            top_label, top_score = max(zip(result['labels'], result['scores']), key=lambda x:x[1]) # Retrieve the highest score and its corresponding label for each query.
            query_intent_results.append((top_label, top_score)) # Append label-score tuples for every query to the results list

        queries_processed = min(query_index + batch_size, len(queries))
        print(f"Processed {queries_processed}/{len(queries)} queries")

    return query_intent_results


if __name__ == "__main__":

    # Load the cleaned dataset sample
    sample_queries_df = pd.read_csv("data/processed/cleaned_twcs_sample.csv")

    # Convert the 'cleaned_text' column of the dataframe into a list of strings for batch processing
    queries_list = sample_queries_df['cleaned_text'].tolist()

    '''
    # Apply the batch intent recognition function to the sample list of queries
    intent_recognition_results = batch_intent_recognition(queries_list)
    '''

    small_sample_df = sample_queries_df.head(100).copy()
    print(small_sample_df.head(10))

    small_sample_list = small_sample_df['cleaned_text'].tolist()

    test_results = batch_intent_recognition(small_sample_list)

    test_results_df = pd.DataFrame({
        "author_id" : small_sample_df["author_id"].to_numpy(),
        "cleaned_text" : small_sample_df["cleaned_text"].to_numpy(),
        "label" : [result[0] for result in test_results],
        "score" : [result[1] for result in test_results]
    })

    print(test_results_df.head(30))