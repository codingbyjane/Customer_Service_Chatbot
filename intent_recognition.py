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



def batch_intent_recognition(queries, batch_size=8):

    query_intent_results = []

    for query_index in range(0, len(queries), batch_size):
        batch = queries[query_index:query_index + 8]
        batch_intent_results = zs_classifier(batch, candidate_labels, batch_size=batch_size, truncation=True, max_length=128)
        query_intent_results.extend(batch_intent_results)

        queries_processed = min(query_index + batch_size/len(queries))
        print(f"Processed {queries_processed}/{len(queries)} queries")

        intent = query_intent_results['labels'][0]
        score = query_intent_results['scores'][0]

    return intent, score



if __name__ == "__main__":

    example_sentence = "I need help with my order. It hasn't arrived yet."

    example_result = zs_classifier(example_sentence, candidate_labels)
    print(example_result)


    sample_queries_df = pd.read_csv("data/processed/cleaned_twcs_sample.csv")

    # Convert the 'cleaned_text' column of the dataframe into a list of strings for batch processing
    queries_list = sample_queries_df['cleaned_text'].tolist()