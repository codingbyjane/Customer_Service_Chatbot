import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

os.makedirs("figures", exist_ok=True)

# Load the data
query_sentiment_df = pd.read_csv("data/processed/twcs_sample_sentiment_results.csv")
print(query_sentiment_df.head(10))

query_intent_df = pd.read_csv("data/processed/twcs_sample_intent_results.csv")
print(query_intent_df.head(10))

sentiment_counts = query_sentiment_df["sentiment"].value_counts()
print(sentiment_counts)

sentiment_labels = ["negative", "neutral", "positive"]

# Colour mapping
sentiment_colours = {
    "negative" : "#a14242",
    "neutral" : "#ffe463",
    "positive" : "#6fa287"
}
intent_colour = "#48cbcf"

# Define a function to plot the distribution of sentiment analysis results
def plot_sentiment_distribution(dataframe):
    sentiment_counts = dataframe["sentiment"].value_counts().reindex(sentiment_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8)) # Unpacking the subplots result into figure and axes variables
    bars = ax.bar(sentiment_counts.index, # Accessing sentiment names
                    sentiment_counts.values, # Accessing sentiment counts
                    color=[sentiment_colours[sentiment] for sentiment in sentiment_counts.index]) # Look up a colour corresponding to each sentiment with a list comprehension
    
    ax.bar_label(bars, padding=2)
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Sentiment Distribution of Customer Queries")

    fig.savefig("figures/sentiment_distribution.png")
    plt.show()

# Define a function to plot the distribution of intent labels
def plot_intent_distribution(dataframe, intent_column="label"):
    intent_counts = dataframe[intent_column].value_counts().sort_values()
    wrap_labels = [textwrap.fill(label, 30) for label in intent_counts.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(wrap_labels, 
                  intent_counts.values, 
                  color=intent_colour)
    
    ax.bar_label(bars, padding=2)
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Intent Distribution of Customer Queries")

    fig.tight_layout()
    fig.savefig("figures/intent_distribution.png")
    plt.show()

plot_sentiment_distribution(query_sentiment_df)
plot_intent_distribution(query_intent_df)