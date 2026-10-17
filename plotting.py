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
    sentiment_counts = dataframe["sentiment"].value_counts().reindex(sentiment_labels, fill_value=0) # Count all unique sentiment instances in specified order (fill possible missing values/categories with 0)

    fig, ax = plt.subplots(figsize=(12, 8)) # Unpacking the subplots result into figure and axes variables

    bars = ax.bar(sentiment_counts.index, # Accessing sentiment labels (x-coordinates of the bars)
                    sentiment_counts.values, # Accessing sentiment counts (y-coordinates/height of the bars)
                    color=[sentiment_colours[sentiment] for sentiment in sentiment_counts.index]) # Look up a colour corresponding to each sentiment with a list comprehension
    
    ax.bar_label(bars, padding=2) # Display each bar's numerical value
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Sentiment Distribution of Customer Queries")

    fig.savefig("figures/sentiment_distribution.png")
    plt.show()

# Define a function to plot the distribution of intent labels
def plot_intent_distribution(dataframe, intent_column="label"):
    intent_counts = dataframe[intent_column].value_counts().sort_values() # Count unique intent label instances and sort them in ascending order
    wrap_labels = [textwrap.fill(label, 40) for label in intent_counts.index] # Wrap the labels so that every line is maximally 40 characters long

    fig, ax = plt.subplots(figsize=(15, 7))

    # Create a horizontal bar chart for the distribution of intents
    bars = ax.barh(wrap_labels, 
                  intent_counts.values, 
                  color=intent_colour)
    
    ax.bar_label(bars, padding=2) # Display each bar's numerical value
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Intent Distribution of Customer Queries")

    fig.tight_layout()
    fig.savefig("figures/intent_distribution.png")
    plt.show()

plot_sentiment_distribution(query_sentiment_df)
plot_intent_distribution(query_intent_df)