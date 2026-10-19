# Customer Service Chatbot: Intent Recognition and Sentiment Analysis Implementation
A primitive proof-of-concept chatbot prototype created for demonstration of intent recognition and sentiment analysis through implementing pre-trained NLP models. Sentiment analysis and intent recognition tasks were performed with pre-trained Hugging Face transformer models.

## Overall project pipeline
1. **Preprocessing** (`preprocess.py`): loads the Customer Support on Twitter dataset, filters inbound opening message customer queries, cleans and preprocesses the text (URLs, mentions, whitespace, duplicates) and saves a random sample of consisting 3000 queries.
2. **Sentiment** (`sentiment_analysis.py`): loads the sentiment analysis pipeline for `cardiffnlp/twitter-roberta-base-sentiment-latest`, assigns each customer query with a sentiment label (negative, neutral, positive) and a corresponding confidence score.
3. **Intent** (`intent_recognition.py`): loads the sentiment analysis pipeline for `facebook/bart-large-mnli` used for zero-shot classification against [N] hand-written candidate intents.
**Chatbot** (`chatbot.py`): runs both models on every user input message, picks the predefined response for the detected intent and applies escalation rules.
5. **Plots** (`plots.py`): polts distribution charts for the sentiment and intent results, saved in `figures/`.

## Setup
This project requires Python version [3.9] or newer.

Installation guideline:
```bash
python3 -m venv .chatbot_env
source .chatbot_env/bin/activate
pip install -r requirements.txt
```

## Dataset
Dataset: *Customer Support on Twitter* from Kaggle ([https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data]).