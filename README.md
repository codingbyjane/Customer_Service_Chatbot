# Customer Service Chatbot: Intent Recognition and Sentiment Analysis Implementation
A primitive proof-of-concept chatbot prototype created for demonstration of intent recognition and sentiment analysis through implementing pre-trained NLP models. Sentiment analysis and intent recognition tasks were performed with pre-trained Hugging Face transformer models.

## Overall project pipeline
1. **Preprocessing** (`preprocess.py`): loads the Customer Support on Twitter dataset, filters inbound opening message customer queries, cleans and preprocesses the text (URLs, mentions, whitespace, duplicates) and saves a random sample of consisting 3000 queries.
2. **Sentiment** (`sentiment_analysis.py`): loads the sentiment analysis pipeline for `cardiffnlp/twitter-roberta-base-sentiment-latest`, assigns each customer query with a sentiment label (negative, neutral, positive) and a corresponding confidence score.
3. **Intent** (`intent_recognition.py`): loads the sentiment analysis pipeline for `facebook/bart-large-mnli` used for zero-shot classification against [N] hand-written candidate intents.
4. **Chatbot** (`chatbot.py`): runs both models on every user input message, picks the predefined response for the detected intent and applies escalation rules.
5. **Plots** (`plots.py`): polts distribution charts for the sentiment and intent results, saved in `figures/`.

## Setup
This project requires Python version `3.9` or newer.

Installation guideline:
```bash
python3 -m venv .chatbot_env
source .chatbot_env/bin/activate
pip install -r requirements.txt
```

## Dataset
Dataset: *Customer Support on Twitter* from Kaggle (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data).
This dataset is not included directly in this repository because of its size. Place the [twcs.csv] file at:
```
data/raw/twcs.csv
```

## Running the project
Run from the project root, in this order:

```bash
python preprocess.py           # clean data and create the sample
python sentiment_analysis.py   # sentiment analysis on the sample -> data/processed/
python intent_recognition.py   # intent on the sample (slower processing) -> data/processed/
python plots.py                # figures and plots -> figures/
python chatbot.py              # interactive chat in the terminal (type "quit" of "exit" to exit)
```

## Project structure

```
data/raw/            original dataset (not tracked)
data/processed/      cleaned data and model results
figures/             generated charts
preprocess.py
sentiment_analysis.py
intent_recognition.py
chatbot.py
plots.py
requirements.txt
```

## Example conversation turns
- refund: "Hello, I would like to return a jacket I bought last week and get a refund."
- order cancellation: "I'd like to cancel my order please."
- billing: "I think I was charged twice for the same payment on my card."
- delivery: "My parcel has not arrived yet and the tracking has not updated."
- service disruption: "Is there a problem with the internet in my area today? It has been down since this morning."
- account/login: "I can't log in to my account, it says my password is wrong."
- product inquiry: "Does the blue version of this phone come with a charger?"
- staff/service complaint: "The staff at the store were not very helpful today."
- status update: "I'd like to ask how long do I have to wait for a reply?"
- technical problem: "The app keeps freezing when I open the settings page."

### Escalation rules testing
- human request: "I would like to speak to a human agent please."
- strongly negative: "I am absolutely furious, this is the worst service I have ever experienced and you are useless!"
- mildly negative: "My delivery is two days late, which is a bit disappointing."
- sarcasm: "Oh great, thanks for cancelling my flight again, this is unacceptable."
- low confidence: "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

## Acknowledgements
Models by Cardiff NLP and Facebook AI, via Hugging Face Transformers. Dataset by Thought Vector (https://www.kaggle.com/organizations/thoughtvector). Please check the dataset's license on Kaggle before reusing the data.