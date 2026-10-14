# Importing libraries
import pandas as pd

# Importing sentiment analysis and intent recognition funcitons
from sentiment_analysis import get_sentiment
from intent_recognition import get_intent, candidate_labels

responses_dict = {
    "requesting a refund or return" : "Sure, I can help with that. Please provide your order number and I'll start the refund process.",
    "requesting an order cancellation" : "",
    "a billing or payment issue" : "",
    "a delivery or shipping issue" : "Could you please provide the order number so I can check the shipping status?",
    "service cancellation complaint" : "",
    "an account or login issue" : "Please provide the username or email linked to your account",
    "a product inquiry" : "I'm happy to help with ",
    "giving thanks or positive feedback" : "You're very welcome! Let me know if you need anything else.",
    "poor customer service" : "",
    "requesting a status update" : "",
    "reporting a technical or software problem" : "",
    "requesting to contact a human agent" : ""
}

empathy_sentence = "I'm sorry about the inconvenience."
clarify_message = "I'm not sure I understood. Could you please rephrase or add more detail?"
human_agent_intent = "requesting to contact a human agent"
escalation_message = "Please wait, I am connecting you with a human agent."


class CustomerServiceChatbot:
    def __init__(self, intent_confidence_threshold = 0.3, negative_sentiment_threshold=0.65, max_failed_rounds=2):
        self.intent_confidence_threshold = intent_confidence_threshold
        self.negative_sentiment_threshold = negative_sentiment_threshold
        self.max_failed_rounds = max_failed_rounds

    def analyse_query(self, text):
        sentiment_label, sentiment_score = get_sentiment(text) # get_sentiment function returns a label-score tuple which is going to be stored in sentiment_label, sentiment_score variables accordingly
        intent_label, intent_score = get_intent(text) # get_intent function returns label-score tuple for intent recognition results

        query_analysis = {
            "text" : text,
            "sentiment" : sentiment_label,
            "sentiment score" : sentiment_score,
            "intent" : intent_label,
            "intent_score" : intent_score
        }

        # Return a dictionary containing the input query along with sentiment analysis and intent recognition results
        return query_analysis

    def decide_response(self, query_analysis):

        if query_analysis['intent'] == human_agent_intent and query_analysis['intent_score'] >= self.intent_confidence_threshold:
            return "escalate"