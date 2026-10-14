# Importing libraries
import pandas as pd

# Importing sentiment analysis and intent recognition funcitons
from sentiment_analysis import get_sentiment
from intent_recognition import get_intent, candidate_labels

responses_dict = {
    "requesting a refund or return" : "",
    "requesting an order cancellation" : "",
    "a billing or payment issue" : "",
    "a delivery or shipping issue" : "",
    "service cancellation complaint" : "",
    "an account or login issue" : "",
    "a product inquiry" : "",
    "giving thanks or positive feedback" : "You're very welcome! Let me know if you need anything else.",
    "poor customer service" : "",
    "requesting a status update" : "",
    "reporting a technical or software problem" : "",
    "requesting to contact a human agent" : ""
}