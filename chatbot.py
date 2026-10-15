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
        self.failed_rounds = 0
        self.results_log = [] # a list of dictionaries containing infomation about processing each user input query

    def analyse_query(self, text):
        sentiment_label, sentiment_score = get_sentiment(text) # get_sentiment function returns a label-score tuple which is going to be stored in sentiment_label, sentiment_score variables accordingly
        intent_label, intent_score = get_intent(text) # get_intent function returns label-score tuple for intent recognition results

        query_analysis = {
            "text" : text,
            "sentiment" : sentiment_label,
            "sentiment_score" : sentiment_score,
            "intent" : intent_label,
            "intent_score" : intent_score
        }

        # Return a dictionary containing the input query along with sentiment analysis and intent recognition results
        return query_analysis

    def decide_response(self, query_analysis):

        if query_analysis["intent"] == human_agent_intent and query_analysis["intent_score"] >= self.intent_confidence_threshold:
            return "escalate", "request to connect with a human agent" # connecting with a human agent if a user asks

        if query_analysis["sentiment"] == "negative" and query_analysis["sentiment_score"] >= self.negative_sentiment_threshold:
            return "escalate", "strongly negative sentiment" # connecting with a human agent if the sentiment of the query is strongly negative

        if query_analysis["intent_score"] <= self.intent_confidence_threshold:
            if self.failed_rounds + 1 >= self.max_failed_rounds:
                return "escalate", "repeated low-confidence intents" # connecting with a human agent after 2 consecutive low-confidence intents
            return "clarify", "low-confidence intent" # asking to clarify a low-confidence intent
        
        return "answer", "high-confidence intent"

    def respond(self, text):
        query_analysis = self.analyse_query(text)
        decided_action, reason = self.decide_response(query_analysis)

        self.failed_rounds = self.failed_rounds + 1 if decided_action == "clarify" else 0

        if decided_action == "escalate":
            reply = escalation_message

        elif decided_action == "clarify":
            reply = clarify_message

        else: 
            reply = responses_dict[query_analysis["intent"]] # accessing the mapped response from the predefined dictionary

            if query_analysis["sentiment"] == "negative":
                reply = empathy_sentence + reply

        self.results_log.append({"decided_action" : decided_action, "reason" : reason}) # append the query analysis results to the results log

    def run(self):
        print("Bot: Hello! How can I help you today?")

        while True: # Run the chatbot until an explicit exit command is executed
            user_input_text = input("You: ").strip() # Stripping the input sentence of unneccessary whitespaces

            if user_input_text.lower() in {"quit", "exit"}:
                print("Bot: Goodbye!")
                break

            if not user_input_text:
                continue
            
            print(f"Bot: {self.respond(user_input_text)}")

if __name__ == "__main__":
    chatbot = CustomerServiceChatbot()
