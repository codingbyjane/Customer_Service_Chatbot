# Importing libraries
import pandas as pd

# Importing sentiment analysis and intent recognition funcitons
from sentiment_analysis import get_sentiment
from intent_recognition import get_intent, candidate_labels

empathy_sentence = "I'm sorry about the inconvenience."
clarify_message = "I'm not sure I understood. Could you please rephrase or add more detail?"
human_agent_intent = "requesting to speak to a human agent"
escalation_message = "Please wait, I am connecting you with a human agent."
thanks_message = "You're very welcome! Let me know if you need anything else."

human_agent_intent = "requesting to speak to a human agent"
thanks_intent = "giving thanks or positive feedback"

responses_dict = {
    "requesting a refund or return" : "I can help with that. Please provide your order number and I'll start the refund or return process.",
    "requesting an order cancellation" : "I can help cancel your order. Please share your order number.",
    "a billing or payment issue" : "Could you provide the transaction reference and the amount charged? Please don't send the full card number.",
    "a delivery or shipping issue" : "Could you please provide your order or tracking number so I can check the shipping status?",
    "a service delay, cancellation, or disruption" : "Thank you for your feedback, it helps us improve our services in the future. Would you like me to connect you to our team member?",
    "an account or login issue" : "Please provide the username or email adress linked to your account.",
    "reporting a technical or software problem" : "Which device and app version are you using?",
    "a product inquiry" : "I'm happy to help with any of your questions. What would you like me to clarify?",
    "complaining about staff or customer service quality" : "Thank you for letting us know. I've noted your feedback for our team to review.",
    "asking for a status update on an earlier request or asking about wait time" : "Thank you for your patience. Please share your ticket number so I can check the case status.",
    "requesting a change to an existing order" : "I can help with that. Please share your order number and tell me what you'd like to change.",
    thanks_intent : thanks_message,
    human_agent_intent : escalation_message
}

# Checking if the candidate labels and messages match
assert set(responses_dict) == set(candidate_labels)
assert human_agent_intent in candidate_labels and thanks_intent in candidate_labels

class CustomerServiceChatbot:
    def __init__(self, intent_confidence_threshold = 0.3, negative_sentiment_threshold=0.85, max_failed_rounds=2):
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

        if query_analysis["sentiment"] == "negative" and query_analysis["intent"] == thanks_intent:
            return "escalate", "likely sarcasm"

        if query_analysis["sentiment"] == "negative" and query_analysis["sentiment_score"] >= self.negative_sentiment_threshold:
            return "escalate", "strongly negative sentiment" # connecting with a human agent if the sentiment of the query is strongly negative

        if query_analysis["intent_score"] < self.intent_confidence_threshold:
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
                reply = empathy_sentence + " " + reply

        self.results_log.append({**query_analysis, "decided_action" : decided_action, "reason" : reason}) # append the query analysis results to the results log (**query_analysis unpacks the dictionary created in analyse_query method)

        return reply

    def run(self):
        print("\nBot: Hello! How can I help you today?")

        while True: # Run the chatbot until an explicit exit command is executed
            text = input("You: ").strip() # Stripping the input sentence of unneccessary whitespaces

            if text.lower() in {"quit", "exit"}:
                print("Bot: Goodbye!")
                break

            if not text:
                continue
            
            print(f"Bot: {self.respond(text)}")


if __name__ == "__main__":
    chatbot = CustomerServiceChatbot()

    chatbot.run()

    # Save the chatbot decision and response log into a pandas dataframe
    chatbot_results_log_df = pd.DataFrame(chatbot.results_log)
    print(chatbot_results_log_df)