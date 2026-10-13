# Importing dependencies
import pandas as pd
from transformers import pipeline

zs_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = ["refund", "delivery issue", "order cancellation", "technical support", "product inquiry", "billing issue", "payment issue", "general inquiry"]

def get_intent(text, candidate_labels):
    intent_result = zs_classifier(text, candidate_labels)
    