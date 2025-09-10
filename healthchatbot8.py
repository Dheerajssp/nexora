import nltk
import spacy
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

class HealthChatbot:
    def _init_(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.symptom_classifier = None
        self.disease_matcher = TfidfVectorizer()
        self.symptom_database = self.load_symptom_database()
        self.emergency_keywords = [
            'heart attack', 'chest pain', 'difficulty breathing', 
            'severe bleeding', 'unconscious', 'stroke', 'allergic reaction'
        ]
        
    def load_symptom_database(self):
        """Load symptom-disease database"""
        # In production, this would load from a proper medical database
        return {
            'fever': {
                'possible_conditions': ['flu', 'cold', 'infection', 'covid-19'],
                'questions': ['How high is your fever?', 'Any other symptoms?'],
                'advice': 'Take rest, drink fluids, monitor temperature'
            },
            'headache': {
                'possible_conditions': ['tension headache', 'migraine', 'sinus infection'],
                'questions': ['Where is the pain located?', 'How severe (1-10)?'],
                'advice': 'Stay hydrated, rest in dark room, avoid screens'
            },
            'cough': {
                'possible_conditions': ['cold', 'flu', 'bronchitis', 'pneumonia'],
                'questions': ['Is it dry or with phlegm?', 'Any fever?'],
                'advice': 'Stay hydrated, use honey, avoid smoke'
            },
            'stomach pain': {
                'possible_conditions': ['indigestion', 'gastritis', 'food poisoning'],
                'questions': ['When did it start?', 'What did you eat recently?'],
                'advice': 'Eat light foods, stay hydrated, rest'
            }
        }
    
    def preprocess_text(self, text):
        """Preprocess user input"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract symptoms, body parts, and intensity
        symptoms = []
        body_parts = []
        intensity = None
        
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop:
                if self.is_symptom(token.text):
                    symptoms.append(token.text)
                elif self.is_body_part(token.text):
                    body_parts.append(token.text)
        
        # Extract intensity/severity
        intensity_words = ['mild', 'moderate', 'severe', 'extreme']
        for word in intensity_words:
            if word in text:
                intensity = word
                break
        
        return {
            'original_text': text,
            'symptoms': symptoms,
            'body_parts': body_parts,
            'intensity': intensity,
            'processed_tokens': [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        }
    
    def is_symptom(self, word):
        """Check if word is a medical symptom"""
        symptoms_list = [
            'fever', 'headache', 'cough', 'pain', 'nausea', 'vomiting', 
            'diarrhea', 'fatigue', 'dizziness', 'rash', 'swelling'
        ]
        return word in symptoms_list
    
    def is_body_part(self, word):
        """Check if word is a body part"""
        body_parts = [
            'head', 'chest', 'stomach', 'back', 'throat', 'eye', 
            'ear', 'nose', 'leg', 'arm', 'hand', 'foot'
        ]
        return word in body_parts
    
    def check_emergency(self, text):
        """Check if symptoms indicate emergency"""
        text_lower = text.lower()
        
        emergency_indicators = {
            'critical': [
                'chest pain', 'difficulty breathing', 'severe bleeding',
                'unconscious', 'heart attack', 'stroke'
            ],
            'urgent': [
                'high fever', 'severe pain', 'persistent vomiting',
                'severe headache', 'allergic reaction'
            ]
        }
        
        for severity, keywords in emergency_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        'is_emergency': True,
                        'severity': severity,
                        'action': 'Call emergency services immediately' if severity == 'critical' 
                                else 'Seek immediate medical attention'
                    }
        
        return {'is_emergency': False}
    
    def analyze_symptoms(self, user_input):
        """Analyze user symptoms and provide recommendations"""
        processed = self.preprocess_text(user_input)
        
        # Check for emergency first
        emergency_