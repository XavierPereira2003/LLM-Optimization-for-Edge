import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
import sys
from datasets import load_from_disk
import random

log_path="./logs/"
os.makedirs("./logs/", exist_ok=True) 
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/Services.log"),

    ]
)


class Classifier:
    def __init__(self) -> None:
        self.logger=logging.getLogger("Bert_logger")
        model_path =os.path.join(os.path.dirname(__file__), "/Bert_classifier")
        self.logger.debug(f"File Path {model_path}")

        self.classifier_model=AutoModelForSequenceClassification.from_pretrained("/home/deon/code/Project/Expo/Models/Bert_classifier")
        self.tokenizer=AutoTokenizer.from_pretrained("/home/deon/code/Project/Expo/Models/Bert_classifier")
        
        self.logger.info("Model and Tokenizer is loaded")
        

    def classify_text(self, input_text:str)->int:
        self.logger.info("Classifying")
        # Tokenize the input, specifying return_tensors='pt' only for the tokenizer
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        self.logger.debug(f"Tokenized the input \"{input_text}\"")
        with torch.no_grad():
            outputs = self.classifier_model(**inputs) 

        # Get predictions
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()  
        self.logger.debug(f"Generated the output '{predicted_class}'")
        return int(predicted_class)

class BoolqDataset:
    def __init__(self) -> None:
        print((f'File Path {os.path.join(os.path.dirname(os.path.abspath(__file__)), "/Boolq")}'))
        #        model_path =os.path.join(os.path.dirname(os.path.abspath(__file__)), "/Bert_classifier")
        self.data=load_from_disk("/home/deon/code/Project/Expo/Models/Boolq")
        

    def randomData(self)->str:
        return self.data[random.randint(0,9427)]['input_text']

class CopaDataset:
    def __init__(self) -> None:
        self.data=load_from_disk("/home/deon/code/Project/Expo/Models/copa")

    def randomData(self)->str:
        return self.data[random.randint(0,400)]['input_text']

class CombindedDataset:
    def __init__(self) -> None:
        self.data=load_from_disk("/home/deon/code/Project/Expo/Models/Boolq_and_copa")

    def randomData(self)->str:
        return self.data[random.randint(0,9827)]['input_text']

if __name__=="__main__":
    inputs=CopaDataset().randomData()
    classifier=Classifier()
    predicted_class=classifier.classify_text(inputs)
    print(predicted_class)