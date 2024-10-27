#Model Libraries
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
import logging
import os
import sys
from datasets import load_from_disk
import random
# Configure logging to current directory
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/copa_daemon_service.log"),  # Log file in current directory
 #       logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)



class CopqModel:
    def __init__(self) -> None:
        self.logger= logging.getLogger('CopqLogger')
        
        self.model_name = "google/flan-t5-small"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.logger.debug("Model and Tokenizer is loaded")

        # Load the LoRA model
        self.lora_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flan_t5_copa_lora_saved")
        self.lora_config = LoraConfig.from_pretrained(self.lora_model_path)
        self.model = get_peft_model(self.model, self.lora_config)

    def predict(self, prompt:str) -> str:
        self.logger.debug(f"Input text: {prompt}")
        inputs=self.model_tokenizer(
                prompt,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
        )
        with torch.no_grad():
            outputs=self.model.generate(**inputs)
        pred = self.model_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        self.logger.debug(f"Prediciton:{pred}")
        return pred
    
if __name__=="__main__":
    copa=load_from_disk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "copa"))

    inputs=copa[random.randint(0,400)]['input_text']
    Copa=CopqModel()

    outputs=Copa.predict(inputs)
    print(outputs)