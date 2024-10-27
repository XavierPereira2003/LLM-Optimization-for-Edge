

import sys
import socket
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/master.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger('InferenceClient')

@dataclass
class InferenceRequest:
    """Structure for inference requests"""
    input_data: Any
    model_params: Optional[Dict] = None
    request_id: str = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def to_json(self) -> Dict:
        return {
            "request_id": self.request_id,
            "input_data": self.input_data,
            "timestamp": datetime.now().isoformat()
        }

class InferenceClient:
    def __init__(self, host: str = 'localhost', port: int = 5000, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout

    def infer(self, input_data: Union[Dict, str, list], 
              model_params: Optional[Dict] = None) -> Dict:
        """
        Make an inference request to the daemon service
        
        Args:
            input_data: The input data for inference
            model_params: Optional parameters for the inference
            
        Returns:
            Dictionary containing the inference results
        """
        request = InferenceRequest(
            input_data=input_data,
            model_params=model_params
        )
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                
                # Send request
                request_json = json.dumps(request.to_json())
                sock.send(request_json.encode('utf-8'))
                logger.info(f"Sent request: {request_json}")
                
                # Receive response
                response_data = ""
                while True:
                    chunk = sock.recv(4096).decode('utf-8')
                    if not chunk:
                        break
                    response_data += chunk
                    
                    try:
                        response = json.loads(response_data)
                        break
                    except json.JSONDecodeError:
                        continue
                
                logger.info(f"Received response: {response}")
                
                if response.get("status") == "error":
                    raise InferenceError(
                        f"Inference failed: {response.get('error', 'Unknown error')}"
                    )
                    
                return response
                
        except socket.timeout:
            raise InferenceError("Inference request timed out")
        except ConnectionRefusedError:
            raise InferenceError("Could not connect to inference service")
        except Exception as e:
            raise InferenceError(f"Inference failed: {str(e)}")

class InferenceError(Exception):
    """Custom exception for inference errors"""
    pass



class Classifier:
    def __init__(self) -> None:
        self.classifier_model=AutoModelForSequenceClassification.from_pretrained("./Bert_classifier")
        self.tokenizer=AutoTokenizer.from_pretrained("./Bert_classifier")
        self.logger=logging.getLogger("Model_logger")
        self.logger.info("Model and Tokenizer is loaded")
        
        

    def classify_text(self, input_text:str)->int:
        
        # Tokenize the input, specifying return_tensors='pt' only for the tokenizer
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        self.logger.debug(f"Tokenized the input \"{input_text}\"")
        
        with torch.no_grad():
            outputs = self.classifier_model(**inputs) 

        # Get predictions
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()  
        self.logger.debug(f"Generated the output '{predicted_class}'")
        self.logger.info("Returning the value")
        return int(predicted_class)
    
if __name__ == "__main__":
    inputs="do good samaritan laws protect those who help at an accident"

    classifier=Classifier()

    predicted_class=classifier.classify_text(inputs)


    jason_input={
        "text":inputs,
    }

    boolq_inference=InferenceClient(port=5000)
    copa_inference=InferenceClient(port=5001)
    logger.debug(f"The class for {inputs} is {predicted_class}")
    
    if predicted_class==0:
        
        try:
            result=boolq_inference.infer(
                input_data=inputs
            )
            print(result)
            logger.info("Returned Result{result}")
        except InferenceError as e:
            print(f"Error during inference{e}")
            logger.error("Error during inference boolq{e}")
    else:
        
        try:
            result=copa_inference.infer(
                input_data=inputs
            )
            print(result)
            logger.info("Returned Result{result}")
        except InferenceError as e:
            print(f"Error during inference{e}")
            logger.error("Error during inference boolq{e}")