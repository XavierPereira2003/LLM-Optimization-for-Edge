#!/usr/bin/env python3
import socket
import sys
import logging
import signal
import json
from datetime import datetime


#Model Libraries
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig

# Configure logging to current directory
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/boolq_daemon_service.log"),  # Log file in current directory
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)
logger = logging.getLogger('BoolqLogger')

class BoolqDaemonService:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.model=BooqlModel()

        
    def setup_socket(self):
        """Initialize and setup the socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            logger.info(f"Boolq service listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to setup socket: {str(e)}")
            sys.exit(1)

    def process_request(self, data: dict) -> dict:
        """Process the received JSON data and return a response"""
        try:
            # Add processing timestamp

            prompt=data['text']
            computer_message=self.model.predict(prompt)
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "message": computer_message
            }

            return response
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def receive_json(self, client_socket: socket.socket) -> dict:
        """Receive and parse JSON data from client"""
        buffer = ""
        while True:
            chunk = client_socket.recv(4096).decode('utf-8')
            if not chunk:
                raise ConnectionError("Client disconnected")
            
            buffer += chunk
            try:
                data = json.loads(buffer)
                return data
            except json.JSONDecodeError:
                if len(buffer) > 5* 1024 * 1024:
                    raise ValueError("Request too large")
                continue

    def handle_client(self, client_socket: socket.socket):
        """Handle individual client connections with JSON processing"""
        try:
            client_socket.settimeout(30)
            
            while True:
                try:
                    data = self.receive_json(client_socket)
                    logger.info(f"Received JSON: {data}")
                    
                    response = self.process_json(data)
                    
                    response_json = json.dumps(response)
                    client_socket.send(f"{response_json}\n".encode('utf-8'))
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "status": "error",
                        "timestamp": datetime.now().isoformat(),
                        "error": "Invalid JSON format",
                        "details": str(e)
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    break
                    
                except ConnectionError:
                    logger.info("Client disconnected")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling client: {str(e)}")
        finally:
            client_socket.close()

    def run(self):
        """Main service loop"""
        self.running = True
        self.setup_socket()
        
        def signal_handler(signo, frame):
            logger.info("Received signal to terminate")
            self.running = False
            self.sock.close()
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.info("JSON daemon service started")
        
        while self.running:
            try:
                client_sock, address = self.sock.accept()
                logger.info(f"Accepted connection from {address}")
                self.handle_client(client_sock)
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {str(e)}")


class BooqlModel:
    def __init__(sel    f) -> None:
        self.model_name = "google/flan-t5-small"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load the LoRA model
        self.lora_model_path = "./flan_t5_boolq_lora_saved"
        self.lora_config = LoraConfig.from_pretrained(self.lora_model_path)
        self.model = get_peft_model(self.model, self.lora_config)

    def predict(self, prompt:str) -> str:
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
        return pred

def run_server():
    """Run the server"""
    service = BoolqDaemonService()
    service.run()

if __name__ == "__main__":
    run_server()