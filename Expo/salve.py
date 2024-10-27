#!/usr/bin/env python3
import socket
import sys
import logging
import signal
import json
from datetime import datetime

# Configure logging to current directory
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daemon_service.log"),  # Log file in current directory
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)
logger = logging.getLogger('./logs/JsonDaemonService')

class JsonDaemonService:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        
    def setup_socket(self):
        """Initialize and setup the socket"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            logger.info(f"JSON service listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to setup socket: {str(e)}")
            sys.exit(1)

    def process_json(self, data: dict) -> dict:
        """Process the received JSON data and return a response"""
        try:
            # Add processing timestamp
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "received_data": data,
                "message": "JSON processed successfully"
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
                if len(buffer) > 1024 * 1024:  # 1MB limit
                    raise ValueError("JSON data too large")
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

def run_server():
    """Run the server"""
    service = JsonDaemonService()
    service.run()

if __name__ == "__main__":
    run_server()