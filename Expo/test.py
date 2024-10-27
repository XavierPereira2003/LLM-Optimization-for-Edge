import socket
import json
from datetime import datetime

def test_service():
    # Test data
    test_data = {
        "message": "Hello, JSON Service!",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "key1": "value1",
            "key2": "value2"
        }
    }
    
    # Connect and send data
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 5001))
        s.send(json.dumps(test_data).encode('utf-8'))
        
        # Receive and print response
        response = s.recv(4096).decode('utf-8')
        print("Server response:", json.loads(response))

if __name__ == "__main__":
    test_service()