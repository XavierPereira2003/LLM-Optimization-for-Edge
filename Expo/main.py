import grpc
import json
from services_pb2 import JsonRequest
from services_pb2_grpc import JsonProcessor1Stub, JsonProcessor2Stub
import time
import logging
from Models.Bert import Classifier, BoolqDataset, CombindedDataset, CopaDataset
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/Service.log"),
    ]
)
logger=logging.getLogger("MainService")

class MainService:
    def __init__(self):
        # Create channels to both services
        self.channel1 = grpc.insecure_channel('localhost:50051')
        self.channel2 = grpc.insecure_channel('localhost:50052')
        
        # Create stubs for both services
        self.stub1 = JsonProcessor1Stub(self.channel1)
        self.stub2 = JsonProcessor2Stub(self.channel2)
        
        print("JSON Main Service initialized and connected to processing services")
        logger.info("JSON Main Service initialized and connected to processing services")

        self.model=Classifier()

    
    def process_json(self, data):
        try:
            
            json_data={
                "metadata": {
                    "processor": "Boolq Service",
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "processing_id": str(time.time())
                },
                "prompt":data
            }
            json_data=json.dumps(json_data)
            predictedClass= self.model.classify_text(data)
            if predictedClass==0:
                # Step 1: Process with Service1
                request1 = JsonRequest(json_data=json_data)
                response1 = self.stub1.ProcessJson(request1)
                logger.debug(f"respones1: {response1}")
                if not response1.success:
                    return {
                        "error": f"Service1 processing failed: {response1.message}",
                        "success": False
                    }
                
                result={
                    "json_data": response1.json_data,
                    "success": response1.success,
                    "message": response1.message
                }
            else:
                # Step 2: Process with Service2
                request2 = JsonRequest(json_data=json_data)
                response2 = self.stub2.ProcessJson(request2)
                
                if not response2.success:
                    return {
                        "error": f"Service2 processing failed: {response2.message}",
                        "success": False
                }

                result={
                    "json_data": response2.json_data,
                    "success": response2.success,
                    "message": response2.message
                }

            result["success"] = True
            return result
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON format: {str(e)}",
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Main service error: {str(e)}",
                "success": False
            }
    
    def close(self):
        self.channel1.close()
        self.channel2.close()
        print("JSON Main Service connections closed")
#rakuten_python main.py
# Example usage (run.py)
if __name__ == '__main__':
    test=CopaDataset().randomData()
    # Create and use the main service
    main_service = MainService()
    try:
        result = main_service.process_json(test)
        print(result)
    finally:
        main_service.close()