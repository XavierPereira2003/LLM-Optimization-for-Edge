
import grpc
from concurrent import futures
import time
import json
from services_pb2 import JsonRequest, JsonResponse
from services_pb2_grpc import JsonProcessor1Servicer, add_JsonProcessor1Servicer_to_server
from Models.Boolq import BoolqModel
import logging
log_path="./logs/"
 
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/Service.log"),
    ]
)
logger=logging.getLogger("BoolqService")

class BoolqService(JsonProcessor1Servicer):
    def __init__(self) -> None:
        self.model=BoolqModel()
        logger.info("Boolq Model is loaded")

    def ProcessJson(self, request, context):
        logger.log("Request Process is Initiated")
        try:
            # Validate and parse JSON data
            if not request.json_data:
                raise ValueError("Empty JSON data received")
            
            data = json.loads(request.json_data)
            logger.debug(f"The input{data}")
            prompt=data['prompt']
            ouptut=self.model.predict(prompt)
            # Process the JSON data
            Return_Data = {
                "metadata": {
                    "processor": "Boolq Service",
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "processing_id": str(time.time())
                },
                "original_data": data,
                "Output":ouptut
            }
            logger.debug(f" The output:{Return_Data}")
            return JsonResponse(
                json_data=json.dumps(Return_Data),
                success=True,
                message="JSON successfully processed by Service1"
            )
        except json.JSONDecodeError as e:
            return JsonResponse(
                json_data="",
                success=False,
                message=f"Invalid JSON format: {str(e)}"
            )
        except Exception as e:
            return JsonResponse(
                json_data="",
                success=False,
                message=f"Service1 processing error: {str(e)}"
            )

def serve():
    logger.info("BoolqService is Running")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_JsonProcessor1Servicer_to_server(BoolqService(), server)
    server.add_insecure_port('[::]:50051')
    print("JSON Service 1 starting on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    print("Running....")
    serve()