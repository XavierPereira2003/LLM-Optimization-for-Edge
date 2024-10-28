import grpc
from concurrent import futures
import time
import json
from services_pb2 import JsonRequest, JsonResponse
from services_pb2_grpc import JsonProcessor2Servicer, add_JsonProcessor2Servicer_to_server
from Models.CoPA import CopaModel
import logging
 
logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/Service.log"),
    ]
)
logger=logging.getLogger("CopaService")
class CopaService(JsonProcessor2Servicer):
    def __init__(self) -> None:
        self.model=CopaModel()
        logger.info("CoPA Model is loaded")
    def ProcessJson(self, request, context):
        try:
            # Validate and parse JSON data from Service1
            if not request.json_data:
                raise ValueError("Empty JSON data received")
            
            data = json.loads(request.json_data)
            
            prompt=data['prompt']
            ouptut=self.model.predict(prompt)
            # Process the JSON data
            Return_Data = {
                "metadata": {
                    "processor": "CoPa   Service",
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "processing_id": str(time.time())
                },
                "original_data": data,
                "Output":ouptut
            }
            Return_Data=json.dumps(Return_Data)
            logger.debug(f"Return_data: {Return_Data}")
            return JsonResponse(
                json_data=Return_Data,
                success=True,
                message="JSON successfully processed by Service2"
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
                message=f"Service2 processing error: {str(e)}"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_JsonProcessor2Servicer_to_server(CopaService(), server)
    server.add_insecure_port('[::]:50052')
    print("JSON Service 2 starting on port 50052...")
    server.start()
    server.wait_for_termination()


#rakuten_python copa_service.py

if __name__ == '__main__':
    print("Running....")
    serve()