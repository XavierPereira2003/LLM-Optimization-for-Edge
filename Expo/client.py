import grpc
import json
import services_pb2
import services_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50053') as channel:
        stub = services_pb2_grpc.MainServiceStub(channel)
        # Send a JSON data request
        data = {"message": "Hello, services!"}
        request = services_pb2.DataRequest(json_data=json.dumps(data))
        response = stub.AggregateData(request)
        print("MainService response:", response.result)

if __name__ == "__main__":  
    run()
