# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import services_pb2 as services__pb2

GRPC_GENERATED_VERSION = '1.67.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in services_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class JsonProcessor1Stub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessJson = channel.unary_unary(
                '/services.JsonProcessor1/ProcessJson',
                request_serializer=services__pb2.JsonRequest.SerializeToString,
                response_deserializer=services__pb2.JsonResponse.FromString,
                _registered_method=True)


class JsonProcessor1Servicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessJson(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JsonProcessor1Servicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessJson': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessJson,
                    request_deserializer=services__pb2.JsonRequest.FromString,
                    response_serializer=services__pb2.JsonResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'services.JsonProcessor1', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('services.JsonProcessor1', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JsonProcessor1(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessJson(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/services.JsonProcessor1/ProcessJson',
            services__pb2.JsonRequest.SerializeToString,
            services__pb2.JsonResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class JsonProcessor2Stub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessJson = channel.unary_unary(
                '/services.JsonProcessor2/ProcessJson',
                request_serializer=services__pb2.JsonRequest.SerializeToString,
                response_deserializer=services__pb2.JsonResponse.FromString,
                _registered_method=True)


class JsonProcessor2Servicer(object):
    """Missing associated documentation comment in .proto file."""

    def ProcessJson(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_JsonProcessor2Servicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessJson': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessJson,
                    request_deserializer=services__pb2.JsonRequest.FromString,
                    response_serializer=services__pb2.JsonResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'services.JsonProcessor2', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('services.JsonProcessor2', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class JsonProcessor2(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ProcessJson(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/services.JsonProcessor2/ProcessJson',
            services__pb2.JsonRequest.SerializeToString,
            services__pb2.JsonResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)