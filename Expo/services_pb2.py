# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: services.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'services.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0eservices.proto\x12\x08services\" \n\x0bJsonRequest\x12\x11\n\tjson_data\x18\x01 \x01(\t\"C\n\x0cJsonResponse\x12\x11\n\tjson_data\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t2N\n\x0eJsonProcessor1\x12<\n\x0bProcessJson\x12\x15.services.JsonRequest\x1a\x16.services.JsonResponse2N\n\x0eJsonProcessor2\x12<\n\x0bProcessJson\x12\x15.services.JsonRequest\x1a\x16.services.JsonResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'services_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_JSONREQUEST']._serialized_start=28
  _globals['_JSONREQUEST']._serialized_end=60
  _globals['_JSONRESPONSE']._serialized_start=62
  _globals['_JSONRESPONSE']._serialized_end=129
  _globals['_JSONPROCESSOR1']._serialized_start=131
  _globals['_JSONPROCESSOR1']._serialized_end=209
  _globals['_JSONPROCESSOR2']._serialized_start=211
  _globals['_JSONPROCESSOR2']._serialized_end=289
# @@protoc_insertion_point(module_scope)