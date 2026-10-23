"""CUDA driver structures used to construct graph fixtures independently of export."""

import ctypes as C


class KernelParams(C.Structure):
    _fields_ = [("func", C.c_void_p), ("grid", C.c_uint * 3), ("block", C.c_uint * 3),
                ("shared", C.c_uint), ("params", C.POINTER(C.c_void_p)),
                ("extra", C.POINTER(C.c_void_p)), ("kern", C.c_void_p), ("ctx", C.c_void_p)]


class MemsetParams(C.Structure):
    _fields_ = [("dst", C.c_uint64), ("pitch", C.c_size_t), ("value", C.c_uint),
                ("element_size", C.c_uint), ("width", C.c_size_t), ("height", C.c_size_t)]
