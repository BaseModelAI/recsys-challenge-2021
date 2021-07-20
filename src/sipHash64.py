import ctypes
from ctypes import cdll


lib = cdll.LoadLibrary('./libsiphash64.so')
lib.sipHash64.restype = ctypes.c_ulonglong


def sipHash64(s):
    st = s.encode('ascii', errors='ignore')
    return lib.sipHash64(st, len(st))
