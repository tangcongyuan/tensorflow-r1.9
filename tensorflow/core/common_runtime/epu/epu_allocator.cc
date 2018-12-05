#include "tensorflow/core/common_runtime/epu/epu_allocator.h"

namespace tensorflow {
  
EPUAllocator::EPUAllocator() {}
EPUAllocator::~EPUAllocator() {}

string EPUAllocator::Name() { return "device:EPU"; }

/* The void* could be a handle to a hardware allocation descriptor
* See tensorflow/stream_executor/device_memory.h
*/
void* EPUAllocator::AllocateRaw(size_t alignnment, size_t num_bytes) {
  void* p = port::AlignedMalloc(num_bytes, alignnment);
  return p;
}

void EPUAllocator::DeallocateRaw(void* ptr) {
  port::AlignedFree(ptr);
}

}

