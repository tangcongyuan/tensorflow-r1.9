#include "tensorflow/core/common_runtime/epu/epu_device.h"

namespace tensorflow {

EPUDevice::EPUDevice(const SessionOptions& options, const string& name,
                     Bytes memory_limit, const DeviceLocality& locality,
                     const string& physical_device_desc,
                     EPUAllocator* epu_allocator, Allocator* cpu_allocator,
                     EPUDeviceContext* ctx)
    : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_EPU,
                                                         memory_limit, locality,
                                                         physical_device_desc)),
      cpu_allocator_(cpu_allocator),
      epu_allocator_(epu_allocator),
      device_context_(ctx) {
  // Do I need to free these contexts later?
  ;
}

EPUDevice::~EPUDevice() { delete epu_allocator_; }

Allocator* EPUDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host())
    return this->cpu_allocator_;
  else
    return this->epu_allocator_;
}

void EPUDevice::Compute(OpKernel* op_kernel,
                        OpKernelContext* context) {
  op_kernel->Compute(context);
}

Status EPUDevice::FillContextMap(
    const Graph* graph, DeviceContextMap* device_context_map) {
  printf("---FillContextMap()---\n");
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    printf("===node n===\n");
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }
  return Status::OK();
}

Status EPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(GetAllocator(alloc_attrs), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
  }
  *tensor = parsed;
  return Status::OK();
}

Status EPUDevice::Sync() { return Status::OK(); }
}

