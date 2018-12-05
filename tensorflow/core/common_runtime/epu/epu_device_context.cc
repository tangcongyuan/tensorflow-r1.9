#include "tensorflow/core/common_runtime/epu/epu_device_context.h"

namespace tensorflow {
  
void EPUDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
               Tensor* device_tensor,
               StatusCallback done) const {  
  *device_tensor = *cpu_tensor;
  done(Status::OK());
}

void EPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
               StringPiece tensor_name, Device* device,
               Tensor* cpu_tensor, StatusCallback done) {
  *cpu_tensor = *device_tensor;
  done(Status::OK());
}

};

