#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
  
  class EPUDeviceContext : public DeviceContext {
    void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
			       Tensor* device_tensor,
			       StatusCallback done) const override;

    void CopyDeviceTensorToCPU(const Tensor* device_tensor,
			       StringPiece tensor_name, Device* device,
			       Tensor* cpu_tensor, StatusCallback done) override;
  };

}

