#include <vector>
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

#include "tensorflow/core/common_runtime/epu/epu_allocator.h"
#include "tensorflow/core/common_runtime/epu/epu_device_context.h"

namespace tensorflow {
  
  class EPUDevice : public LocalDevice {
  public:
    EPUDevice(const SessionOptions& options, const string& name, Bytes memory_limit, const DeviceLocality& locality, const string& physical_device_desc,
	      EPUAllocator *epu_allocator, Allocator *cpu_allocator, EPUDeviceContext* ctx);

    ~EPUDevice() override;
      
    Allocator* GetAllocator(AllocatorAttributes attr) override;
    
    void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
    
    Status FillContextMap(const Graph *graph, DeviceContextMap *device_context_map) override;

    Status MakeTensorFromProto(const TensorProto& tensor_proto,
			       const AllocatorAttributes alloc_attrs,
			       Tensor* tensor) override;
    
    Status Sync() override;
    
  protected:
    Allocator *epu_allocator_;
    Allocator *cpu_allocator_;  // Not owned
    EPUDeviceContext* device_context_;
  };

}
