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

namespace tensorflow {

  class EPUDeviceFactory : public DeviceFactory {
  public:
    Status CreateDevices(const SessionOptions& options,
		       const string& name_prefix,
		       std::vector<Device*>* devices) override; 
    
  private:
    void GetValidDeviceIds(std::vector<int>* ids);
  };

}

