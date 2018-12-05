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
  
  class EPUAllocator : public Allocator {
    public:
      EPUAllocator();

      ~EPUAllocator() override;
      
      string Name() override;
      
      /* The void* could be a handle to a hardware allocation descriptor
      * See tensorflow/stream_executor/device_memory.h
      */
      void* AllocateRaw(size_t alignnment, size_t num_bytes) override; 
      
      void DeallocateRaw(void* ptr) override;

    private:
      TF_DISALLOW_COPY_AND_ASSIGN(EPUAllocator);
  };

}

