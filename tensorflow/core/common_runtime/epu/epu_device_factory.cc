#include "tensorflow/core/common_runtime/epu/epu_device_factory.h"
#include "tensorflow/core/common_runtime/epu/epu_device.h"

namespace tensorflow {

Status EPUDeviceFactory::CreateDevices(const SessionOptions& options,
                                       const string& name_prefix,
                                       std::vector<Device*>* devices) {
  int n = INT_MAX;
  auto iter = options.config.device_count().find("EPU");
  if (iter != options.config.device_count().end()) {
    n = iter->second;
  }
  std::vector<int> valid_epu_ids;
  GetValidDeviceIds(&valid_epu_ids);
  if (static_cast<size_t>(n) > valid_epu_ids.size()) {
    n = valid_epu_ids.size();
  }
  for (int i = 0; i < n; i++) {
    devices->push_back(new EPUDevice(
        options, strings::StrCat(name_prefix, "/device:EPU:", i),
        Bytes(256 << 20), DeviceLocality(), "test epu", new EPUAllocator(),
        cpu_allocator(), new EPUDeviceContext()));
  }
  return Status::OK();
}

void EPUDeviceFactory::GetValidDeviceIds(std::vector<int>* ids) {
  for (int i = 0; i < 1; i++) ids->push_back(i);
}

REGISTER_LOCAL_DEVICE_FACTORY("EPU", EPUDeviceFactory);
}

