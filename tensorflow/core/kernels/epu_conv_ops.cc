#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/conv_ops.h"

#include <string.h>
#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/epu/epu_device.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/deep_conv2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
#include "tensorflow/core/kernels/xsmm_conv2d.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class Conv2DEpuOp : public BinaryOp<T> {
 public:
  explicit Conv2DEpuOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    const int64 stride_h = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    const int64 dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    const int64 dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    const int64 dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    const int64 dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    VLOG(0) << "\33[42m Enter " << this->name() << " Compute \33[0m";
    const Tensor& input = context->input(0);
    
    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth or be evenly divisible by filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    const int64 patch_depth = filter.dim_size(2);
    OP_REQUIRES(context, in_depth % patch_depth == 0,
                errors::InvalidArgument(
                    "input depth must be evenly divisible by filter depth: ",
                    in_depth, " vs ", patch_depth));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride and dilation from the second and third
    // dimensions only (we do not support striding or dilation on the batch or
    // depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_rows, filter_rows, dilation_rows,
                                stride_rows, padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_cols, filter_cols, dilation_cols,
                                stride_cols, padding_, &out_cols, &pad_cols));
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";
    VLOG(0) << "\33[1;33m ============================================ \33[0m\n";


    VLOG(0) << "\33[1;33m ================== in EPU Conv2D Op =============== \33[0m\n"
            << "Conv2D: in_depth = " << in_depth
            << ", patch_depth = " << patch_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", dilation_rows = " << dilation_rows
            << ", dilation_cols = " << dilation_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // We only support same stride along rows/cols dimension
    assert (stride_rows == stride_cols);

    printf("input.TotalBytes() = %lu\n", input.TotalBytes());
    printf("filter.TotalBytes() = %lu\n", filter.TotalBytes());
    printf("output->TotalBytes() = %lu\n", output->TotalBytes());

    const auto input_ptr = input.flat<T>().data();
    const auto weight_ptr = filter.flat<T>().data();
    auto output_ptr = output->flat<T>().data();

    // TODO(pwchou): we use nullptr as bias_ptr and output as 2nd input temporarily.
    const auto bias_ptr = nullptr;
    const auto second_input_ptr = output->flat<T>().data();

//     RunEpuConvolution(input_ptr, weight_ptr, output_ptr, bias_ptr,
//                       second_input_ptr, input_rows, input_cols, in_depth,
//                       filter_rows, filter_cols, pad_rows, pad_cols, out_rows,
//                       out_cols, out_depth, stride_rows,
//                       /*relu=*/false, /*maxpool=*/false);

    /*
#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
    if (LaunchXsmmConvOp<Device, T>::Run(
            context, input, filter, batch, input_rows, input_cols, in_depth,
            filter_rows, filter_cols, pad_rows, pad_cols, out_rows, out_cols,
            out_depth, dilation_rows, dilation_cols, stride_rows, stride_cols,
            output, data_format_)) {
      return;
    }
#endif

    if (LaunchDeepConvOp<Device, T>::Run(
            context, input, filter, batch, input_rows, input_cols, in_depth,
            filter_rows, filter_cols, pad_rows, pad_cols, out_rows, out_cols,
            out_depth, dilation_rows, dilation_cols, stride_rows, stride_cols,
            output, data_format_)) {
      return;
    }

    launcher_(context, use_cudnn_, cudnn_use_autotune_, input, filter,
              dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
              output, data_format_);
              */
  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DEpuOp);
};

#define REGISTER_EPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv2D").Device(DEVICE_EPU).TypeConstraint<T>("T"), \
      Conv2DEpuOp<EPUDevice, T>);

TF_CALL_half(REGISTER_EPU);

}
    
