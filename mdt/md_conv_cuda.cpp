
#include <torch/extension.h>


torch::Tensor md_conv_cuda_forward(
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	const int kernel_d,
	const int kernel_h,
	const int kernel_w,
	const int stride_d,
	const int stride_h,
	const int stride_w,
	const int pad_d,
	const int pad_h,
	const int pad_w,
	const int a1, const int a2,
	const int b1, const int b2);

std::vector<torch::Tensor> md_conv_cuda_backward(
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	const torch::Tensor& grad_output,
	const int kernel_d,
	const int kernel_h,
	const int kernel_w,
	const int stride_d,
	const int stride_h,
	const int stride_w,
	const int pad_d,
	const int pad_h,
	const int pad_w,
	const int a1, const int a2,
	const int b1, const int b2);

// C++ interface


torch::Tensor md_conv_forward(
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	const int kernel_d,
	const int kernel_h,
	const int kernel_w,
	const int stride_d,
	const int stride_h,
	const int stride_w,
	const int pad_d,
	const int pad_h,
	const int pad_w,
	const int a1, const int a2,
	const int b1, const int b2
	) {

	return md_conv_cuda_forward(
		input, weight, bias,
		kernel_d, kernel_h, kernel_w,
		stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
		a1,a2,b1,b2
	);

}
std::vector<torch::Tensor> md_conv_backward(
	const torch::Tensor& input,
	const torch::Tensor& weight,
	const torch::Tensor& bias,
	const torch::Tensor& grad_output,
	const int kernel_d,
	const int kernel_h,
	const int kernel_w,
	const int stride_d,
	const int stride_h,
	const int stride_w,
	const int pad_d,
	const int pad_h,
	const int pad_w,
	const int a1, const int a2,
	const int b1, const int b2) {

	return md_conv_cuda_backward(
		input, weight, bias, grad_output,
		kernel_d, kernel_h, kernel_w,
		stride_d, stride_h, stride_w,
		pad_d, pad_h, pad_w,
		a1,a2,b1,b2);

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &md_conv_forward, "md_conv forward (CUDA)");
	m.def("backward", &md_conv_backward, "md_conv backward (CUDA)");
}

