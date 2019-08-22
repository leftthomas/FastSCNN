#include <vector>
#include <torch/extension.h>
#include<ATen/ATen.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "md_vol2col_cuda.cuh"


at::Tensor md_conv_cuda_forward(
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
	const int b1, const int b2)
{

	AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

	const torch::Tensor input_ = input.is_contiguous() ? input : input.contiguous();

	AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
	AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
	AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");

    const int group = 1;
	const int batch = input.size(0);
	const int channels = input.size(1);
	const int depth = input.size(2);
	const int height = input.size(3);
	const int width = input.size(4);

	const int channels_out = weight.size(0);
	const int channels_kernel = weight.size(1);
	const int kernel_d_ = weight.size(2);
	const int kernel_h_ = weight.size(3);
	const int kernel_w_ = weight.size(4);


	AT_ASSERTM(kernel_d_ == kernel_d && kernel_h_ == kernel_h && kernel_w_ == kernel_w,
		"Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).", kernel_d_, kernel_h_, kernel_w_, kernel_d, kernel_h, kernel_w);


	int kh = std::abs(a1);
	int kw = std::abs(b1);
	kh = kh > 0 ? kh + 1 : 0;
	kw = kw > 0 ? kw + 1 : 0;

	const int depth_out = (depth + 2 * pad_d - kernel_d)  / stride_d + 1;
	const int height_out = (height + 2 * pad_h - kernel_h-kh ) / stride_h + 1;
	const int width_out = (width + 2 * pad_w -kernel_w-kw) / stride_w + 1;

	const int kernel_dhw = kernel_d * kernel_h * kernel_w;
	const int out_dhw = depth_out * height_out * width_out;

	auto output = at::empty({ batch * out_dhw, channels_out }, input.options());
	auto output_n = output.view({ batch ,out_dhw, channels_out });

	auto weight_g = weight.view({ group, channels_out / group, channels_kernel,kernel_d, kernel_h, kernel_w });
	auto bias_g = bias.view({ group, channels_out / group });

	auto columns = at::empty({ channels * kernel_dhw, out_dhw }, input.options());
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		input.scalar_type(), "md_down_conv_forward<>", [&] {
			// For each elt in batch, do:
			for (int n = 0; n < batch; n++) {
				// Matrix multiply per output:
				auto input_n = input_.select(0, n);
				//Tensor output_n = output.select(0, n);
				vol2col(stream,
					input_n.data<scalar_t>(),
					channels,
					depth, height, width,
					depth_out, height_out, width_out,
					kernel_d, kernel_h, kernel_w,
					pad_d, pad_h, pad_w,
					stride_d, stride_h, stride_w,
					a1,a2,b1,b2,
					columns.data<scalar_t>()
				);

				auto columns_g = columns.view({ group, channels / group * kernel_dhw, out_dhw });

				auto output_g = output_n.select(0, n).view({ out_dhw, group, channels_out / group });

				for (int g = 0; g < group; ++g)
				{
					auto columns_gm = columns_g.select(0, g).t();

					auto weight_gm = weight_g.select(0, g).view({ channels_out / group, channels_kernel * kernel_dhw }).t();

					auto output_m = at::addmm(bias_g.select(0, g), columns_gm, weight_gm);

					output_g.select(1, g) = output_m.view({ out_dhw, channels_out / group });
				}
			}
		});

	output = output.view({ batch, depth_out, height_out, width_out, channels_out }).permute({ 0, 4, 1, 2, 3 }).contiguous();
	return output;
}







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
	const int b1, const int b2)
{

	AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");
	const torch::Tensor input_ = input.is_contiguous() ? input : input.contiguous();
	const torch::Tensor grad_output_ = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

	AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
	AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
	AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");

	const int batch = input.size(0);
	const int channels = input.size(1);
	const int depth = input.size(2);
	const int height = input.size(3);
	const int width = input.size(4);

	const int channels_out = weight.size(0);
	const int channels_kernel = weight.size(1);
	const int kernel_d_ = weight.size(2);
	const int kernel_h_ = weight.size(3);
	const int kernel_w_ = weight.size(4);

	AT_ASSERTM(kernel_d_ == kernel_d && kernel_h_ == kernel_h && kernel_w_ == kernel_w,
		"Input shape and kernel shape wont match: (%d x %d x %d vs %d x %d x %d).", kernel_d_, kernel_h_, kernel_w_, kernel_d, kernel_h, kernel_w);
    const int group = 1;
	int kh = std::abs(a1);
	int kw = std::abs(b1);
	kh = kh > 0 ? kh + 1 : 0;
	kw = kw > 0 ? kw + 1 : 0;

	const int depth_out = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
	const int height_out = (height + 2 * pad_h - kernel_h - kh) / stride_h + 1;
	const int width_out = (width + 2 * pad_w - kernel_w - kw) / stride_w + 1;

	const int kernel_dhw = kernel_d * kernel_h * kernel_w;
	const int out_dhw = depth_out * height_out * width_out;

	auto options = input.options();
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	// grad
	auto grad_input = at::zeros_like(input, options);
	auto grad_weight = at::zeros_like(weight, options);
	auto grad_bias = at::zeros_like(bias, options);

	auto weight_g = weight.view({ group, channels_out / group, channels_kernel, kernel_d, kernel_h, kernel_w });
	auto grad_weight_g = grad_weight.view({ group, channels_out / group, channels_kernel, kernel_d, kernel_h, kernel_w });
	auto grad_bias_g = grad_bias.view({ group, channels_out / group });



	auto columns = at::empty({ channels * kernel_dhw, out_dhw }, options);
	//auto grad_output_n = grad_output.view({ batch , channels_out, depth_out, height_out, width_out });

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
		input.scalar_type(), "md_conv_backward<>", [&] {
			// For each elt in batch, do:
			for (int n = 0; n < batch; n++) {
				auto ones = at::ones({ out_dhw }, options);
				auto input_n = input_.select(0, n);
				auto grad_input_n = grad_input.select(0, n);

				auto grad_output_g = grad_output_.select(0, n).view({ group, channels_out / group, depth_out, height_out, width_out });



				auto columns_g = columns.view({ group, channels / group * kernel_dhw, out_dhw });
				for (int g = 0; g < group; ++g)
				{
					auto grad_output_gm = grad_output_g.select(0, g).view({ channels_out / group, out_dhw });
					auto weight_gm = weight_g.select(0, g).view({ channels_out / group, channels_kernel * kernel_dhw }).t();
					columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);
				}


				//std::cout << columns << std::endl;
				//std::cout << grad_input_n << std::endl;

				col2vol(stream,
					columns.data<scalar_t>(),
					channels,
					depth, height, width,
					depth_out, height_out, width_out,
					kernel_d, kernel_h, kernel_w,
					pad_d, pad_h, pad_w,
					stride_d, stride_h, stride_w,
					a1, a2, b1, b2,
					grad_input_n.data<scalar_t>());

				//std::cout << grad_input_n <<std::endl;


				vol2col(stream,
					input_n.data<scalar_t>(),
					channels,
					depth, height, width,
					depth_out, height_out, width_out,
					kernel_d, kernel_h, kernel_w,
					pad_d, pad_h, pad_w,
					stride_d, stride_h, stride_w,
					a1,a2,b1,b2,
					columns.data<scalar_t>()
				);

				//std::cout << columns << std::endl;

				for (int g = 0; g < group; ++g)
				{
					auto grad_output_gm = grad_output_g.select(0, g).view({ channels_out / group, out_dhw });


					auto columns_gm = columns_g.select(0, g).t();
					auto grad_weight_gm = grad_weight_g.select(0, g).view({ channels_out / group, channels_kernel * kernel_dhw });
					auto grad_bias_gm = grad_bias_g.select(0, g);


					grad_weight_g.select(0, g) = at::addmm(grad_weight_gm, grad_output_gm, columns_gm).view_as(grad_weight_g.select(0, g));

					grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, ones);
				}
			}
		});
	return { grad_input, grad_weight, grad_bias };
}
