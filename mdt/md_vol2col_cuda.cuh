
#include "cuda_runtime.h"
//#include<cublas_v2.h>
#include <cuda.h>
#include<torch/extension.h>



#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
	AT_ASSERTM(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
	return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


template <typename T>
__global__ void vol2col_kernel(
	const int n,
	const T* data_vol,
	const int depth,
	const int height,
	const int width,
	const int ksize_t,
	const int ksize_h,
	const int ksize_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int depth_col,
	const int height_col,
	const int width_col,
	const int a1, const int a2,
	const int b1, const int b2,
	T* data_col) {
	CUDA_KERNEL_LOOP(index, n) {
		int w_out = index % width_col;
		index /= width_col;
		int h_out = index % height_col;
		index /= height_col;
		int t_out = index % depth_col;
		int channel_in = index / depth_col;
		int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
		int t_in = t_out * stride_t - pad_t;
		int h_in = h_out * stride_h - pad_h;
		int w_in = w_out * stride_w - pad_w;
		int col_len = depth_col * height_col * width_col;
		data_col +=
			((channel_out * depth_col + t_out) * height_col + h_out) * width_col +
			w_out;
		int t, h, w;
		data_vol += ((channel_in * depth + t_in) * height + h_in) * width + w_in;
		for (int i = 0; i < ksize_t; ++i) {
			for (int j = 0; j < ksize_h; ++j) {
				for (int k = 0; k < ksize_w; ++k) {
					t = t_in + i;
					h = h_in + a1 * i + a2;
					w = w_in + b1 * i + b2;
					*data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height &&
						w < width)
						? data_vol
						[((height+a1) * i + a2 ) *width +b1 * i + b2]
					: static_cast<T>(0);
					data_col += col_len;
				}
			}
		}
	}
}

template <typename T>
void vol2col(
	cudaStream_t stream,
	const T* data_vol,
	const int channels,
	const int depth,
	const int height,
	const int width,
	const int depth_col,
	const int height_col,
	const int width_col,
	const int ksize_t,
	const int ksize_h,
	const int ksize_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int a1, const int a2,
	const int b1, const int b2,
	T* data_col) {
	// We are going to launch channels * depth_col * height_col * width_col
	// kernels, each kernel responsible for copying a single-channel grid.
	int num_kernels = channels * depth_col * height_col * width_col;
	// Launch
	vol2col_kernel << <GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream >> > (
		num_kernels,
		data_vol,
		depth,
		height,
		width,
		ksize_t,
		ksize_h,
		ksize_w,
		pad_t,
		pad_h,
		pad_w,
		stride_t,
		stride_h,
		stride_w,
		depth_col,
		height_col,
		width_col,
		a1, a2, b1, b2,
		data_col);
	AT_CUDA_CHECK(cudaGetLastError());
}




template <typename T>
__global__ void vol2im_kernel_h(
	const int n,
	const T* data_col,
	const int depth,
	const int height,
	const int width,
	const int channels,
	const int kernel_t,
	const int kernel_h,
	const int kernel_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int depth_col,
	const int height_col,
	const int width_col,
	const int a, const int b,
	T* data_vol) {
	CUDA_KERNEL_LOOP(index, n) {
		T val = 0;
		int absa = a > 0 ? a : -a;
		const int w_im = index % width + pad_w;
		const int h_im = (index / width) % height + pad_h;
		const int t_im = (index / width / height) % depth + pad_t;
		const int c_im = index / (width * height * depth);
		int kernel_extent_w = kernel_w ;
		int kernel_extent_h = kernel_h + absa + 1;
		int kernel_extent_t = kernel_t;
		// compute the start and end of the output
		const int w_col_start =
			(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
		const int w_col_end = w_im / stride_w + 1 < width_col ? w_im / stride_w + 1 : width_col;
		const int h_col_start =
			(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
		const int h_col_end = h_im / stride_h + 1 < height_col ? h_im / stride_h + 1 : height_col;
		const int t_col_start =
			(t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
		const int t_col_end = t_im / stride_t + 1 < depth_col ? t_im / stride_t + 1 : depth_col;
		// TODO: use LCM of stride and dilation to avoid unnecessary loops
		int t_k, h_k, w_k,data_col_index;
		for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
			for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
				for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
					t_k = t_im - t_col * stride_t;
					h_k = h_im - h_col * stride_h;
					w_k = w_im - w_col * stride_w;
					data_col_index =
						(((((c_im * kernel_t + t_k) * kernel_h + 0) * kernel_w +
							w_k) *
							depth_col +
							t_col) *
							height_col +
							h_col) *
						width_col +
						w_col;
					val += ((t_k * a + b - h_k) ? static_cast<T>(0) : data_col[data_col_index]);

				}
			}
		}
		data_vol[index] = val;
	}
}

template <typename T>
__global__ void vol2im_kernel_w(
	const int n,
	const T* data_col,
	const int depth,
	const int height,
	const int width,
	const int channels,
	const int kernel_t,
	const int kernel_h,
	const int kernel_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int depth_col,
	const int height_col,
	const int width_col,
	const int a, const int b,
	T* data_vol) {
	CUDA_KERNEL_LOOP(index, n) {
		T val = 0;
		int absa = a > 0 ? a : -a;
		const int w_im = index % width + pad_w;
		const int h_im = (index / width) % height + pad_h;
		const int t_im = (index / width / height) % depth + pad_t;
		const int c_im = index / (width * height * depth);
		int kernel_extent_w = kernel_w + absa+ 1;
		int kernel_extent_h = kernel_h;
		int kernel_extent_t = kernel_t;
		// compute the start and end of the output
		const int w_col_start =
			(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
		const int w_col_end = w_im / stride_w + 1 < width_col ? w_im / stride_w + 1 : width_col;
		const int h_col_start =
			(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
		const int h_col_end = h_im / stride_h + 1 < height_col ? h_im / stride_h + 1 : height_col;
		const int t_col_start =
			(t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
		const int t_col_end = t_im / stride_t + 1 < depth_col ? t_im / stride_t + 1 : depth_col;
		// TODO: use LCM of stride and dilation to avoid unnecessary loops
		for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
			for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
				for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
					int t_k = t_im - t_col * stride_t;
					int h_k = h_im - h_col * stride_h;
					int w_k = w_im - w_col * stride_w;
					int data_col_index =
						(((((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w +
							0) *
							depth_col +
							t_col) *
							height_col +
							h_col) *
						width_col +
						w_col;
					val += ((t_k * a + b - w_k) ? static_cast<T>(0) : data_col[data_col_index]);

				}
			}
		}
		data_vol[index] = val;
	}
}


template <typename T>
void col2vol(
	cudaStream_t stream,
	const T* data_col,
	const int channels,
	const int depth,
	const int height,
	const int width,
	const int output_depth,
	const int output_height,
	const int output_width,
	const int patch_t,
	const int patch_h,
	const int patch_w,
	const int pad_t,
	const int pad_h,
	const int pad_w,
	const int stride_t,
	const int stride_h,
	const int stride_w,
	const int a1, const int a2,
	const int b1, const int b2,
	T* data_vol) {
	int num_kernels = channels * depth * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	if (a1 == 0 && a2 == 0) {
		vol2im_kernel_w<T>
			<< <GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream >> > (
				num_kernels,
				data_col,
				depth,
				height,
				width,
				channels,
				patch_t,
				patch_h,
				patch_w,
				pad_t,
				pad_h,
				pad_w,
				stride_t,
				stride_h,
				stride_w,
				output_depth,
				output_height,
				output_width,
				b1, b2,
				data_vol);
	}
	else {
		vol2im_kernel_h<T>
			<< <GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream >> > (
				num_kernels,
				data_col,
				depth,
				height,
				width,
				channels,
				patch_t,
				patch_h,
				patch_w,
				pad_t,
				pad_h,
				pad_w,
				stride_t,
				stride_h,
				stride_w,
				output_depth,
				output_height,
				output_width,
				a1, a2,
				data_vol);
	}
	AT_CUDA_CHECK(cudaGetLastError());
}