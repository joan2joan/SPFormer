source = '''
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 256 

#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh> 
#include <ATen/cuda/DeviceUtils.cuh> 

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ color_dist,
    scalar_t* __restrict__ contrast_dist,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    const int color_channels = channels - 1; // 最后一维是对比度特征
    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;
    int init_spix_index = spixel_indices[b * num_pixels + p];

    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);
    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w ||
        y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) {
        color_dist[index] = 1e16;
        contrast_dist[index] = 1e16;
        return;
    }

    // 计算颜色距离（L2 norm）
    scalar_t color_distance = 0;
    for (int c = 0; c < color_channels; ++c) {
        scalar_t diff = pixel_features[b * cp + c * num_pixels + p] - 
                       spixel_features[b * cs + c * num_spixels + init_spix_index];
        color_distance += diff * diff;
    }
    color_dist[index] = color_distance;

    // 计算对比度距离（L2 norm）
    scalar_t contrast_diff = pixel_features[b * cp + color_channels * num_pixels + p] - 
                           spixel_features[b * cs + color_channels * num_spixels + init_spix_index];
    scalar_t contrast_distance = contrast_diff * contrast_diff;
    contrast_dist[index] = contrast_distance;
}

std::vector<torch::Tensor> forward_cuda(
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor spixel_indices,
    int num_spixels_w, int num_spixels_h
){
    int batchsize = pixel_features.size(0); 
    int channels = pixel_features.size(1); 
    int num_pixels = pixel_features.size(2); 
    int num_spixels = spixel_features.size(2); 

    auto color_dist = torch::zeros({batchsize, 9,num_pixels}, pixel_features.options()); 
    auto contrast_dist = torch::zeros_like(color_dist);

    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(pixel_features.type(),  "forward_kernel", ([&] {
        forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            pixel_features.data_ptr<scalar_t>(), 
            spixel_features.data_ptr<scalar_t>(), 
            spixel_indices.data_ptr<scalar_t>(), 
            color_dist.data_ptr<scalar_t>(), 
            contrast_dist.data_ptr<scalar_t>(), 
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h 
        );
    }));

    return {color_dist, contrast_dist};
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ color_dist_grad,
    const scalar_t* __restrict__ contrast_dist_grad,
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ pixel_feature_grad,
    scalar_t* __restrict__ spixel_feature_grad,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    const int color_channels = channels - 1;
    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;

    int init_spix_index = spixel_indices[b * num_pixels + p];

    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);

    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w ||
        y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) {
        return;
    }

    // 计算目标超像素索引 
    int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;

    // 获取梯度值 
    scalar_t color_grad_val = color_dist_grad[index];
    scalar_t contrast_grad_val = contrast_dist_grad[index];

    // 颜色通道梯度计算 (L2 norm的导数)
    for (int c = 0; c < color_channels; ++c) {
        scalar_t diff = pixel_features[b * cp + c * num_pixels + p] - 
                       spixel_features[b * cs + c * num_spixels + query_spixel_index];

        scalar_t grad = 2 * diff * color_grad_val;
        atomicAdd(&pixel_feature_grad[b * cp + c * num_pixels + p], grad);
        atomicAdd(&spixel_feature_grad[b * cs + c * num_spixels + query_spixel_index], -grad);
    }

    // 对比度通道梯度计算 (L2 norm的导数)
    scalar_t contrast_diff = pixel_features[b * cp + color_channels * num_pixels + p] - 
                           spixel_features[b * cs + color_channels * num_spixels + query_spixel_index];

    scalar_t contrast_grad = 2 * contrast_diff * contrast_grad_val;
    atomicAdd(&pixel_feature_grad[b * cp + color_channels * num_pixels + p], contrast_grad);
    atomicAdd(&spixel_feature_grad[b * cs + color_channels * num_spixels + query_spixel_index], -contrast_grad);
}

std::vector<torch::Tensor> backward_cuda(
    const torch::Tensor color_dist_grad,
    const torch::Tensor contrast_dist_grad,
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor spixel_indices,
    int num_spixels_w, int num_spixels_h 
){
    int batchsize = pixel_features.size(0); 
    int channels = pixel_features.size(1); 
    int num_pixels = pixel_features.size(2); 
    int num_spixels = spixel_features.size(2); 

    auto pixel_features_grad = torch::zeros_like(pixel_features);
    auto spixel_features_grad = torch::zeros_like(spixel_features);

    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(pixel_features.type(),  "backward_kernel", ([&] {
        backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            color_dist_grad.data_ptr<scalar_t>(), 
            contrast_dist_grad.data_ptr<scalar_t>(), 
            pixel_features.data_ptr<scalar_t>(), 
            spixel_features.data_ptr<scalar_t>(), 
            spixel_indices.data_ptr<scalar_t>(), 
            pixel_features_grad.data_ptr<scalar_t>(), 
            spixel_features_grad.data_ptr<scalar_t>(), 
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h
        );
    }));

    return {pixel_features_grad, spixel_features_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &forward_cuda, "pair_wise_distance forward");
  m.def("backward",  &backward_cuda, "pair_wise_distance backward");
}
'''
