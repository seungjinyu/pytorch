#pragma once
#include <ATen/ATen.h>

extern "C" {

// lifecycle
void jin_init_if_needed();
void jin_reset_counters();

// generic
bool jin_has_key(const char* key);
bool jin_overwrite_by_key(at::Tensor& t, const char* key);

// conv2d
void jin_overwrite_conv_input(at::Tensor& t);
void jin_overwrite_conv_weight(at::Tensor& t);

// relu
void jin_overwrite_relu_saved(at::Tensor& t);
void jin_overwrite_relu(at::Tensor& t); // alias

// addmm
void jin_overwrite_addmm_mat1(at::Tensor& t);
void jin_overwrite_addmm_mat2(at::Tensor& t);
void jin_advance_addmm();

// maxpool2d
void jin_overwrite_maxpool2d_input(at::Tensor& t);
void jin_overwrite_maxpool2d_indices(at::Tensor& t);

void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step);

// batchnorm
void jin_overwrite_batchnorm_input(at::Tensor& t);
void jin_overwrite_batchnorm_running_mean(at::Tensor& t);
void jin_overwrite_batchnorm_running_var(at::Tensor& t);
void jin_overwrite_batchnorm_weight(at::Tensor& t);
void jin_overwrite_batchnorm_result1(at::Tensor& t);
void jin_overwrite_batchnorm_result2(at::Tensor& t);

C10_EXPORT void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step);


} // extern "C"