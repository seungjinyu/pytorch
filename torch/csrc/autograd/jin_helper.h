#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <cstdint>
#include <vector>

extern "C" {

// role
bool jin_is_role_B();

// lifecycle
void jin_init_if_needed();
void jin_reset_counters();

// generic
bool jin_has_key(const char* key);
bool jin_overwrite_by_key(at::Tensor& t, const char* key);

// conv2d
void jin_overwrite_conv_input(at::Tensor& t);
void jin_overwrite_conv_weight(at::Tensor& t);

// relu (old overwrite path)
void jin_overwrite_relu_saved(at::Tensor& t);
void jin_overwrite_relu(at::Tensor& t); // alias

// relu (mask-based path)
void jin_capture_relu_mask(const at::Tensor& out);
at::Tensor jin_relu_backward_from_mask(const at::Tensor& grad);

// addmm
void jin_overwrite_addmm_mat1(at::Tensor& t);
void jin_overwrite_addmm_mat2(at::Tensor& t);
void jin_advance_addmm();

// maxpool2d
void jin_overwrite_maxpool2d_input(at::Tensor& t);
void jin_overwrite_maxpool2d_indices(at::Tensor& t);

// payload
C10_EXPORT void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step);

// batchnorm
void jin_overwrite_batchnorm_input(at::Tensor& t);
void jin_overwrite_batchnorm_running_mean(at::Tensor& t);
void jin_overwrite_batchnorm_running_var(at::Tensor& t);
void jin_overwrite_batchnorm_weight(at::Tensor& t);
void jin_overwrite_batchnorm_result1(at::Tensor& t);
void jin_overwrite_batchnorm_result2(at::Tensor& t);

} // extern "C"

// maxpool 2x2, stride=2, padding=0 helpers
std::vector<uint8_t> jin_pack_maxpool2x2_flat_indices_to_2bit(
    const at::Tensor& flat_indices,
    int64_t input_h,
    int64_t input_w
);

at::Tensor jin_unpack_maxpool2x2_2bit_to_flat_indices(
    const std::vector<uint8_t>& packed,
    int64_t N,
    int64_t C,
    int64_t Hout,
    int64_t Wout,
    int64_t input_h,
    int64_t input_w,
    c10::Device device
);

at::Tensor jin_make_maxpool2d_indices_2bit_tensor(
    const at::Tensor& flat_indices
);