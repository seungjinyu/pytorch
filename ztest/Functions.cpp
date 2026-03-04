#include "torch/csrc/autograd/FunctionsManual.h"
#include "torch/csrc/dynamo/compiled_autograd.h"
#include "torch/csrc/autograd/jin_helper.h"

// @generated from ../tools/autograd/templates/Functions.cpp

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace torch::autograd::generated::details;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd::generated {

variable_list AbsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.sgn()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AbsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AbsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AcosBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * -((-self * self + 1).rsqrt()).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AcosBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AcosBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AddBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, maybe_multiply(grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AddBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_scalar_type);
    args.collect(self_scalar_type);
}
variable_list AddBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_scalar_type);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_scalar_type);
    saved.after(self_scalar_type);
    return result;
}
variable_list AddBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AddBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
}
variable_list AddBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    return result;
}
variable_list AddbmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto batch1_ix = gen.range(1);
  auto batch2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto batch1 = batch1_.unpack();
  auto batch2 = batch2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ batch1_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad.unsqueeze(0).expand_symint({ batch1_sym_argsize_0, batch1_sym_argsize_1, batch2_sym_argsize_2 }).bmm(batch2.transpose(1, 2).conj()), alpha.conj())) : Tensor();
    copy_range(grad_inputs, batch1_ix, grad_result);
  }
  if (task_should_compute_output({ batch2_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(batch1.transpose(1, 2).conj().bmm(grad.unsqueeze(0).expand_symint({ batch1_sym_argsize_0, batch1_sym_argsize_1, batch2_sym_argsize_2 })), alpha.conj())) : Tensor();
    copy_range(grad_inputs, batch2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AddbmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(batch1_);
    args.collect(batch1_sym_argsize_0);
    args.collect(batch1_sym_argsize_1);
    args.collect(batch2_);
    args.collect(batch2_sym_argsize_2);
    args.collect(beta);
}
variable_list AddbmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(batch1_);
    saved.before(batch1_sym_argsize_0);
    saved.before(batch1_sym_argsize_1);
    saved.before(batch2_);
    saved.before(batch2_sym_argsize_2);
    saved.before(beta);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(batch1_);
    saved.after(batch1_sym_argsize_0);
    saved.after(batch1_sym_argsize_1);
    saved.after(batch2_);
    saved.after(batch2_sym_argsize_2);
    saved.after(beta);
    return result;
}
variable_list AddcdivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor1_ix = gen.range(1);
  auto tensor2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto tensor1 = tensor1_.unpack();
  auto tensor2 = tensor2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor1_scalar_type, grad * (value / tensor2).conj())) : Tensor();
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor2_scalar_type, -grad * (value * tensor1 / (tensor2 * tensor2)).conj())) : Tensor();
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void AddcdivBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
    args.collect(tensor1_);
    args.collect(tensor1_scalar_type);
    args.collect(tensor2_);
    args.collect(tensor2_scalar_type);
    args.collect(value);
}
variable_list AddcdivBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    saved.before(tensor1_);
    saved.before(tensor1_scalar_type);
    saved.before(tensor2_);
    saved.before(tensor2_scalar_type);
    saved.before(value);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    saved.after(tensor1_);
    saved.after(tensor1_scalar_type);
    saved.after(tensor2_);
    saved.after(tensor2_scalar_type);
    saved.after(value);
    return result;
}
variable_list AddcmulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor1_ix = gen.range(1);
  auto tensor2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto tensor1 = tensor1_.unpack();
  auto tensor2 = tensor2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor1_scalar_type, grad * (tensor2 * value).conj())) : Tensor();
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor2_scalar_type, grad * (tensor1 * value).conj())) : Tensor();
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void AddcmulBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
    args.collect(tensor1_);
    args.collect(tensor1_scalar_type);
    args.collect(tensor2_);
    args.collect(tensor2_scalar_type);
    args.collect(value);
}
variable_list AddcmulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    saved.before(tensor1_);
    saved.before(tensor1_scalar_type);
    saved.before(tensor2_);
    saved.before(tensor2_scalar_type);
    saved.before(value);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    saved.after(tensor1_);
    saved.after(tensor1_scalar_type);
    saved.after(tensor2_);
    saved.after(tensor2_scalar_type);
    saved.after(value);
    return result;
}
variable_list AddmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat1_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat1 = mat1_.unpack();
  jin_overwrite_addmm_mat1(mat1);

  auto mat2 = mat2_.unpack();
  // ✅ mat2 grad를 계산할 때만 overwrite + idx advance
  if (task_should_compute_output({ mat2_ix })) {
    jin_overwrite_addmm_mat2(mat2);
  } else {
    // 그래도 idx는 다음 addmm를 위해 advance 해야 함
    jin_advance_addmm();
  }

  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat1_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_backward(grad, mat2, mat1_sym_sizes, mat1_sym_strides, mat1_layout, alpha)) : Tensor();
    copy_range(grad_inputs, mat1_ix, grad_result);
  }
  if (task_should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, mat1, mat2_sym_sizes, mat2_sym_strides, mat2_layout, alpha)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AddmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(beta);
    args.collect(mat1_);
    args.collect(mat1_layout);
    args.collect(mat1_sym_sizes);
    args.collect(mat1_sym_strides);
    args.collect(mat2_);
    args.collect(mat2_layout);
    args.collect(mat2_sym_sizes);
    args.collect(mat2_sym_strides);
}
variable_list AddmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(beta);
    saved.before(mat1_);
    saved.before(mat1_layout);
    saved.before(mat1_sym_sizes);
    saved.before(mat1_sym_strides);
    saved.before(mat2_);
    saved.before(mat2_layout);
    saved.before(mat2_sym_sizes);
    saved.before(mat2_sym_strides);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(beta);
    saved.after(mat1_);
    saved.after(mat1_layout);
    saved.after(mat1_sym_sizes);
    saved.after(mat1_sym_strides);
    saved.after(mat2_);
    saved.after(mat2_layout);
    saved.after(mat2_sym_sizes);
    saved.after(mat2_sym_strides);
    return result;
}
variable_list SparseAddmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat1_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat1 = mat1_.unpack();
  auto mat2 = mat2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat1_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_sparse_backward(grad, mat1, mat2, alpha)) : Tensor();
    copy_range(grad_inputs, mat1_ix, grad_result);
  }
  if (task_should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, mat1, mat2_sym_sizes, mat2_sym_strides, mat2_layout, alpha)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseAddmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(beta);
    args.collect(mat1_);
    args.collect(mat2_);
    args.collect(mat2_layout);
    args.collect(mat2_sym_sizes);
    args.collect(mat2_sym_strides);
}
variable_list SparseAddmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(beta);
    saved.before(mat1_);
    saved.before(mat2_);
    saved.before(mat2_layout);
    saved.before(mat2_sym_sizes);
    saved.before(mat2_sym_strides);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(beta);
    saved.after(mat1_);
    saved.after(mat2_);
    saved.after(mat2_layout);
    saved.after(mat2_sym_sizes);
    saved.after(mat2_sym_strides);
    return result;
}
variable_list AddmvBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat_ix = gen.range(1);
  auto vec_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat = mat_.unpack();
  auto vec = vec_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad.ger(vec.conj()), alpha.conj())) : Tensor();
    copy_range(grad_inputs, mat_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ vec_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(mat.t().conj().mv(grad), alpha.conj())) : Tensor();
    copy_range(grad_inputs, vec_ix, grad_result);
  }
  return grad_inputs;
}
void AddmvBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(beta);
    args.collect(mat_);
    args.collect(vec_);
}
variable_list AddmvBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(beta);
    saved.before(mat_);
    saved.before(vec_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(beta);
    saved.after(mat_);
    saved.after(vec_);
    return result;
}
variable_list AddrBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto vec1_ix = gen.range(1);
  auto vec2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto vec1 = vec1_.unpack();
  auto vec2 = vec2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ vec1_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad.mv(vec2.conj()), alpha.conj())) : Tensor();
    copy_range(grad_inputs, vec1_ix, grad_result);
  }
  if (task_should_compute_output({ vec2_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad.t().mv(vec1.conj()), alpha.conj())) : Tensor();
    copy_range(grad_inputs, vec2_ix, grad_result);
  }
  return grad_inputs;
}
void AddrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(beta);
    args.collect(vec1_);
    args.collect(vec2_);
}
variable_list AddrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(beta);
    saved.before(vec1_);
    saved.before(vec2_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(beta);
    saved.after(vec1_);
    saved.after(vec2_);
    return result;
}
variable_list AffineGridGeneratorBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto theta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ theta_ix })) {
    auto grad_result = any_grad_defined ? (affine_grid_generator_backward_symint(grad, size, align_corners)) : Tensor();
    copy_range(grad_inputs, theta_ix, grad_result);
  }
  return grad_inputs;
}
void AffineGridGeneratorBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(size);
}
variable_list AffineGridGeneratorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(size);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(size);
    return result;
}
variable_list AliasBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AliasBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list AliasBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list AngleBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (angle_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AngleBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AngleBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AcoshBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self.is_complex() ? grad * ((self + 1).rsqrt() * (self - 1).rsqrt()).conj() : grad * (self * self - 1).rsqrt()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AcoshBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AcoshBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AcoshBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of acosh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AcoshBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list AcoshBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list AsinhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (self.pow(2) + 1).rsqrt().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsinhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AsinhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AsinhBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of asinh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsinhBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list AsinhBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list AtanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * 1 / (1 - self.pow(2)).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AtanhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AtanhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AtanhBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of atanh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AtanhBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list AtanhBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list AsStridedBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (as_strided_backward(grad, self_geometry, size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsStridedBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_geometry);
    args.collect(size);
    args.collect(storage_offset);
    args.collect(stride);
}
variable_list AsStridedBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_geometry);
    saved.before(size);
    saved.before(storage_offset);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(self_geometry);
    saved.after(size);
    saved.after(storage_offset);
    saved.after(stride);
    return result;
}
variable_list AsStridedBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (as_strided_backward(grad, self_geometry, size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsStridedBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_geometry);
    args.collect(size);
    args.collect(storage_offset);
    args.collect(stride);
}
variable_list AsStridedBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_geometry);
    saved.before(size);
    saved.before(storage_offset);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(self_geometry);
    saved.after(size);
    saved.after(storage_offset);
    saved.after(stride);
    return result;
}
variable_list AsinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (-self * self + 1).rsqrt().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AsinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AtanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self * self + 1).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AtanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AtanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list Atan2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, other_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ other_ix }),
      };
    auto grad_result = atan2_backward(grad, self, other, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ other_ix })) {
        copy_range(grad_inputs, other_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void Atan2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list Atan2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list BaddbmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto batch1_ix = gen.range(1);
  auto batch2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto batch1 = batch1_.unpack();
  auto batch2 = batch2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ batch1_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad.bmm(batch2.transpose(1, 2).conj()), alpha.conj())) : Tensor();
    copy_range(grad_inputs, batch1_ix, grad_result);
  }
  if (task_should_compute_output({ batch2_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(batch1.transpose(1, 2).conj().bmm(grad), alpha.conj())) : Tensor();
    copy_range(grad_inputs, batch2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BaddbmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(batch1_);
    args.collect(batch2_);
    args.collect(beta);
}
variable_list BaddbmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(batch1_);
    saved.before(batch2_);
    saved.before(beta);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(batch1_);
    saved.after(batch2_);
    saved.after(beta);
    return result;
}
variable_list BernoulliBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BernoulliBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list BernoulliBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list BernoulliBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto p_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ p_ix })) {
    auto grad_result = any_grad_defined ? (p_info.zeros()) : Tensor();
    copy_range(grad_inputs, p_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BernoulliBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(p_info);
}
variable_list BernoulliBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p_info);
    variable_list result = apply(variable_list(grads));
    saved.after(p_info);
    return result;
}
variable_list BernoulliBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BernoulliBackward2::compiled_args(CompiledNodeArgs& args) {

}
variable_list BernoulliBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list BmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (self.transpose(1, 2).conj().bmm(grad)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.bmm(mat2.transpose(1, 2).conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void BmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mat2_);
    args.collect(self_);
}
variable_list BmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mat2_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(mat2_);
    saved.after(self_);
    return result;
}
variable_list MatmulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, other_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ other_ix }),
      };
    auto grad_result = matmul_backward(grad, self, other, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ other_ix })) {
        copy_range(grad_inputs, other_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void MatmulBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list MatmulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list CatBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto tensors_ix = gen.range(tensors_size_);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (task_should_compute_output({ tensors_ix })) {
    auto grad_result = cat_tensors_backward(grad, tensors_args_sizes_symint, tensors_args_scalartypes, dim);
    copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}
void CatBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(tensors_args_scalartypes);
    args.collect(tensors_args_sizes_symint);
}
variable_list CatBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(tensors_args_scalartypes);
    saved.before(tensors_args_sizes_symint);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(tensors_args_scalartypes);
    saved.after(tensors_args_sizes_symint);
    return result;
}
variable_list CauchyBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CauchyBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CauchyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list CeilBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CeilBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CeilBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list CholeskyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_backward(grad, upper, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CholeskyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(upper);
    args.collect(result_);
}
variable_list CholeskyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(upper);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(upper);
    saved.after(result_);
    return result;
}
variable_list LinalgCholeskyExBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto L = L_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_backward(grad, upper, L)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgCholeskyExBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(upper);
    args.collect(L_);
}
variable_list LinalgCholeskyExBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(upper);
    saved.before(L_);
    variable_list result = apply(variable_list(grads));
    saved.after(upper);
    saved.after(L_);
    return result;
}
variable_list CholeskySolveBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto input2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input2 = input2_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, input2_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ input2_ix }),
      };
    auto grad_result = cholesky_solve_backward(grad, self, input2, result, upper, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ input2_ix })) {
        copy_range(grad_inputs, input2_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void CholeskySolveBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(input2_);
    args.collect(self_);
    args.collect(upper);
    args.collect(result_);
}
variable_list CholeskySolveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(input2_);
    saved.before(self_);
    saved.before(upper);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(input2_);
    saved.after(self_);
    saved.after(upper);
    saved.after(result_);
    return result;
}
variable_list CholeskyInverseBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_inverse_backward(grad, self, upper, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CholeskyInverseBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(upper);
    args.collect(result_);
}
variable_list CholeskyInverseBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(upper);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(upper);
    saved.after(result_);
    return result;
}
variable_list ClampBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto min_ix = gen.range(1);
  auto max_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto max = max_.unpack();
  auto min = min_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ min_ix, max_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ min_ix }),
        task_should_compute_output({ max_ix }),
      };
    auto grad_result = clamp_backward_min_max(grad, self, min, max, grad_input_mask);
      if (task_should_compute_output({ min_ix })) {
        copy_range(grad_inputs, min_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ max_ix })) {
        copy_range(grad_inputs, max_ix, std::get<1>(grad_result));
      }
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (clamp_backward(grad, self, min, max)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(max_);
    args.collect(min_);
    args.collect(self_);
}
variable_list ClampBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max_);
    saved.before(min_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max_);
    saved.after(min_);
    saved.after(self_);
    return result;
}
variable_list ClampBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (clamp_backward(grad, self, min, max)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(max);
    args.collect(min);
    args.collect(self_);
}
variable_list ClampBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max);
    saved.before(min);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max);
    saved.after(min);
    saved.after(self_);
    return result;
}
variable_list ClampMinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self >= min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampMinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(min);
    args.collect(self_);
}
variable_list ClampMinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(min);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(min);
    saved.after(self_);
    return result;
}
variable_list ClampMinBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto min_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto min = min_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ min_ix })) {
    auto grad_result = any_grad_defined ? (where(self < min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, min_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self >= min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampMinBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(min_);
    args.collect(self_);
}
variable_list ClampMinBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(min_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(min_);
    saved.after(self_);
    return result;
}
variable_list ClampMaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self <= max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampMaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(max);
    args.collect(self_);
}
variable_list ClampMaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max);
    saved.after(self_);
    return result;
}
variable_list ClampMaxBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto max_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto max = max_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ max_ix })) {
    auto grad_result = any_grad_defined ? (where(self > max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, max_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self <= max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ClampMaxBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(max_);
    args.collect(self_);
}
variable_list ClampMaxBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max_);
    saved.after(self_);
    return result;
}
variable_list CloneBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CloneBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CloneBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ToCopyBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_to_copy_backward(grad, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToCopyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_options);
}
variable_list ToCopyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_options);
    variable_list result = apply(variable_list(grads));
    saved.after(self_options);
    return result;
}
variable_list CoalesceBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CoalesceBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CoalesceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ComplexBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto real_ix = gen.range(1);
  auto imag_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto imag = imag_.unpack();
  auto real = real_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ imag_ix })) {
    auto grad_result = any_grad_defined ? (at::imag(grad)) : Tensor();
    copy_range(grad_inputs, imag_ix, grad_result);
  }
  if (task_should_compute_output({ real_ix })) {
    auto grad_result = any_grad_defined ? (at::real(grad)) : Tensor();
    copy_range(grad_inputs, real_ix, grad_result);
  }
  return grad_inputs;
}
void ComplexBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(imag_);
    args.collect(real_);
}
variable_list ComplexBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(imag_);
    saved.before(real_);
    variable_list result = apply(variable_list(grads));
    saved.after(imag_);
    saved.after(real_);
    return result;
}
variable_list PolarBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto abs_ix = gen.range(1);
  auto angle_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ abs_ix, angle_ix })) {
  
    auto grad_result = polar_backward(grad, result);
      if (task_should_compute_output({ abs_ix })) {
        copy_range(grad_inputs, abs_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ angle_ix })) {
        copy_range(grad_inputs, angle_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void PolarBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list PolarBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ConjBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ConjBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ConjBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NegViewBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.neg()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NegViewBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NegViewBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ConjPhysicalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj_physical()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ConjPhysicalBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ConjPhysicalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ConjPhysicalBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj_physical()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ConjPhysicalBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list ConjPhysicalBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list CopysignBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (copysign_tensor_self_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CopysignBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_);
    args.collect(result_);
}
variable_list CopysignBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list CopysignBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (copysign_tensor_self_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CopysignBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list CopysignBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list CosBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self.sin().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CosBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list CosBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list CoshBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.sinh().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CoshBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list CoshBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list LinalgCrossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (at::linalg_cross(grad, self.conj(), dim)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::linalg_cross(other.conj(), grad, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgCrossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(other_);
    args.collect(self_);
}
variable_list LinalgCrossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list LogcumsumexpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (logcumsumexp_backward(grad, self, result, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogcumsumexpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(result_);
}
variable_list LogcumsumexpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list CumprodBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cumprod_backward(grad.to(self_scalar_type), self, dim, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CumprodBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(self_scalar_type);
    args.collect(result_);
}
variable_list CumprodBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(self_scalar_type);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(self_scalar_type);
    saved.after(result_);
    return result;
}
variable_list CumsumBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cumsum_backward(grad.to(self_scalar_type), dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CumsumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_scalar_type);
}
variable_list CumsumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_scalar_type);
    return result;
}
variable_list CummaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cummaxmin_backward(grad, self, indices, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CummaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(indices_);
}
variable_list CummaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(indices_);
    return result;
}
variable_list CumminBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cummaxmin_backward(grad, self, indices, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CumminBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(indices_);
}
variable_list CumminBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(indices_);
    return result;
}
variable_list ConvTbcBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto bias = bias_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
  
    auto grad_result = grad.defined() ? conv_tbc_backward(grad, self, weight, bias, pad) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvTbcBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_);
    args.collect(pad);
    args.collect(self_);
    args.collect(weight_);
}
variable_list ConvTbcBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_);
    saved.before(pad);
    saved.before(self_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_);
    saved.after(pad);
    saved.after(self_);
    saved.after(weight_);
    return result;
}
variable_list CtcLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto log_probs = log_probs_.unpack();
  auto targets = targets_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
void CtcLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(blank);
    args.collect(input_lengths);
    args.collect(log_probs_);
    args.collect(target_lengths);
    args.collect(targets_);
    args.collect(zero_infinity);
    args.collect(result0_);
    args.collect(result1_);
}
variable_list CtcLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(blank);
    saved.before(input_lengths);
    saved.before(log_probs_);
    saved.before(target_lengths);
    saved.before(targets_);
    saved.before(zero_infinity);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(blank);
    saved.after(input_lengths);
    saved.after(log_probs_);
    saved.after(target_lengths);
    saved.after(targets_);
    saved.after(zero_infinity);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list CtcLossBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input_lengths = input_lengths_.unpack();
  auto log_probs = log_probs_.unpack();
  auto target_lengths = target_lengths_.unpack();
  auto targets = targets_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
void CtcLossBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(blank);
    args.collect(input_lengths_);
    args.collect(log_probs_);
    args.collect(target_lengths_);
    args.collect(targets_);
    args.collect(zero_infinity);
    args.collect(result0_);
    args.collect(result1_);
}
variable_list CtcLossBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(blank);
    saved.before(input_lengths_);
    saved.before(log_probs_);
    saved.before(target_lengths_);
    saved.before(targets_);
    saved.before(zero_infinity);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(blank);
    saved.after(input_lengths_);
    saved.after(log_probs_);
    saved.after(target_lengths_);
    saved.after(targets_);
    saved.after(zero_infinity);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list Deg2RadBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (deg2rad_backward(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Deg2RadBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list Deg2RadBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list LinalgDetBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto A = A_.unpack();
  auto LU = LU_.unpack(shared_from_this());
  auto pivots = pivots_.unpack(shared_from_this());
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (linalg_det_backward(grad, result, A, LU, pivots)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgDetBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(A_);
    args.collect(LU_);
    args.collect(pivots_);
    args.collect(result_);
}
variable_list LinalgDetBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(A_);
    saved.before(LU_);
    saved.before(pivots_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(A_);
    saved.after(LU_);
    saved.after(pivots_);
    saved.after(result_);
    return result;
}
variable_list LinalgSlogdetBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_logabsdet = grads[1];
  const auto& grad_sign = grads[0];
  auto A = A_.unpack();
  auto LU = LU_.unpack(shared_from_this());
  auto pivots = pivots_.unpack(shared_from_this());
  auto sign = sign_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (slogdet_backward(grad_sign, grad_logabsdet, A, sign, LU, pivots)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgSlogdetBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(A_);
    args.collect(LU_);
    args.collect(pivots_);
    args.collect(sign_);
}
variable_list LinalgSlogdetBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(A_);
    saved.before(LU_);
    saved.before(pivots_);
    saved.before(sign_);
    variable_list result = apply(variable_list(grads));
    saved.after(A_);
    saved.after(LU_);
    saved.after(pivots_);
    saved.after(sign_);
    return result;
}
variable_list BlockDiagBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto tensors_ix = gen.range(tensors_size_);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (task_should_compute_output({ tensors_ix })) {
    auto grad_result = block_diag_backward(grad, tensors_args_sizes, tensors_args_scalartypes);
    copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}
void BlockDiagBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(tensors_args_scalartypes);
    args.collect(tensors_args_sizes);
}
variable_list BlockDiagBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(tensors_args_scalartypes);
    saved.before(tensors_args_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(tensors_args_scalartypes);
    saved.after(tensors_args_sizes);
    return result;
}
variable_list DiagEmbedBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.diagonal(offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DiagEmbedBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim1);
    args.collect(dim2);
    args.collect(offset);
}
variable_list DiagEmbedBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim1);
    saved.before(dim2);
    saved.before(offset);
    variable_list result = apply(variable_list(grads));
    saved.after(dim1);
    saved.after(dim2);
    saved.after(offset);
    return result;
}
variable_list DiagonalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (diagonal_backward_symint(grad, self_sym_sizes, offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DiagonalBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim1);
    args.collect(dim2);
    args.collect(offset);
    args.collect(self_sym_sizes);
}
variable_list DiagonalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim1);
    saved.before(dim2);
    saved.before(offset);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim1);
    saved.after(dim2);
    saved.after(offset);
    saved.after(self_sym_sizes);
    return result;
}
variable_list DiagonalBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad.diagonal(offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void DiagonalBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim1);
    args.collect(dim2);
    args.collect(offset);
}
variable_list DiagonalBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim1);
    saved.before(dim2);
    saved.before(offset);
    variable_list result = apply(variable_list(grads));
    saved.after(dim1);
    saved.after(dim2);
    saved.after(offset);
    return result;
}
variable_list DistBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-norm_backward(grad, self - other, p, result)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self - other, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DistBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list DistBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list DivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_other_backward(grad, self, other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DivBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
    args.collect(self_scalar_type);
}
variable_list DivBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    saved.after(self_scalar_type);
    return result;
}
variable_list DivBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DivBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other);
    args.collect(self_scalar_type);
}
variable_list DivBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other);
    saved.after(self_scalar_type);
    return result;
}
variable_list DivBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_other_backward(grad, self, other, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DivBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(rounding_mode);
    args.collect(self_);
    args.collect(self_scalar_type);
}
variable_list DivBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(rounding_mode);
    saved.before(self_);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(rounding_mode);
    saved.after(self_);
    saved.after(self_scalar_type);
    return result;
}
variable_list DivBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DivBackward3::compiled_args(CompiledNodeArgs& args) {
    args.collect(other);
    args.collect(rounding_mode);
    args.collect(self_scalar_type);
}
variable_list DivBackward3::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other);
    saved.before(rounding_mode);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other);
    saved.after(rounding_mode);
    saved.after(self_scalar_type);
    return result;
}
variable_list DotBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto tensor = tensor_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * tensor.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.conj()) : Tensor();
    copy_range(grad_inputs, tensor_ix, grad_result);
  }
  return grad_inputs;
}
void DotBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(tensor_);
}
variable_list DotBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(tensor_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(tensor_);
    return result;
}
variable_list VdotBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * other) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void VdotBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list VdotBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list FusedDropoutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_fused_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FusedDropoutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(result1_);
}
variable_list FusedDropoutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(result1_);
    return result;
}
variable_list NativeDropoutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_native_dropout_backward(grad, result1, (!train.has_value() || !train.value() ? 1 : (p == 1 ? 0.0 : 1.0 / (1.0 - p)))) : native_dropout_backward(grad, result1, (!train.has_value() || !train.value() ? 1 : (p == 1 ? 0.0 : 1.0 / (1.0 - p))))) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
void NativeDropoutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(train);
    args.collect(result1_);
}
variable_list NativeDropoutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(train);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(train);
    saved.after(result1_);
    return result;
}
variable_list NativeDropoutBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto mask_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (native_dropout_double_backward(grad, grad_output, mask, scale)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ mask_ix })) {
    auto grad_result = not_implemented("native_dropout_backward: mask");
    copy_range(grad_inputs, mask_ix, grad_result);
  }
  return grad_inputs;
}
void NativeDropoutBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(mask_);
    args.collect(scale);
}
variable_list NativeDropoutBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(mask_);
    saved.before(scale);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(mask_);
    saved.after(scale);
    return result;
}
variable_list EqBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void EqBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list EqBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list EqBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void EqBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list EqBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list ErfBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ErfBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ErfBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ErfcBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ErfcBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ErfcBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SpecialErfcxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? ((2.0 * self * result - 2.0 / sqrt(M_PI)) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialErfcxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SpecialErfcxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list ErfinvBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ErfinvBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ErfinvBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ExpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * result.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ExpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ExpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list Exp2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * result.conj() * M_LN2) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Exp2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list Exp2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list Expm1Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (result.conj() + 1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Expm1Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list Expm1Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ExpandBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::sum_to(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ExpandBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ExpandBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list ExponentialBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ExponentialBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ExponentialBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FakeQuantizePerTensorAffineCachemaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FakeQuantizePerTensorAffineCachemaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list FakeQuantizePerTensorAffineCachemaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list FakeQuantizeLearnablePerTensorAffineBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto scale_ix = gen.range(1);
  auto zero_point_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto scale = scale_.unpack();
  auto self = self_.unpack();
  auto zero_point = zero_point_.unpack();
  if (task_should_compute_output({ self_ix, scale_ix, zero_point_ix })) {
  
    auto grad_result = grad.defined() ? _fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ scale_ix })) {
        copy_range(grad_inputs, scale_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ zero_point_ix })) {
        copy_range(grad_inputs, zero_point_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void FakeQuantizeLearnablePerTensorAffineBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_factor);
    args.collect(quant_max);
    args.collect(quant_min);
    args.collect(scale_);
    args.collect(self_);
    args.collect(zero_point_);
}
variable_list FakeQuantizeLearnablePerTensorAffineBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_factor);
    saved.before(quant_max);
    saved.before(quant_min);
    saved.before(scale_);
    saved.before(self_);
    saved.before(zero_point_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_factor);
    saved.after(quant_max);
    saved.after(quant_min);
    saved.after(scale_);
    saved.after(self_);
    saved.after(zero_point_);
    return result;
}
variable_list FakeQuantizePerChannelAffineCachemaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_channel_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FakeQuantizePerChannelAffineCachemaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list FakeQuantizePerChannelAffineCachemaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list FakeQuantizeLearnablePerChannelAffineBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto scale_ix = gen.range(1);
  auto zero_point_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto scale = scale_.unpack();
  auto self = self_.unpack();
  auto zero_point = zero_point_.unpack();
  if (task_should_compute_output({ self_ix, scale_ix, zero_point_ix })) {
  
    auto grad_result = grad.defined() ? _fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ scale_ix })) {
        copy_range(grad_inputs, scale_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ zero_point_ix })) {
        copy_range(grad_inputs, zero_point_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void FakeQuantizeLearnablePerChannelAffineBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(axis);
    args.collect(grad_factor);
    args.collect(quant_max);
    args.collect(quant_min);
    args.collect(scale_);
    args.collect(self_);
    args.collect(zero_point_);
}
variable_list FakeQuantizeLearnablePerChannelAffineBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(axis);
    saved.before(grad_factor);
    saved.before(quant_max);
    saved.before(quant_min);
    saved.before(scale_);
    saved.before(self_);
    saved.before(zero_point_);
    variable_list result = apply(variable_list(grads));
    saved.after(axis);
    saved.after(grad_factor);
    saved.after(quant_max);
    saved.after(quant_min);
    saved.after(scale_);
    saved.after(self_);
    saved.after(zero_point_);
    return result;
}
variable_list FusedMovingAvgObsFqHelperBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FusedMovingAvgObsFqHelperBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list FusedMovingAvgObsFqHelperBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list FillBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FillBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list FillBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FillBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (grad.sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
void FillBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list FillBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FillBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FillBackward2::compiled_args(CompiledNodeArgs& args) {

}
variable_list FillBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FillBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (grad.sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
void FillBackward3::compiled_args(CompiledNodeArgs& args) {

}
variable_list FillBackward3::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FloorBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FloorBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list FloorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FmodBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FmodBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list FmodBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FmodBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-grad * self.div(other, /*rounding_mode=*/"trunc")) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FmodBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list FmodBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list FracBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FracBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list FracBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FrexpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto exponent = exponent_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / exponent.exp2()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FrexpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent_);
}
variable_list FrexpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent_);
    return result;
}
variable_list GatherBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (gather_backward(grad, self, dim, index, sparse_grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GatherBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
    args.collect(self_);
    args.collect(sparse_grad);
}
variable_list GatherBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    saved.before(self_);
    saved.before(sparse_grad);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    saved.after(self_);
    saved.after(sparse_grad);
    return result;
}
variable_list GeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list GeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list GeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list GeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list GeometricBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeometricBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list GeometricBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list GeqrfBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("geqrf");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeqrfBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list GeqrfBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list GridSampler2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grid = grid_.unpack();
  auto input = input_.unpack();
  if (task_should_compute_output({ input_ix, grid_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ grid_ix }),
      };
    auto grad_result = grad.defined() ? grid_sampler_2d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void GridSampler2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(grid_);
    args.collect(input_);
    args.collect(interpolation_mode);
    args.collect(padding_mode);
}
variable_list GridSampler2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(grid_);
    saved.before(input_);
    saved.before(interpolation_mode);
    saved.before(padding_mode);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(grid_);
    saved.after(input_);
    saved.after(interpolation_mode);
    saved.after(padding_mode);
    return result;
}
variable_list GridSampler3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grid = grid_.unpack();
  auto input = input_.unpack();
  if (task_should_compute_output({ input_ix, grid_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ grid_ix }),
      };
    auto grad_result = grad.defined() ? grid_sampler_3d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void GridSampler3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(grid_);
    args.collect(input_);
    args.collect(interpolation_mode);
    args.collect(padding_mode);
}
variable_list GridSampler3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(grid_);
    saved.before(input_);
    saved.before(interpolation_mode);
    saved.before(padding_mode);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(grid_);
    saved.after(input_);
    saved.after(interpolation_mode);
    saved.after(padding_mode);
    return result;
}
variable_list GridSampler2DCpuFallbackBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grid = grid_.unpack();
  auto input = input_.unpack();
  if (task_should_compute_output({ input_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? _grid_sampler_2d_cpu_fallback_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void GridSampler2DCpuFallbackBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(grid_);
    args.collect(input_);
    args.collect(interpolation_mode);
    args.collect(padding_mode);
}
variable_list GridSampler2DCpuFallbackBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(grid_);
    saved.before(input_);
    saved.before(interpolation_mode);
    saved.before(padding_mode);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(grid_);
    saved.after(input_);
    saved.after(interpolation_mode);
    saved.after(padding_mode);
    return result;
}
variable_list GtBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GtBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list GtBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list GtBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GtBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list GtBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list HardsigmoidBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardsigmoid_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardsigmoidBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list HardsigmoidBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list HardswishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardswish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardswishBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list HardswishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list HardswishBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (hardswish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::where(at::logical_and(-3.0 < self, self < 3.0), grad * grad_output / 3.0, at::zeros({}, self_options))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardswishBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(self_);
    args.collect(self_options);
}
variable_list HardswishBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(self_);
    saved.before(self_options);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(self_);
    saved.after(self_options);
    return result;
}
variable_list HypotBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * other / result) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / result) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HypotBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
    args.collect(result_);
}
variable_list HypotBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list I0Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::special_i1(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void I0Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list I0Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SpecialI0EBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (at::special_i1e(self) - self.sgn() * result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialI0EBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SpecialI0EBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list SpecialI1Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (i1_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialI1Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SpecialI1Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list SpecialI1EBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (i1e_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialI1EBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SpecialI1EBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list IgammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * exp((self - 1) * log(other) - other - lgamma(self))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("igamma: input");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void IgammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list IgammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list IgammacBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-grad * exp((self - 1) * log(other) - other - lgamma(self))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("igammac: input");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void IgammacBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list IgammacBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list IndexBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (index_backward(grad.new_zeros_symint(self_sym_sizes, self_options), indices, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void IndexBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_options);
    args.collect(self_sym_sizes);
}
variable_list IndexBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UnsafeIndexBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_unsafe_index_put(grad.new_zeros_symint(self_sym_sizes, self_options), indices, grad, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsafeIndexBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_options);
    args.collect(self_sym_sizes);
}
variable_list UnsafeIndexBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    return result;
}
variable_list IndexAddBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(source_dim > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0)), alpha)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
void IndexAddBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(dim);
    args.collect(index_);
    args.collect(source_);
    args.collect(source_dim);
}
variable_list IndexAddBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(dim);
    saved.before(index_);
    saved.before(source_);
    saved.before(source_dim);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(dim);
    saved.after(index_);
    saved.after(source_);
    saved.after(source_dim);
    return result;
}
variable_list IndexReduceBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto self = self_.unpack();
  auto source = source_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, source_ix })) {
  
    auto grad_result = index_reduce_backward(grad, self, dim, index, source, reduce, include_self, result);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ source_ix })) {
        copy_range(grad_inputs, source_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void IndexReduceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(include_self);
    args.collect(index_);
    args.collect(reduce);
    args.collect(self_);
    args.collect(source_);
    args.collect(result_);
}
variable_list IndexReduceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(include_self);
    saved.before(index_);
    saved.before(reduce);
    saved.before(self_);
    saved.before(source_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(include_self);
    saved.after(index_);
    saved.after(reduce);
    saved.after(self_);
    saved.after(source_);
    saved.after(result_);
    return result;
}
variable_list IndexCopyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.index_fill(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (source_dim > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0))) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
void IndexCopyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
    args.collect(source_);
    args.collect(source_dim);
}
variable_list IndexCopyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    saved.before(source_);
    saved.before(source_dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    saved.after(source_);
    saved.after(source_dim);
    return result;
}
variable_list IndexFillBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.index_fill(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void IndexFillBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
}
variable_list IndexFillBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    return result;
}
variable_list IndexFillBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.index_fill(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (grad.index_select(dim, std::get<0>(at::_unique(index, /*sorted=*/false))).sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
void IndexFillBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
}
variable_list IndexFillBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    return result;
}
variable_list IndexPutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.index_put(indices, values_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.index(indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
void IndexPutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(accumulate);
    args.collect(indices_);
    args.collect(values_info);
}
variable_list IndexPutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(accumulate);
    saved.before(indices_);
    saved.before(values_info);
    variable_list result = apply(variable_list(grads));
    saved.after(accumulate);
    saved.after(indices_);
    saved.after(values_info);
    return result;
}
variable_list UnsafeIndexPutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : at::_unsafe_index_put(grad, indices, values_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (at::_unsafe_index(grad, indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
void UnsafeIndexPutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(accumulate);
    args.collect(indices_);
    args.collect(values_info);
}
variable_list UnsafeIndexPutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(accumulate);
    saved.before(indices_);
    saved.before(values_info);
    variable_list result = apply(variable_list(grads));
    saved.after(accumulate);
    saved.after(indices_);
    saved.after(values_info);
    return result;
}
variable_list IndexPutImplBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.index_put(indices, values_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.index(indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
void IndexPutImplBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(accumulate);
    args.collect(indices_);
    args.collect(values_info);
}
variable_list IndexPutImplBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(accumulate);
    saved.before(indices_);
    saved.before(values_info);
    variable_list result = apply(variable_list(grads));
    saved.after(accumulate);
    saved.after(indices_);
    saved.after(values_info);
    return result;
}
variable_list IndexSelectBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (index_select_backward_symint(grad, self_sym_sizes, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void IndexSelectBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
    args.collect(self_sym_sizes);
}
variable_list IndexSelectBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    saved.after(self_sym_sizes);
    return result;
}
variable_list LinalgInvExBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto inverse = inverse_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (-at::matmul(inverse.mH(), at::matmul(grad, inverse.mH()))) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgInvExBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(inverse_);
}
variable_list LinalgInvExBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(inverse_);
    variable_list result = apply(variable_list(grads));
    saved.after(inverse_);
    return result;
}
variable_list LinalgPinvBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pinv_backward(grad, result, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgPinvBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list LinalgPinvBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list KthvalueBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void KthvalueBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list KthvalueBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list LeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list LeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list LeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list LeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list LerpBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto end_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ end_ix })) {
    auto grad_result = any_grad_defined ? (grad * weight.conj()) : Tensor();
    copy_range(grad_inputs, end_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (weight.isComplex() ? grad * (1 - weight.conj().toComplexDouble()) : grad * (1 - weight.toDouble())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LerpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(weight);
}
variable_list LerpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(weight);
    variable_list result = apply(variable_list(grads));
    saved.after(weight);
    return result;
}
variable_list LerpBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto end_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto end = end_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ end_ix })) {
    auto grad_result = any_grad_defined ? (grad * weight.conj()) : Tensor();
    copy_range(grad_inputs, end_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (1 - weight).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (grad * (end - self).conj()) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
void LerpBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(end_);
    args.collect(self_);
    args.collect(weight_);
}
variable_list LerpBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(end_);
    saved.before(self_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(end_);
    saved.after(self_);
    saved.after(weight_);
    return result;
}
variable_list LgammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * digamma(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LgammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list LgammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list DigammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DigammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list DigammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list PolygammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(n + 1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PolygammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(n);
    args.collect(self_);
}
variable_list PolygammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(n);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(n);
    saved.after(self_);
    return result;
}
variable_list PolygammaBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(n + 1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PolygammaBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(n);
    args.collect(self_);
}
variable_list PolygammaBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(n);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(n);
    saved.after(self_);
    return result;
}
variable_list LogBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.div(self.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list LogBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list Log10Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self.conj() * 2.3025850929940456)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Log10Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list Log10Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list Log1PBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log1p_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Log1PBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list Log1PBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list Log2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self.conj() * 0.6931471805599453)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Log2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list Log2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list LogaddexpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + exp(self - other)).conj()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + exp(other - self)).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogaddexpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list LogaddexpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list Logaddexp2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + pow(2, self - other))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + pow(2, other - self))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Logaddexp2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list Logaddexp2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list XlogyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / other) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::xlogy(grad, other).masked_fill((self == 0.) & (other <= 0.), 0.)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void XlogyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list XlogyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list XlogyBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / other) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
void XlogyBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self);
}
variable_list XlogyBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self);
    return result;
}
variable_list XlogyBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (other.toDouble() > 0. ? at::xlogy(grad,  other) : at::xlogy(grad,  other).masked_fill(self == 0., 0.)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void XlogyBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(other);
    args.collect(self_);
}
variable_list XlogyBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other);
    saved.after(self_);
    return result;
}
variable_list SpecialXlog1PyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / (other + 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::special_xlog1py(grad,  other).masked_fill((self == 0.) & (other <= -1.), 0.)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialXlog1PyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list SpecialXlog1PyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list SpecialXlog1PyBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / (other + 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialXlog1PyBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self);
}
variable_list SpecialXlog1PyBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self);
    return result;
}
variable_list SpecialXlog1PyBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (other.toDouble() > -1. ? at::special_xlog1py(grad,  other) : at::special_xlog1py(grad,  other).masked_fill(self == 0., 0.)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialXlog1PyBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(other);
    args.collect(self_);
}
variable_list SpecialXlog1PyBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other);
    saved.after(self_);
    return result;
}
variable_list SpecialZetaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self * special_zeta(self + 1., other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("zeta");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialZetaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list SpecialZetaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list SpecialZetaBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self * special_zeta(self.toDouble() + 1., other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialZetaBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self);
}
variable_list SpecialZetaBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self);
    return result;
}
variable_list SpecialZetaBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("zeta");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialZetaBackward2::compiled_args(CompiledNodeArgs& args) {

}
variable_list SpecialZetaBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list LogNormalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogNormalBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list LogNormalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list LogsumexpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (logsumexp_backward(grad, self, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogsumexpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result_);
}
variable_list LogsumexpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list LinalgLstsqBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto b_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto b = b_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, b_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ b_ix }),
      };
    auto grad_result = linalg_lstsq_backward(grad, self, b, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ b_ix })) {
        copy_range(grad_inputs, b_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void LinalgLstsqBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(b_);
    args.collect(self_);
}
variable_list LinalgLstsqBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(b_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(b_);
    saved.after(self_);
    return result;
}
variable_list LtBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LtBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list LtBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list LtBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LtBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list LtBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list LinalgLuFactorExBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto LU = LU_.unpack(shared_from_this());
  auto pivots = pivots_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (lu_factor_ex_backward(grad, LU, pivots, pivot)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgLuFactorExBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(pivot);
    args.collect(LU_);
    args.collect(pivots_);
}
variable_list LinalgLuFactorExBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(pivot);
    saved.before(LU_);
    saved.before(pivots_);
    variable_list result = apply(variable_list(grads));
    saved.after(pivot);
    saved.after(LU_);
    saved.after(pivots_);
    return result;
}
variable_list LinalgLuBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_L = grads[0];
  const auto& grad_U = grads[1];
  auto L = L_.unpack(shared_from_this());
  auto P = P_.unpack(shared_from_this());
  auto U = U_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (linalg_lu_backward(grad_L, grad_U, P, L, U, pivot)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgLuBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(pivot);
    args.collect(L_);
    args.collect(P_);
    args.collect(U_);
}
variable_list LinalgLuBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(pivot);
    saved.before(L_);
    saved.before(P_);
    saved.before(U_);
    variable_list result = apply(variable_list(grads));
    saved.after(pivot);
    saved.after(L_);
    saved.after(P_);
    saved.after(U_);
    return result;
}
variable_list LinalgLuSolveBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto LU_ix = gen.range(1);
  auto B_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto LU = LU_.unpack();
  auto pivots = pivots_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ B_ix })) {
    auto grad_result = any_grad_defined ? (at::linalg_lu_solve(LU, pivots, grad, left, !adjoint)) : Tensor();
    copy_range(grad_inputs, B_ix, grad_result);
  }
  if (task_should_compute_output({ LU_ix })) {
    auto grad_result = any_grad_defined ? (linalg_lu_solve_LU(grad, LU, pivots, result, left, adjoint)) : Tensor();
    copy_range(grad_inputs, LU_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgLuSolveBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(LU_);
    args.collect(adjoint);
    args.collect(left);
    args.collect(pivots_);
    args.collect(result_);
}
variable_list LinalgLuSolveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(LU_);
    saved.before(adjoint);
    saved.before(left);
    saved.before(pivots_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(LU_);
    saved.after(adjoint);
    saved.after(left);
    saved.after(pivots_);
    saved.after(result_);
    return result;
}
variable_list LuUnpackBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto LU_data_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_L = grads[0];
  const auto& grad_U = grads[1];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ LU_data_ix })) {
    auto grad_result = any_grad_defined ? (lu_unpack_backward(grad_L, grad_U, LU_data_sym_argsize_minus_2, LU_data_sym_argsize_minus_1)) : Tensor();
    copy_range(grad_inputs, LU_data_ix, grad_result);
  }
  return grad_inputs;
}
void LuUnpackBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(LU_data_sym_argsize_minus_1);
    args.collect(LU_data_sym_argsize_minus_2);
}
variable_list LuUnpackBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(LU_data_sym_argsize_minus_1);
    saved.before(LU_data_sym_argsize_minus_2);
    variable_list result = apply(variable_list(grads));
    saved.after(LU_data_sym_argsize_minus_1);
    saved.after(LU_data_sym_argsize_minus_2);
    return result;
}
variable_list MaskedFillBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedFillBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list MaskedFillBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list MaskedFillBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (masked_fill_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedFillBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
}
variable_list MaskedFillBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    return result;
}
variable_list MaskedScatterBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (masked_scatter_backward_symint(grad, mask, source_sym_sizes)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
    args.collect(source_sym_sizes);
}
variable_list MaskedScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    saved.before(source_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    saved.after(source_sym_sizes);
    return result;
}
variable_list MaskedScatterBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad_output_info.zeros().masked_scatter(mask, grad)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedScatterBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_info);
    args.collect(mask_);
}
variable_list MaskedScatterBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_info);
    saved.before(mask_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_info);
    saved.after(mask_);
    return result;
}
variable_list MaskedSelectBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (masked_select_backward(grad, self, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedSelectBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
    args.collect(self_);
}
variable_list MaskedSelectBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    saved.after(self_);
    return result;
}
variable_list LinalgMatrixExpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_matrix_exp_differential(self, grad, /*adjoint*/ true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgMatrixExpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list LinalgMatrixExpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list MaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list MaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list MaxBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list MaxBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list MaximumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaximumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list MaximumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list FmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill((self >= other).logical_or_(other.isnan()), 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill((self >= other).logical_or_(other.isnan()).logical_not_(), 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list FmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list MeanBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand_symint(self_sym_sizes) / self_sym_numel) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MeanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_numel);
    args.collect(self_sym_sizes);
}
variable_list MeanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_numel);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_numel);
    saved.after(self_sym_sizes);
    return result;
}
variable_list MeanBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mean_backward(grad, self_sym_sizes, dim, self_sym_numel, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MeanBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_numel);
    args.collect(self_sym_sizes);
}
variable_list MeanBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_numel);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_numel);
    saved.after(self_sym_sizes);
    return result;
}
variable_list MedianBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MedianBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list MedianBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list NanmedianBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NanmedianBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list NanmedianBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list MedianBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MedianBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list MedianBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list NanmedianBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NanmedianBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list NanmedianBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list MinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list MinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list MinBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MinBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list MinBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list MinimumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MinimumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list MinimumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list FminBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill((self <= other).logical_or_(other.isnan()), 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.masked_fill((self <= other).logical_or_(other.isnan()).logical_not_(), 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FminBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list FminBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list AmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result_);
}
variable_list AmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list AminBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AminBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result_);
}
variable_list AminBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list MmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, self, mat2_sym_sizes, mat2_sym_strides, mat2_layout, 1)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_backward(grad, mat2, self_sym_sizes, self_sym_strides, self_layout, 1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mat2_);
    args.collect(mat2_layout);
    args.collect(mat2_sym_sizes);
    args.collect(mat2_sym_strides);
    args.collect(self_);
    args.collect(self_layout);
    args.collect(self_sym_sizes);
    args.collect(self_sym_strides);
}
variable_list MmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mat2_);
    saved.before(mat2_layout);
    saved.before(mat2_sym_sizes);
    saved.before(mat2_sym_strides);
    saved.before(self_);
    saved.before(self_layout);
    saved.before(self_sym_sizes);
    saved.before(self_sym_strides);
    variable_list result = apply(variable_list(grads));
    saved.after(mat2_);
    saved.after(mat2_layout);
    saved.after(mat2_sym_sizes);
    saved.after(mat2_sym_strides);
    saved.after(self_);
    saved.after(self_layout);
    saved.after(self_sym_sizes);
    saved.after(self_sym_strides);
    return result;
}
variable_list ModeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ModeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list ModeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list MulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, self, other_scalar_type)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MulBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(other_scalar_type);
    args.collect(self_);
    args.collect(self_scalar_type);
}
variable_list MulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(other_scalar_type);
    saved.before(self_);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(other_scalar_type);
    saved.after(self_);
    saved.after(self_scalar_type);
    return result;
}
variable_list MulBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MulBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other);
    args.collect(self_scalar_type);
}
variable_list MulBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(other);
    saved.after(self_scalar_type);
    return result;
}
variable_list MvBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto vec_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto vec = vec_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.ger(vec.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ vec_ix })) {
    auto grad_result = any_grad_defined ? (self.conj().t().mv(grad)) : Tensor();
    copy_range(grad_inputs, vec_ix, grad_result);
  }
  return grad_inputs;
}
void MvBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(vec_);
}
variable_list MvBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(vec_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(vec_);
    return result;
}
variable_list MvlgammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mvlgamma_backward(grad, self, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MvlgammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(self_);
}
variable_list MvlgammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(self_);
    return result;
}
variable_list NanToNumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::isfinite(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NanToNumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list NanToNumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list NativeBatchNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());

  // [JIN] overwrite BN saved vars (Node B)
  jin_overwrite_batchnorm_input(input);
  jin_overwrite_batchnorm_running_mean(running_mean);
  jin_overwrite_batchnorm_running_var(running_var);
  jin_overwrite_batchnorm_weight(weight);
  jin_overwrite_batchnorm_result1(result1);
  jin_overwrite_batchnorm_result2(result2);
  
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeBatchNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(training);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeBatchNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(training);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(training);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NativeBatchNormLegitBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeBatchNormLegitBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(training);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeBatchNormLegitBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(training);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(training);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NativeBatchNormLegitNoTrainingBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, /*training=*/false, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeBatchNormLegitNoTrainingBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeBatchNormLegitNoTrainingBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NativeBatchNormLegitBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_batch_norm_backward(grad, input, weight, Tensor(), Tensor(), result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeBatchNormLegitBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(input_);
    args.collect(training);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeBatchNormLegitBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(input_);
    saved.before(training);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(input_);
    saved.after(training);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NativeBatchNormBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_invstd_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_out = grad_out_.unpack();
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_invstd = save_invstd_.unpack();
  auto save_mean = save_mean_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, grad_out_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ grad_out_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, running_mean, running_var, train, eps, save_mean, save_invstd, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ grad_out_ix })) {
        copy_range(grad_inputs, grad_out_ix, std::get<2>(grad_result));
      }
  }
  if (task_should_compute_output({ save_invstd_ix })) {
    auto grad_result = not_implemented("native_batch_norm_backward save_invstd");
    copy_range(grad_inputs, save_invstd_ix, grad_result);
  }
  if (task_should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("native_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  return grad_inputs;
}
void NativeBatchNormBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(grad_out_);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(save_invstd_);
    args.collect(save_mean_);
    args.collect(train);
    args.collect(weight_);
}
variable_list NativeBatchNormBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(grad_out_);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(save_invstd_);
    saved.before(save_mean_);
    saved.before(train);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(grad_out_);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(save_invstd_);
    saved.after(save_mean_);
    saved.after(train);
    saved.after(weight_);
    return result;
}
variable_list NativeLayerNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto bias = bias_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_layer_norm_backward_symint(grad, input, normalized_shape, result1, result2, weight, bias, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeLayerNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_);
    args.collect(input_);
    args.collect(normalized_shape);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeLayerNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_);
    saved.before(input_);
    saved.before(normalized_shape);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_);
    saved.after(input_);
    saved.after(normalized_shape);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NativeLayerNormBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto mean_ix = gen.range(1);
  auto rstd_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_out = grad_out_.unpack();
  auto input = input_.unpack();
  auto mean = mean_.unpack();
  auto rstd = rstd_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ bias_ix })) {
    auto grad_result = any_grad_defined ? (Tensor()) : Tensor();
    copy_range(grad_inputs, bias_ix, grad_result);
  }
  if (task_should_compute_output({ input_ix, weight_ix, grad_out_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ grad_out_ix }),
      };
    auto grad_result = layer_norm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, mean, rstd, normalized_shape, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ grad_out_ix })) {
        copy_range(grad_inputs, grad_out_ix, std::get<2>(grad_result));
      }
  }
  if (task_should_compute_output({ mean_ix })) {
    auto grad_result = not_implemented("native_layer_norm_backward mean");
    copy_range(grad_inputs, mean_ix, grad_result);
  }
  if (task_should_compute_output({ rstd_ix })) {
    auto grad_result = not_implemented("native_layer_norm_backward rstd");
    copy_range(grad_inputs, rstd_ix, grad_result);
  }
  return grad_inputs;
}
void NativeLayerNormBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_out_);
    args.collect(input_);
    args.collect(mean_);
    args.collect(normalized_shape);
    args.collect(rstd_);
    args.collect(weight_);
}
variable_list NativeLayerNormBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_out_);
    saved.before(input_);
    saved.before(mean_);
    saved.before(normalized_shape);
    saved.before(rstd_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_out_);
    saved.after(input_);
    saved.after(mean_);
    saved.after(normalized_shape);
    saved.after(rstd_);
    saved.after(weight_);
    return result;
}
variable_list NativeGroupNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = GradMode::is_enabled() || grads[1].defined() || grads[2].defined() ? infinitely_differentiable_native_group_norm_backward(grads[0], grads[1], grads[2], input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask) : (grads[0].defined() ? native_group_norm_backward_symint(grads[0].device().is_xpu() ? grads[0] : grads[0].contiguous(grads[0].device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), input.device().is_xpu() ? input : input.contiguous(input.device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), result1, result2, weight, N, C, HxW, group, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>());
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NativeGroupNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(C);
    args.collect(HxW);
    args.collect(N);
    args.collect(eps);
    args.collect(group);
    args.collect(input_);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list NativeGroupNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(C);
    saved.before(HxW);
    saved.before(N);
    saved.before(eps);
    saved.before(group);
    saved.before(input_);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(C);
    saved.after(HxW);
    saved.after(N);
    saved.after(eps);
    saved.after(group);
    saved.after(input_);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list NeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list NeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list NeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_info);
    args.collect(self_info);
}
variable_list NeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_info);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(other_info);
    saved.after(self_info);
    return result;
}
variable_list NegBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.neg()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NegBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NegBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NextafterBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = not_implemented("nextafter");
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("nextafter");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NextafterBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NextafterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list NormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list NormBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self, p, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NormBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list NormBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list NormBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self.to(grad.scalar_type()), p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NormBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list NormBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list NormBackward3::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self.to(grad.scalar_type()), p, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NormBackward3::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list NormBackward3::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list LinalgVectorNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_vector_norm_backward(grad, self, ord, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgVectorNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(ord);
    args.collect(self_);
    args.collect(result_);
}
variable_list LinalgVectorNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(ord);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(ord);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list PdistBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_pdist_backward(grad, self, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PdistBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(self_);
    args.collect(result_);
}
variable_list PdistBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list PdistBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto pdist_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ grad_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, grad_ix, grad_result);
  }
  if (task_should_compute_output({ pdist_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, pdist_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PdistBackwardBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list PdistBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list EuclideanDistBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ x1_ix, x2_ix })) {
  
    auto grad_result = _euclidean_dist_backward(grad, x1, x2, result);
      if (task_should_compute_output({ x1_ix })) {
        copy_range(grad_inputs, x1_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ x2_ix })) {
        copy_range(grad_inputs, x2_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void EuclideanDistBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(x1_);
    args.collect(x2_);
    args.collect(result_);
}
variable_list EuclideanDistBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(x1_);
    saved.before(x2_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(x1_);
    saved.after(x2_);
    saved.after(result_);
    return result;
}
variable_list CdistBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ x1_ix })) {
    auto grad_result = any_grad_defined ? (_cdist_backward(grad.contiguous(), x1, x2, p, result)) : Tensor();
    copy_range(grad_inputs, x1_ix, grad_result);
  }
  if (task_should_compute_output({ x2_ix })) {
    auto grad_result = any_grad_defined ? (_cdist_backward(grad.mT().contiguous(), x2, x1, p, result.mT().contiguous())) : Tensor();
    copy_range(grad_inputs, x2_ix, grad_result);
  }
  return grad_inputs;
}
void CdistBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(p);
    args.collect(x1_);
    args.collect(x2_);
    args.collect(result_);
}
variable_list CdistBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(p);
    saved.before(x1_);
    saved.before(x2_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(p);
    saved.after(x1_);
    saved.after(x2_);
    saved.after(result_);
    return result;
}
variable_list CdistBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto cdist_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ cdist_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, cdist_ix, grad_result);
  }
  if (task_should_compute_output({ grad_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, grad_ix, grad_result);
  }
  if (task_should_compute_output({ x1_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, x1_ix, grad_result);
  }
  if (task_should_compute_output({ x2_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, x2_ix, grad_result);
  }
  return grad_inputs;
}
void CdistBackwardBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CdistBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NormalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NormalBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NormalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NormalBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto mean_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mean_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros_symint(mean_sym_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, mean_ix, grad_result);
  }
  return grad_inputs;
}
void NormalBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(mean_sym_sizes);
}
variable_list NormalBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mean_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(mean_sym_sizes);
    return result;
}
variable_list NormalBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto std_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ std_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros_symint(std_sym_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, std_ix, grad_result);
  }
  return grad_inputs;
}
void NormalBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(std_sym_sizes);
}
variable_list NormalBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(std_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(std_sym_sizes);
    return result;
}
variable_list NormalBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto mean_ix = gen.range(1);
  auto std_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ mean_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros_symint(mean_sym_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, mean_ix, grad_result);
  }
  if (task_should_compute_output({ std_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros_symint(std_sym_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, std_ix, grad_result);
  }
  return grad_inputs;
}
void NormalBackward3::compiled_args(CompiledNodeArgs& args) {
    args.collect(mean_sym_sizes);
    args.collect(std_sym_sizes);
}
variable_list NormalBackward3::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mean_sym_sizes);
    saved.before(std_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(mean_sym_sizes);
    saved.after(std_sym_sizes);
    return result;
}
variable_list LinalgHouseholderProductBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto tau_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto tau = tau_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, tau_ix })) {
  
    auto grad_result = householder_product_backward(grad, result, input, tau);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ tau_ix })) {
        copy_range(grad_inputs, tau_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void LinalgHouseholderProductBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(input_);
    args.collect(tau_);
    args.collect(result_);
}
variable_list LinalgHouseholderProductBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(input_);
    saved.before(tau_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(input_);
    saved.after(tau_);
    saved.after(result_);
    return result;
}
variable_list OrmqrBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto input2_ix = gen.range(1);
  auto input3_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input2 = input2_.unpack();
  auto input3 = input3_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, input2_ix, input3_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ input2_ix }),
        task_should_compute_output({ input3_ix }),
      };
    auto grad_result = ormqr_backward(grad, result, self, input2, input3, left, transpose, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ input2_ix })) {
        copy_range(grad_inputs, input2_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ input3_ix })) {
        copy_range(grad_inputs, input3_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void OrmqrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(input2_);
    args.collect(input3_);
    args.collect(left);
    args.collect(self_);
    args.collect(transpose);
    args.collect(result_);
}
variable_list OrmqrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(input2_);
    saved.before(input3_);
    saved.before(left);
    saved.before(self_);
    saved.before(transpose);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(input2_);
    saved.after(input3_);
    saved.after(left);
    saved.after(self_);
    saved.after(transpose);
    saved.after(result_);
    return result;
}
variable_list PermuteBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (permute_backwards(grad, dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PermuteBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dims);
}
variable_list PermuteBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dims);
    variable_list result = apply(variable_list(grads));
    saved.after(dims);
    return result;
}
variable_list PoissonBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PoissonBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list PoissonBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list PowBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward(grad, self, exponent)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PowBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent);
    args.collect(self_);
}
variable_list PowBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent);
    saved.after(self_);
    return result;
}
variable_list PowBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto exponent_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto exponent = exponent_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ exponent_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_exponent(grad, self, exponent, result)) : Tensor();
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_self(grad, self, exponent)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PowBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent_);
    args.collect(self_);
    args.collect(result_);
}
variable_list PowBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent_);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent_);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list PowBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto exponent_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto exponent = exponent_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ exponent_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_exponent(grad, self, exponent, result)) : Tensor();
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  return grad_inputs;
}
void PowBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent_);
    args.collect(self);
    args.collect(result_);
}
variable_list PowBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent_);
    saved.before(self);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent_);
    saved.after(self);
    saved.after(result_);
    return result;
}
variable_list ProdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (prod_backward(grad, self.to(grad.scalar_type()), result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ProdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list ProdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list ProdBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (prod_backward(grad, self.to(grad.scalar_type()), result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ProdBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result_);
}
variable_list ProdBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list PutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.put(index, source_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (grad.take(index).reshape_as(source)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
void PutBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(accumulate);
    args.collect(index_);
    args.collect(source_);
    args.collect(source_info);
}
variable_list PutBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(accumulate);
    saved.before(index_);
    saved.before(source_);
    saved.before(source_info);
    variable_list result = apply(variable_list(grads));
    saved.after(accumulate);
    saved.after(index_);
    saved.after(source_);
    saved.after(source_info);
    return result;
}
variable_list LinalgQrBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_Q = grads[0];
  const auto& grad_R = grads[1];
  auto Q = Q_.unpack(shared_from_this());
  auto R = R_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (linalg_qr_backward(grad_Q, grad_R, Q, R, mode)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgQrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mode);
    args.collect(Q_);
    args.collect(R_);
}
variable_list LinalgQrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mode);
    saved.before(Q_);
    saved.before(R_);
    variable_list result = apply(variable_list(grads));
    saved.after(mode);
    saved.after(Q_);
    saved.after(R_);
    return result;
}
variable_list Rad2DegBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rad2deg_backward(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Rad2DegBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list Rad2DegBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RandomBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RandomBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list RandomBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RandomBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RandomBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list RandomBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RandomBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RandomBackward2::compiled_args(CompiledNodeArgs& args) {

}
variable_list RandomBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ReciprocalBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-grad * (result * result).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReciprocalBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ReciprocalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list RemainderBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RemainderBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list RemainderBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RemainderBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-grad * self.div(other, /*rounding_mode=*/"floor")) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RemainderBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list RemainderBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list RenormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (renorm_backward(grad, self, p, dim, maxnorm)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RenormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(maxnorm);
    args.collect(p);
    args.collect(self_);
}
variable_list RenormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(maxnorm);
    saved.before(p);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(maxnorm);
    saved.after(p);
    saved.after(self_);
    return result;
}
variable_list RepeatBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (repeat_backward(grad, repeats, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RepeatBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(repeats);
    args.collect(self_sym_sizes);
}
variable_list RepeatBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(repeats);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(repeats);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SpecialEntrBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (-(1 + self.log()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialEntrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list SpecialEntrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SpecialNdtriBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * std::sqrt(2 * M_PI) * (result.square() / 2).exp()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialNdtriBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list SpecialNdtriBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list SpecialLogNdtrBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / std::sqrt(2 * M_PI) * (result + self.pow(2) / 2).neg().exp()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SpecialLogNdtrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SpecialLogNdtrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list ReshapeAliasBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReshapeAliasBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ReshapeAliasBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list RoundBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RoundBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list RoundBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RoundBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RoundBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list RoundBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list RsqrtBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-0.5 * grad * result.pow(3).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RsqrtBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list RsqrtBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ScatterBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.scatter(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.gather(dim, index)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void ScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
}
variable_list ScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    return result;
}
variable_list ScatterBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.scatter(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ScatterBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
}
variable_list ScatterBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    return result;
}
variable_list ScatterAddBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.gather(dim, index)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void ScatterAddBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index_);
}
variable_list ScatterAddBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index_);
    return result;
}
variable_list SelectBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (select_backward_symint(grad, self_sym_sizes, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SelectBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
    args.collect(self_sym_sizes);
}
variable_list SelectBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SelectBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_nested_select_backward_symint(grad, self, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SelectBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
    args.collect(self_);
}
variable_list SelectBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    saved.after(self_);
    return result;
}
variable_list SelectBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad.select_symint(dim, index)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void SelectBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
}
variable_list SelectBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    return result;
}
variable_list SigmoidBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sigmoid_backward(grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SigmoidBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list SigmoidBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list LogitBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_logit_backward(grad, self, eps) : logit_backward(grad, self, eps)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogitBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eps);
    args.collect(self_);
}
variable_list LogitBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eps);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(eps);
    saved.after(self_);
    return result;
}
variable_list SignBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SignBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list SignBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list SgnBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sgn_backward(self, grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SgnBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list SgnBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list SinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.cos().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list SinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SincBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sinc_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SincBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list SincBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SinhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.cosh().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SinhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list SinhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list SliceBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slice_backward_wrapper(grad, self_sym_sizes, dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SliceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(end);
    args.collect(self_sym_sizes);
    args.collect(start);
    args.collect(step);
}
variable_list SliceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(end);
    saved.before(self_sym_sizes);
    saved.before(start);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(end);
    saved.after(self_sym_sizes);
    saved.after(start);
    saved.after(step);
    return result;
}
variable_list SliceBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad.slice_symint(dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void SliceBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(end);
    args.collect(start);
    args.collect(step);
}
variable_list SliceBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(end);
    saved.before(start);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(end);
    saved.after(start);
    saved.after(step);
    return result;
}
variable_list SliceScatterBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slice_scatter_symint(grad, src_info.zeros(), dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.slice_symint(dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void SliceScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(end);
    args.collect(src_info);
    args.collect(start);
    args.collect(step);
}
variable_list SliceScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(end);
    saved.before(src_info);
    saved.before(start);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(end);
    saved.after(src_info);
    saved.after(start);
    saved.after(step);
    return result;
}
variable_list SelectScatterBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (select_scatter_symint(grad, src_info.zeros(), dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.select_symint(dim, index)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void SelectScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
    args.collect(src_info);
}
variable_list SelectScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    saved.before(src_info);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    saved.after(src_info);
    return result;
}
variable_list DiagonalScatterBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (diagonal_scatter(grad, src_info.zeros(), offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.diagonal(offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void DiagonalScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim1);
    args.collect(dim2);
    args.collect(offset);
    args.collect(src_info);
}
variable_list DiagonalScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim1);
    saved.before(dim2);
    saved.before(offset);
    saved.before(src_info);
    variable_list result = apply(variable_list(grads));
    saved.after(dim1);
    saved.after(dim2);
    saved.after(offset);
    saved.after(src_info);
    return result;
}
variable_list AsStridedScatterBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (as_strided_scatter_backward(grad, self_geometry, src_geometry, size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.contiguous().as_strided_symint(size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
void AsStridedScatterBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_geometry);
    args.collect(size);
    args.collect(src_geometry);
    args.collect(storage_offset);
    args.collect(stride);
}
variable_list AsStridedScatterBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_geometry);
    saved.before(size);
    saved.before(src_geometry);
    saved.before(storage_offset);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(self_geometry);
    saved.after(size);
    saved.after(src_geometry);
    saved.after(storage_offset);
    saved.after(stride);
    return result;
}
variable_list LinalgSolveExBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  auto B_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto A = A_.unpack();
  auto LU = LU_.unpack(shared_from_this());
  auto pivots = pivots_.unpack(shared_from_this());
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ A_ix, B_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ A_ix }),
        task_should_compute_output({ B_ix }),
      };
    auto grad_result = linalg_solve_backward(grad, result, A, LU, pivots, left, grad_input_mask[1]);
      if (task_should_compute_output({ A_ix })) {
        copy_range(grad_inputs, A_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ B_ix })) {
        copy_range(grad_inputs, B_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void LinalgSolveExBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(A_);
    args.collect(left);
    args.collect(LU_);
    args.collect(pivots_);
    args.collect(result_);
}
variable_list LinalgSolveExBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(A_);
    saved.before(left);
    saved.before(LU_);
    saved.before(pivots_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(A_);
    saved.after(left);
    saved.after(LU_);
    saved.after(pivots_);
    saved.after(result_);
    return result;
}
variable_list SortBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SortBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list SortBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list SortBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SortBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list SortBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list SplitBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_backward(grads, split_size, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_size);
}
variable_list SplitBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_size);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_size);
    return result;
}
variable_list UnsafeSplitBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_backward(grads, split_size, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsafeSplitBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_size);
}
variable_list UnsafeSplitBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_size);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_size);
    return result;
}
variable_list SplitWithSizesBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_with_sizes_backward(grads, split_sizes, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitWithSizesBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_sizes);
}
variable_list SplitWithSizesBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_sizes);
    return result;
}
variable_list SplitWithSizesBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_nested_split_with_sizes_backward(grads, split_sizes, dim, at::native::get_nested_tensor_impl(self)->get_nested_sizes(), self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitWithSizesBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(self_options);
    args.collect(split_sizes);
}
variable_list SplitWithSizesBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(self_options);
    saved.before(split_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(self_options);
    saved.after(split_sizes);
    return result;
}
variable_list UnsafeSplitWithSizesBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_with_sizes_backward(grads, split_sizes, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsafeSplitWithSizesBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_sizes);
}
variable_list UnsafeSplitWithSizesBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_sizes);
    return result;
}
variable_list SqrtBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (2 * result.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqrtBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list SqrtBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list SqueezeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackwardAutogradNestedTensor0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.unsqueeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list SqueezeBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list SqueezeBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackwardAutogradNestedTensor1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_multiple(grad, dim, self_dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackwardAutogradNestedTensor1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_dim);
}
variable_list SqueezeBackwardAutogradNestedTensor1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_dim);
    return result;
}
variable_list SqueezeBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward3::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward3::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackward4::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward4::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward4::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackward5::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward5::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward5::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list StdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (std_backward(result, grad, self, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void StdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(correction);
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result_);
}
variable_list StdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(correction);
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(correction);
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list StdMeanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (std_mean_backward(grads[0], grads[1], self, result0, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void StdMeanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(correction);
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(result0_);
}
variable_list StdMeanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(correction);
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(result0_);
    variable_list result = apply(variable_list(grads));
    saved.after(correction);
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(result0_);
    return result;
}
variable_list SubBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, maybe_multiply(-grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SubBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_scalar_type);
    args.collect(self_scalar_type);
}
variable_list SubBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_scalar_type);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_scalar_type);
    saved.after(self_scalar_type);
    return result;
}
variable_list SubBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SubBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
}
variable_list SubBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    return result;
}
variable_list RsubBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, maybe_multiply(-grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RsubBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_scalar_type);
    args.collect(self_scalar_type);
}
variable_list RsubBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_scalar_type);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_scalar_type);
    saved.after(self_scalar_type);
    return result;
}
variable_list RsubBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, maybe_multiply(-grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RsubBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(self_scalar_type);
}
variable_list RsubBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(self_scalar_type);
    return result;
}
variable_list SumBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list SumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list SumBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sum_backward(grad, self_sym_sizes, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SumBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_sym_sizes);
}
variable_list SumBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SumBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_nested_sum_backward(grad, self, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SumBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
}
variable_list SumBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    return result;
}
variable_list NansumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nansum_backward(grad.to(self_scalar_type), self, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NansumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
    args.collect(self_scalar_type);
}
variable_list NansumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    saved.before(self_scalar_type);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    saved.after(self_scalar_type);
    return result;
}
variable_list LinalgSvdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_S = grads[1];
  const auto& grad_U = grads[0];
  const auto& grad_Vh = grads[2];
  auto S = S_.unpack(shared_from_this());
  auto U = U_.unpack(shared_from_this());
  auto Vh = Vh_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (svd_backward(full_matrices && grad_U.defined() ? grad_U.narrow_symint(-1, 0, S_sym_argsize_minus_1) : grad_U, grad_S, full_matrices && grad_Vh.defined() ? grad_Vh.narrow_symint(-2, 0, S_sym_argsize_minus_1) : grad_Vh, full_matrices ? U.narrow_symint(-1, 0, S_sym_argsize_minus_1) : U, S, full_matrices ? Vh.narrow_symint(-2, 0, S_sym_argsize_minus_1) : Vh)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgSvdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(full_matrices);
    args.collect(S_);
    args.collect(S_sym_argsize_minus_1);
    args.collect(U_);
    args.collect(Vh_);
}
variable_list LinalgSvdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(full_matrices);
    saved.before(S_);
    saved.before(S_sym_argsize_minus_1);
    saved.before(U_);
    saved.before(Vh_);
    variable_list result = apply(variable_list(grads));
    saved.after(full_matrices);
    saved.after(S_);
    saved.after(S_sym_argsize_minus_1);
    saved.after(U_);
    saved.after(Vh_);
    return result;
}
variable_list LinalgEighBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors = eigenvectors_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, /*is_hermitian=*/true)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgEighBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(eigenvalues_);
    args.collect(eigenvectors_);
}
variable_list LinalgEighBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(eigenvalues_);
    saved.before(eigenvectors_);
    variable_list result = apply(variable_list(grads));
    saved.after(eigenvalues_);
    saved.after(eigenvectors_);
    return result;
}
variable_list LinalgEigBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors = eigenvectors_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, linalg_eig_backward(grads[0], grads[1], eigenvalues, eigenvectors, /*is_hermitian=*/false))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LinalgEigBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_scalar_type);
    args.collect(eigenvalues_);
    args.collect(eigenvectors_);
}
variable_list LinalgEigBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_scalar_type);
    saved.before(eigenvalues_);
    saved.before(eigenvectors_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_scalar_type);
    saved.after(eigenvalues_);
    saved.after(eigenvectors_);
    return result;
}
variable_list TBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.t()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list TBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.t()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TBackward1::compiled_args(CompiledNodeArgs& args) {

}
variable_list TBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list FlipBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.flip(dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FlipBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dims);
}
variable_list FlipBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dims);
    variable_list result = apply(variable_list(grads));
    saved.after(dims);
    return result;
}
variable_list RollBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.roll_symint(fmap(reverse_list_symint(shifts), [](c10::SymInt i){return -i;}), reverse_list(dims))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RollBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dims);
    args.collect(shifts);
}
variable_list RollBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dims);
    saved.before(shifts);
    variable_list result = apply(variable_list(grads));
    saved.after(dims);
    saved.after(shifts);
    return result;
}
variable_list Rot90Backward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.rot90(-k, dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Rot90Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dims);
    args.collect(k);
}
variable_list Rot90Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dims);
    saved.before(k);
    variable_list result = apply(variable_list(grads));
    saved.after(dims);
    saved.after(k);
    return result;
}
variable_list TakeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (take_backward(grad, self, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TakeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(index_);
    args.collect(self_);
}
variable_list TakeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(index_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(index_);
    saved.after(self_);
    return result;
}
variable_list TanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (1 + result.pow(2)).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list TanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list TanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (tanh_backward(grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TanhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list TanhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list TopkBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward_symint(grad, dim, indices, self_sym_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TopkBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
    args.collect(indices_);
}
variable_list TopkBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    saved.after(indices_);
    return result;
}
variable_list TraceBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (trace_backward_symint(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TraceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list TraceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list TransposeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.transpose(dim0, dim1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim0);
    args.collect(dim1);
}
variable_list TransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim0);
    saved.before(dim1);
    variable_list result = apply(variable_list(grads));
    saved.after(dim0);
    saved.after(dim1);
    return result;
}
variable_list TransposeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.transpose(dim0, dim1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TransposeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim0);
    args.collect(dim1);
}
variable_list TransposeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim0);
    saved.before(dim1);
    variable_list result = apply(variable_list(grads));
    saved.after(dim0);
    saved.after(dim1);
    return result;
}
variable_list TriangularSolveBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad_cloned_coefficient = grads[1];
  const auto& grad_solution = grads[0];
  auto A = A_.unpack();
  auto self = self_.unpack();
  auto solution = solution_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, A_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ A_ix }),
      };
    auto grad_result = triangular_solve_backward(grad_solution, grad_cloned_coefficient, self, A, solution, upper, transpose, unitriangular, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ A_ix })) {
        copy_range(grad_inputs, A_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void TriangularSolveBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(A_);
    args.collect(self_);
    args.collect(transpose);
    args.collect(unitriangular);
    args.collect(upper);
    args.collect(solution_);
}
variable_list TriangularSolveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(A_);
    saved.before(self_);
    saved.before(transpose);
    saved.before(unitriangular);
    saved.before(upper);
    saved.before(solution_);
    variable_list result = apply(variable_list(grads));
    saved.after(A_);
    saved.after(self_);
    saved.after(transpose);
    saved.after(unitriangular);
    saved.after(upper);
    saved.after(solution_);
    return result;
}
variable_list LinalgSolveTriangularBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto B_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, B_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ B_ix }),
      };
    auto grad_result = linalg_solve_triangular_backward(grad, self, result, upper, left, unitriangular, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ B_ix })) {
        copy_range(grad_inputs, B_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void LinalgSolveTriangularBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(left);
    args.collect(self_);
    args.collect(unitriangular);
    args.collect(upper);
    args.collect(result_);
}
variable_list LinalgSolveTriangularBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(left);
    saved.before(self_);
    saved.before(unitriangular);
    saved.before(upper);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(left);
    saved.after(self_);
    saved.after(unitriangular);
    saved.after(upper);
    saved.after(result_);
    return result;
}
variable_list TrilBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.tril(diagonal)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TrilBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(diagonal);
}
variable_list TrilBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(diagonal);
    variable_list result = apply(variable_list(grads));
    saved.after(diagonal);
    return result;
}
variable_list TriuBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.triu(diagonal)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TriuBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(diagonal);
}
variable_list TriuBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(diagonal);
    variable_list result = apply(variable_list(grads));
    saved.after(diagonal);
    return result;
}
variable_list TruncBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TruncBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list TruncBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ToDenseBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_dense_backward(grad, self, masked_grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToDenseBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(masked_grad);
    args.collect(self_);
}
variable_list ToDenseBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(masked_grad);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(masked_grad);
    saved.after(self_);
    return result;
}
variable_list ToSparseBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToSparseBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToSparseCsrBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseCsrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseCsrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToSparseCscBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseCscBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseCscBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToSparseBsrBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseBsrBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseBsrBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToSparseBscBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_sparse_backward(grad, self_layout, self_self_sym_blocksize_opt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToSparseBscBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_layout);
    args.collect(self_self_sym_blocksize_opt);
}
variable_list ToSparseBscBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_layout);
    saved.before(self_self_sym_blocksize_opt);
    variable_list result = apply(variable_list(grads));
    saved.after(self_layout);
    saved.after(self_self_sym_blocksize_opt);
    return result;
}
variable_list ToMkldnnBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_mkldnn_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToMkldnnBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ToMkldnnBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list UnfoldBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unfold_backward_symint(grad, self_sym_sizes, dimension, size, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnfoldBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dimension);
    args.collect(self_sym_sizes);
    args.collect(size);
    args.collect(step);
}
variable_list UnfoldBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dimension);
    saved.before(self_sym_sizes);
    saved.before(size);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dimension);
    saved.after(self_sym_sizes);
    saved.after(size);
    saved.after(step);
    return result;
}
variable_list UnfoldBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_in_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_in_ix })) {
    auto grad_result = any_grad_defined ? (grad.unfold(dim, size, step)) : Tensor();
    copy_range(grad_inputs, grad_in_ix, grad_result);
  }
  return grad_inputs;
}
void UnfoldBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(size);
    args.collect(step);
}
variable_list UnfoldBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(size);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(size);
    saved.after(step);
    return result;
}
variable_list UniformBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UniformBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list UniformBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UniqueBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_unique");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UniqueBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list UniqueBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UniqueDimBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_dim");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UniqueDimBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list UniqueDimBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UniqueConsecutiveBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_consecutive");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UniqueConsecutiveBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list UniqueConsecutiveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UniqueDimConsecutiveBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_dim_consecutive");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UniqueDimConsecutiveBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list UniqueDimConsecutiveBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list Unique2Backward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_unique2");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Unique2Backward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list Unique2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UnsafeViewBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsafeViewBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list UnsafeViewBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list LiftBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LiftBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list LiftBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list LiftFreshBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LiftFreshBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list LiftFreshBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UnsqueezeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.squeeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsqueezeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list UnsqueezeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list UnsqueezeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.squeeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsqueezeBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list UnsqueezeBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list VarBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (var_backward(grad, self, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void VarBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(correction);
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
}
variable_list VarBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(correction);
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(correction);
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    return result;
}
variable_list VarMeanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (var_mean_backward(grads[0], grads[1], self, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void VarMeanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(correction);
    args.collect(dim);
    args.collect(keepdim);
    args.collect(self_);
}
variable_list VarMeanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(correction);
    saved.before(dim);
    saved.before(keepdim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(correction);
    saved.after(dim);
    saved.after(keepdim);
    saved.after(self_);
    return result;
}
variable_list ViewBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ViewBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list ViewBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ViewBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ViewAsRealBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_complex(grad.contiguous())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewAsRealBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ViewAsRealBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ViewAsComplexBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_real(grad.contiguous().resolve_conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewAsComplexBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ViewAsComplexBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list WhereBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto condition = condition_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (where(condition, 0, grad)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(condition, grad, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void WhereBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(condition_);
}
variable_list WhereBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(condition_);
    variable_list result = apply(variable_list(grads));
    saved.after(condition_);
    return result;
}
variable_list WeightNormInterfaceBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto v_ix = gen.range(1);
  auto g_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto g = g_.unpack();
  auto v = v_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ v_ix, g_ix })) {
  
    auto grad_result = grad.defined() ? (GradMode::is_enabled() ? _weight_norm_differentiable_backward(grad.contiguous(), v, g, result1, dim) : _weight_norm_interface_backward(grad.contiguous(), v, g, result1, dim)) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ v_ix })) {
        copy_range(grad_inputs, v_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ g_ix })) {
        copy_range(grad_inputs, g_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void WeightNormInterfaceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(g_);
    args.collect(v_);
    args.collect(result1_);
}
variable_list WeightNormInterfaceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(g_);
    saved.before(v_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(g_);
    saved.after(v_);
    saved.after(result1_);
    return result;
}
variable_list ZeroBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ZeroBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ZeroBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list SparseMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sparse_mask_backward(grad, mask, self_layout)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseMaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(mask_);
    args.collect(self_layout);
}
variable_list SparseMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(mask_);
    saved.before(self_layout);
    variable_list result = apply(variable_list(grads));
    saved.after(mask_);
    saved.after(self_layout);
    return result;
}
variable_list SparseCooTensorWithDimsAndTensorsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.sparse_mask(result)._values()) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
void SparseCooTensorWithDimsAndTensorsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list SparseCooTensorWithDimsAndTensorsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list SparseCompressedTensorBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto values = values_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.to_dense().sparse_mask(result).values()) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
void SparseCompressedTensorBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(values_);
    args.collect(result_);
}
variable_list SparseCompressedTensorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(values_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(values_);
    saved.after(result_);
    return result;
}
variable_list SparseSumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_sparse_sum_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseSumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
}
variable_list SparseSumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    return result;
}
variable_list StandardGammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * _standard_gamma_grad(self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void StandardGammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result_);
}
variable_list StandardGammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list StandardGammaGradBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_standard_gamma_grad");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void StandardGammaGradBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list StandardGammaGradBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ValuesBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_sparse_coo_tensor_unsafe_symint(self.indices(), grad, self_sym_sizes)._coalesced_(true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ValuesBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(self_sym_sizes);
}
variable_list ValuesBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(self_sym_sizes);
    return result;
}
variable_list ValuesBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_nested_view_from_buffer(grad.contiguous(), self._nested_tensor_size(), self._nested_tensor_strides(), self._nested_tensor_storage_offsets())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ValuesBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ValuesBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list TrilinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto i1_ix = gen.range(1);
  auto i2_ix = gen.range(1);
  auto i3_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto i1 = i1_.unpack();
  auto i2 = i2_.unpack();
  auto i3 = i3_.unpack();
  if (task_should_compute_output({ i1_ix, i2_ix, i3_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ i1_ix }),
        task_should_compute_output({ i2_ix }),
        task_should_compute_output({ i3_ix }),
      };
    auto grad_result = _trilinear_backward(grad, wrap_opt_if(i1, grad_input_mask[1] || grad_input_mask[2]), wrap_opt_if(i2, grad_input_mask[0] || grad_input_mask[2]), wrap_opt_if(i3, grad_input_mask[0] || grad_input_mask[1]), expand1, expand2, expand3, sumdim, grad_input_mask);
      if (task_should_compute_output({ i1_ix })) {
        copy_range(grad_inputs, i1_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ i2_ix })) {
        copy_range(grad_inputs, i2_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ i3_ix })) {
        copy_range(grad_inputs, i3_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void TrilinearBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(expand1);
    args.collect(expand2);
    args.collect(expand3);
    args.collect(i1_);
    args.collect(i2_);
    args.collect(i3_);
    args.collect(sumdim);
}
variable_list TrilinearBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(expand1);
    saved.before(expand2);
    saved.before(expand3);
    saved.before(i1_);
    saved.before(i2_);
    saved.before(i3_);
    saved.before(sumdim);
    variable_list result = apply(variable_list(grads));
    saved.after(expand1);
    saved.after(expand2);
    saved.after(expand3);
    saved.after(i1_);
    saved.after(i2_);
    saved.after(i3_);
    saved.after(sumdim);
    return result;
}
variable_list ConstantPadNdBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (constant_pad_nd_backward(grad, pad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ConstantPadNdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(pad);
}
variable_list ConstantPadNdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(pad);
    variable_list result = apply(variable_list(grads));
    saved.after(pad);
    return result;
}
variable_list BinaryCrossEntropyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_backward(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_target_backward(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void BinaryCrossEntropyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
}
variable_list BinaryCrossEntropyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list BinaryCrossEntropyBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_double_backward_grad_output(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_double_backward(grad_output, grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_double_backward_target(grad, grad_output, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void BinaryCrossEntropyBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
}
variable_list BinaryCrossEntropyBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto pos_weight = pos_weight_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_with_logits_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_with_logits_target_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void BinaryCrossEntropyWithLogitsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(pos_weight_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(pos_weight_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(pos_weight_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list EmbeddingBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (embedding_backward_symint(grad, indices, weight_sym_argsize_0, padding_idx, scale_grad_by_freq, sparse)) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
void EmbeddingBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(padding_idx);
    args.collect(scale_grad_by_freq);
    args.collect(sparse);
    args.collect(weight_sym_argsize_0);
}
variable_list EmbeddingBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(padding_idx);
    saved.before(scale_grad_by_freq);
    saved.before(sparse);
    saved.before(weight_sym_argsize_0);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(padding_idx);
    saved.after(scale_grad_by_freq);
    saved.after(sparse);
    saved.after(weight_sym_argsize_0);
    return result;
}
variable_list EmbeddingDenseBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (embedding_dense_double_backward_symint(grad, indices, padding_idx)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void EmbeddingDenseBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(padding_idx);
}
variable_list EmbeddingDenseBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(padding_idx);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(padding_idx);
    return result;
}
variable_list EmbeddingBagBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto weight_ix = gen.range(1);
  auto per_sample_weights_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  auto offsets = offsets_.unpack();
  auto per_sample_weights = per_sample_weights_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ per_sample_weights_ix })) {
    auto grad_result = any_grad_defined ? (_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, result1, mode, padding_idx)) : Tensor();
    copy_range(grad_inputs, per_sample_weights_ix, grad_result);
  }
  if (task_should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (_embedding_bag_backward_symint(grad, indices, offsets, result1, result2, result3, weight_sym_argsize_0, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx)) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
void EmbeddingBagBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(mode);
    args.collect(offsets_);
    args.collect(padding_idx);
    args.collect(per_sample_weights_);
    args.collect(scale_grad_by_freq);
    args.collect(sparse);
    args.collect(weight_);
    args.collect(weight_sym_argsize_0);
    args.collect(result1_);
    args.collect(result2_);
    args.collect(result3_);
}
variable_list EmbeddingBagBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(mode);
    saved.before(offsets_);
    saved.before(padding_idx);
    saved.before(per_sample_weights_);
    saved.before(scale_grad_by_freq);
    saved.before(sparse);
    saved.before(weight_);
    saved.before(weight_sym_argsize_0);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(mode);
    saved.after(offsets_);
    saved.after(padding_idx);
    saved.after(per_sample_weights_);
    saved.after(scale_grad_by_freq);
    saved.after(sparse);
    saved.after(weight_);
    saved.after(weight_sym_argsize_0);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    return result;
}
variable_list EmbeddingRenormBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("embedding_renorm");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void EmbeddingRenormBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list EmbeddingRenormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list MseLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_backward(grad, target, self, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void MseLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list MseLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list MultiMarginLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MultiMarginLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(margin);
    args.collect(p);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
}
variable_list MultiMarginLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(margin);
    saved.before(p);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(margin);
    saved.after(p);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list MultilabelMarginLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto is_target = is_target_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (multilabel_margin_loss_backward(grad, self, target, reduction, is_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MultilabelMarginLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(is_target_);
}
variable_list MultilabelMarginLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(is_target_);
    variable_list result = apply(variable_list(grads));
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(is_target_);
    return result;
}
variable_list NllLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto total_weight = total_weight_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss_backward_symint(grad, self, target, weight, reduction, ignore_index, total_weight)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NllLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ignore_index);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
    args.collect(total_weight_);
}
variable_list NllLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ignore_index);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    saved.before(total_weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(ignore_index);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    saved.after(total_weight_);
    return result;
}
variable_list NllLoss2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto total_weight = total_weight_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss2d_backward_symint(grad, self, target, weight, reduction, ignore_index, total_weight)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NllLoss2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ignore_index);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
    args.collect(weight_);
    args.collect(total_weight_);
}
variable_list NllLoss2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ignore_index);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    saved.before(weight_);
    saved.before(total_weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(ignore_index);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    saved.after(weight_);
    saved.after(total_weight_);
    return result;
}
variable_list SmoothL1LossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_backward(grad, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_backward(grad, target, self, reduction, beta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void SmoothL1LossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(beta);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list SmoothL1LossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(beta);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(beta);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list HuberLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_backward(grad, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_backward(grad, target, self, reduction, delta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void HuberLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(delta);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list HuberLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(delta);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(delta);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list SoftMarginLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftMarginLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list SoftMarginLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list ReluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  // [JIN]
  jin_overwrite_relu_saved(result);

  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, result, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ReluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list SiluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_silu_backward(grad, self) : silu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SiluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list SiluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list MishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_mish_backward(grad, self) : mish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MishBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list MishBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list EluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, /* is_result */ false, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void EluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(input_scale);
    args.collect(scale);
    args.collect(self_);
}
variable_list EluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(input_scale);
    saved.before(scale);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(input_scale);
    saved.after(scale);
    saved.after(self_);
    return result;
}
variable_list EluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, /* is_result */ true, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void EluBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(input_scale);
    args.collect(scale);
    args.collect(result_);
}
variable_list EluBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(input_scale);
    saved.before(scale);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(input_scale);
    saved.after(scale);
    saved.after(result_);
    return result;
}
variable_list CeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ false, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CeluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(self_);
}
variable_list CeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(self_);
    return result;
}
variable_list CeluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ true, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void CeluBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(result_);
}
variable_list CeluBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(result_);
    return result;
}
variable_list GeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (gelu_backward(grad, self, approximate)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(approximate);
    args.collect(self_);
}
variable_list GeluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(approximate);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(approximate);
    saved.after(self_);
    return result;
}
variable_list GeluBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (gelu_backward(grad, self, approximate)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (gelu_double_backward(grad, grad_output, self, approximate)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GeluBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(approximate);
    args.collect(grad_output_);
    args.collect(self_);
}
variable_list GeluBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(approximate);
    saved.before(grad_output_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(approximate);
    saved.after(grad_output_);
    saved.after(self_);
    return result;
}
variable_list GluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (glu_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
}
variable_list GluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    return result;
}
variable_list HardshrinkBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardshrinkBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lambd);
    args.collect(self_);
}
variable_list HardshrinkBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lambd);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(lambd);
    saved.after(self_);
    return result;
}
variable_list HardshrinkBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_out_ix })) {
    auto grad_result = any_grad_defined ? (hardshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, grad_out_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardshrinkBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lambd);
    args.collect(self_);
}
variable_list HardshrinkBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lambd);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(lambd);
    saved.after(self_);
    return result;
}
variable_list HardtanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardtanh_backward(grad, self, min_val, max_val)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardtanhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(max_val);
    args.collect(min_val);
    args.collect(self_);
}
variable_list HardtanhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max_val);
    saved.before(min_val);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max_val);
    saved.after(min_val);
    saved.after(self_);
    return result;
}
variable_list LeakyReluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, self, negative_slope, false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LeakyReluBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(negative_slope);
    args.collect(self_);
}
variable_list LeakyReluBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(negative_slope);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(negative_slope);
    saved.after(self_);
    return result;
}
variable_list LeakyReluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, result, negative_slope, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LeakyReluBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(negative_slope);
    args.collect(result_);
}
variable_list LeakyReluBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(negative_slope);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(negative_slope);
    saved.after(result_);
    return result;
}
variable_list LogSigmoidBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto buffer = buffer_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_backward(grad, self, buffer)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogSigmoidBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(buffer_);
}
variable_list LogSigmoidBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(buffer_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(buffer_);
    return result;
}
variable_list LogSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_log_softmax_backward_data(grad, result, dim, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_scalar_type);
    args.collect(result_);
}
variable_list LogSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_scalar_type);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_scalar_type);
    saved.after(result_);
    return result;
}
variable_list SparseLogSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_sparse_log_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseLogSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(result_);
}
variable_list SparseLogSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list MaskedSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_masked_softmax_backward(grad, result, mask, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaskedSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(mask_);
    args.collect(result_);
}
variable_list MaskedSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(mask_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(mask_);
    saved.after(result_);
    return result;
}
variable_list PreluKernelBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix })) {
  
    auto grad_result = grad.defined() ? _prelu_kernel_backward(grad, self, weight) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void PreluKernelBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(weight_);
}
variable_list PreluKernelBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(weight_);
    return result;
}
variable_list PreluKernelBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grads[0].defined() ? (grads[1].defined() ? at::where(self >= 0, grads[0], grads[0] * weight + grads[1] * self) : at::where(self >= 0, grads[0], grads[0] * weight)) : at::where(self >= 0, at::zeros({}, grad_output_options), grads[1] * self)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grads[1].defined() ? at::where(self >= 0, at::zeros({}, self_options), grad_output * grads[1]) : self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (grads[0].defined() ? at::where(self >= 0, at::zeros({}, weight_options), grad_output * grads[0]) : self_info.zeros()) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
void PreluKernelBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(grad_output_options);
    args.collect(self_);
    args.collect(self_info);
    args.collect(self_options);
    args.collect(weight_);
    args.collect(weight_options);
}
variable_list PreluKernelBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(grad_output_options);
    saved.before(self_);
    saved.before(self_info);
    saved.before(self_options);
    saved.before(weight_);
    saved.before(weight_options);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(grad_output_options);
    saved.after(self_);
    saved.after(self_info);
    saved.after(self_options);
    saved.after(weight_);
    saved.after(weight_options);
    return result;
}
variable_list RreluWithNoiseBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto noise = noise_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RreluWithNoiseBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lower);
    args.collect(noise_);
    args.collect(self_);
    args.collect(training);
    args.collect(upper);
}
variable_list RreluWithNoiseBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lower);
    saved.before(noise_);
    saved.before(self_);
    saved.before(training);
    saved.before(upper);
    variable_list result = apply(variable_list(grads));
    saved.after(lower);
    saved.after(noise_);
    saved.after(self_);
    saved.after(training);
    saved.after(upper);
    return result;
}
variable_list RreluWithNoiseBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto noise = noise_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, result, noise, lower, upper, training, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RreluWithNoiseBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(lower);
    args.collect(noise_);
    args.collect(training);
    args.collect(upper);
    args.collect(result_);
}
variable_list RreluWithNoiseBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lower);
    saved.before(noise_);
    saved.before(training);
    saved.before(upper);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(lower);
    saved.after(noise_);
    saved.after(training);
    saved.after(upper);
    saved.after(result_);
    return result;
}
variable_list SoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_softmax_backward_data(grad, result, dim, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_scalar_type);
    args.collect(result_);
}
variable_list SoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_scalar_type);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_scalar_type);
    saved.after(result_);
    return result;
}
variable_list SparseSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_sparse_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseSoftmaxBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(result_);
}
variable_list SparseSoftmaxBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list SparseSparseMatmulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (sparse_sparse_matmul_backward(grad, self, other, 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sparse_sparse_matmul_backward(grad, self, other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SparseSparseMatmulBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list SparseSparseMatmulBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list SoftplusBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softplus_backward(grad, self, beta, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftplusBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(beta);
    args.collect(self_);
    args.collect(threshold);
}
variable_list SoftplusBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(beta);
    saved.before(self_);
    saved.before(threshold);
    variable_list result = apply(variable_list(grads));
    saved.after(beta);
    saved.after(self_);
    saved.after(threshold);
    return result;
}
variable_list SoftshrinkBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftshrinkBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lambd);
    args.collect(self_);
}
variable_list SoftshrinkBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lambd);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(lambd);
    saved.after(self_);
    return result;
}
variable_list ThresholdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ThresholdBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(threshold);
}
variable_list ThresholdBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(threshold);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(threshold);
    return result;
}
variable_list ThresholdBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ThresholdBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(threshold);
}
variable_list ThresholdBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(threshold);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(threshold);
    return result;
}
variable_list ReflectionPad1DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad1d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad1DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReflectionPad1DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list ReflectionPad2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad2d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReflectionPad2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list ReflectionPad3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad3d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReflectionPad3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list ReplicationPad1DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad1d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad1DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReplicationPad1DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list ReplicationPad2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad2d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReplicationPad2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list ReplicationPad3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad3d_backward_symint(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_);
}
variable_list ReplicationPad3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_);
    return result;
}
variable_list UpsampleLinear1DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleLinear1DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales);
    args.collect(self_sym_sizes);
}
variable_list UpsampleLinear1DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleBilinear2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBilinear2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleBilinear2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleBilinear2DAaBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_bilinear2d_aa_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBilinear2DAaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleBilinear2DAaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleBicubic2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBicubic2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleBicubic2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleBicubic2DAaBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_bicubic2d_aa_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBicubic2DAaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleBicubic2DAaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleTrilinear3DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d_backward_symint(grad, output_size, self_sym_sizes, align_corners, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleTrilinear3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleTrilinear3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearest1DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d_backward_symint(grad, output_size, self_sym_sizes, scales)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest1DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearest1DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearestExact1DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact1d_backward_symint(grad, output_size, self_sym_sizes, scales)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact1DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearestExact1DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearest2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d_backward_symint(grad, output_size, self_sym_sizes, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearest2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearestExact2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact2d_backward_symint(grad, output_size, self_sym_sizes, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearestExact2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearest3DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d_backward_symint(grad, output_size, self_sym_sizes, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearest3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list UpsampleNearestExact3DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact3d_backward_symint(grad, output_size, self_sym_sizes, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
    args.collect(self_sym_sizes);
}
variable_list UpsampleNearestExact3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    saved.after(self_sym_sizes);
    return result;
}
variable_list PixelShuffleBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pixel_unshuffle(grad, upscale_factor)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PixelShuffleBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(upscale_factor);
}
variable_list PixelShuffleBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(upscale_factor);
    variable_list result = apply(variable_list(grads));
    saved.after(upscale_factor);
    return result;
}
variable_list PixelUnshuffleBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pixel_shuffle(grad, downscale_factor)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PixelUnshuffleBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(downscale_factor);
}
variable_list PixelUnshuffleBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(downscale_factor);
    variable_list result = apply(variable_list(grads));
    saved.after(downscale_factor);
    return result;
}
variable_list AdaptiveAvgPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool2d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveAvgPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AdaptiveAvgPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AdaptiveAvgPool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool3d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveAvgPool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list AdaptiveAvgPool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list AdaptiveMaxPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (adaptive_max_pool2d_backward(grad, self, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveMaxPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result1_);
}
variable_list AdaptiveMaxPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list AdaptiveMaxPool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (adaptive_max_pool3d_backward(grad, self, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveMaxPool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(result1_);
}
variable_list AdaptiveMaxPool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list AvgPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool2d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AvgPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(count_include_pad);
    args.collect(divisor_override);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
}
variable_list AvgPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(count_include_pad);
    saved.before(divisor_override);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(count_include_pad);
    saved.after(divisor_override);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    return result;
}
variable_list AvgPool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool3d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AvgPool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(count_include_pad);
    args.collect(divisor_override);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
}
variable_list AvgPool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(count_include_pad);
    saved.before(divisor_override);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(count_include_pad);
    saved.after(divisor_override);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    return result;
}
variable_list FractionalMaxPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fractional_max_pool2d_backward(grad, self, kernel_size, output_size, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FractionalMaxPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(kernel_size);
    args.collect(output_size);
    args.collect(self_);
    args.collect(result1_);
}
variable_list FractionalMaxPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(kernel_size);
    saved.before(output_size);
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(kernel_size);
    saved.after(output_size);
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list FractionalMaxPool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fractional_max_pool3d_backward(grad, self, kernel_size, output_size, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FractionalMaxPool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(kernel_size);
    args.collect(output_size);
    args.collect(self_);
    args.collect(result1_);
}
variable_list FractionalMaxPool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(kernel_size);
    saved.before(output_size);
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(kernel_size);
    saved.after(output_size);
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list LinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? linear_backward(input, grad, weight, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void LinearBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(input_);
    args.collect(weight_);
}
variable_list LinearBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(input_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(input_);
    saved.after(weight_);
    return result;
}
variable_list LinearBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, grad_output_ix, weight_ix })) {
  
    auto grad_result = linear_double_backward(grads, self, grad_output, weight);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void LinearBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(self_);
    args.collect(weight_);
}
variable_list LinearBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(self_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(self_);
    saved.after(weight_);
    return result;
}
variable_list MaxPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool2d_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
}
variable_list MaxPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    return result;
}
variable_list MpsConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? mps_convolution_backward_symint(self, grad, weight, padding, stride, dilation, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MpsConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MpsConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MpsConvolutionBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ grad_output_ix }),
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward_symint(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, std::vector<c10::SymInt>(padding.size(), 0), groups, grad_input_mask);
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MpsConvolutionBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(grad_output_);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MpsConvolutionBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(grad_output_);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(grad_output_);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MaxPool2DWithIndicesBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  jin_overwrite_maxpool2d_input(self);

  auto result1 = result1_.unpack(shared_from_this());
  jin_overwrite_maxpool2d_indices(result1);

  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool2d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool2DWithIndicesBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(result1_);
}
variable_list MaxPool2DWithIndicesBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(result1_);
    return result;
}
variable_list MaxPool3DWithIndicesBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool3DWithIndicesBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(result1_);
}
variable_list MaxPool3DWithIndicesBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(result1_);
    return result;
}
variable_list MaxUnpool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxUnpool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
}
variable_list MaxUnpool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    return result;
}
variable_list MaxUnpool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxUnpool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
}
variable_list MaxUnpool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    return result;
}
variable_list ConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();

  jin_overwrite_conv_input(input);
  jin_overwrite_conv_weight(weight);

  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, input, weight, bias_sym_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(transposed);
    args.collect(weight_);
}
variable_list ConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(transposed);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(transposed);
    saved.after(weight_);
    return result;
}
variable_list ConvolutionBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, input, weight, bias_sym_sizes_opt, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvolutionBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(transposed);
    args.collect(weight_);
}
variable_list ConvolutionBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(transposed);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(transposed);
    saved.after(weight_);
    return result;
}
variable_list ConvolutionBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ grad_output_ix, input_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ grad_output_ix }),
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward_symint(grads[0], grads[1], grads[2], grad_output, weight, input, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask);
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvolutionBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(grad_output_);
    args.collect(groups);
    args.collect(input_);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(transposed);
    args.collect(weight_);
}
variable_list ConvolutionBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(grad_output_);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(transposed);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(grad_output_);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(transposed);
    saved.after(weight_);
    return result;
}
variable_list ConvolutionOverrideableBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_overrideable_symint(grad, input, weight, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvolutionOverrideableBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(input_);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(transposed);
    args.collect(weight_);
}
variable_list ConvolutionOverrideableBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(transposed);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(transposed);
    saved.after(weight_);
    return result;
}
variable_list ConvolutionBackwardOverrideableBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ grad_output_ix, input_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ grad_output_ix }),
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward_symint(grads[0], grads[1], grads[2], grad_output, weight, input, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask);
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvolutionBackwardOverrideableBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(grad_output_);
    args.collect(groups);
    args.collect(input_);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(stride);
    args.collect(transposed);
    args.collect(weight_);
}
variable_list ConvolutionBackwardOverrideableBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(grad_output_);
    saved.before(groups);
    saved.before(input_);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(stride);
    saved.before(transposed);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(grad_output_);
    saved.after(groups);
    saved.after(input_);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(stride);
    saved.after(transposed);
    saved.after(weight_);
    return result;
}
variable_list SlowConvTranspose2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, true, output_padding, 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConvTranspose2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConvTranspose2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConvTranspose3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, true, output_padding, 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConvTranspose3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConvTranspose3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConv2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? _slow_conv2d_backward_symint(grad, self, weight, kernel_size, stride, padding, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConv2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConv2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConv2DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ grad_output_ix }),
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward_symint(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, {{1, 1}}, false, {{0, 0}}, 1, grad_input_mask);
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConv2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConv2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list ConvDepthwise2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad.contiguous(), self, weight, bias_sym_sizes_opt, stride, padding, dilation, /*transposed=*/ false, /*output_padding=*/ {{0, 0}}, /*groups=*/ 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvDepthwise2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list ConvDepthwise2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list ConvDepthwise3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad.contiguous(), self, weight, bias_sym_sizes_opt, stride, padding, dilation, /*transposed=*/ false, /*output_padding=*/ {{0, 0, 0}}, /*groups=*/ 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ConvDepthwise3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list ConvDepthwise3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConv3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, /*dilation=*/ {{1, 1, 1}}, false, /*output_padding=*/ {{0, 0, 0}}, 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConv3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConv3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConvDilated2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, false, std::vector<c10::SymInt>(padding.size(), 0), 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConvDilated2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConvDilated2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list SlowConvDilated3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, false, std::vector<c10::SymInt>(padding.size(), 0), 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SlowConvDilated3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list SlowConvDilated3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list Col2ImBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (im2col(grad, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Col2ImBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(stride);
}
variable_list Col2ImBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(stride);
    return result;
}
variable_list Im2ColBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (col2im_symint(grad, {self_sym_argsize_minus_2, self_sym_argsize_minus_1}, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void Im2ColBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_sym_argsize_minus_1);
    args.collect(self_sym_argsize_minus_2);
    args.collect(stride);
}
variable_list Im2ColBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_sym_argsize_minus_1);
    saved.before(self_sym_argsize_minus_2);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_sym_argsize_minus_1);
    saved.after(self_sym_argsize_minus_2);
    saved.after(stride);
    return result;
}
variable_list AdaptiveAvgPool2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool2d_symint(grad, {grad_output_sym_argsize_minus_2, grad_output_sym_argsize_minus_1})) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveAvgPool2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_sym_argsize_minus_1);
    args.collect(grad_output_sym_argsize_minus_2);
    args.collect(self_info);
}
variable_list AdaptiveAvgPool2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_sym_argsize_minus_1);
    saved.before(grad_output_sym_argsize_minus_2);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_sym_argsize_minus_1);
    saved.after(grad_output_sym_argsize_minus_2);
    saved.after(self_info);
    return result;
}
variable_list AdaptiveAvgPool3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool3d_symint(grad, { grad_output_sym_argsize_minus_3, grad_output_sym_argsize_minus_2, grad_output_sym_argsize_minus_1 })) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveAvgPool3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_sym_argsize_minus_1);
    args.collect(grad_output_sym_argsize_minus_2);
    args.collect(grad_output_sym_argsize_minus_3);
    args.collect(self_info);
}
variable_list AdaptiveAvgPool3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_sym_argsize_minus_1);
    saved.before(grad_output_sym_argsize_minus_2);
    saved.before(grad_output_sym_argsize_minus_3);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_sym_argsize_minus_1);
    saved.after(grad_output_sym_argsize_minus_2);
    saved.after(grad_output_sym_argsize_minus_3);
    saved.after(self_info);
    return result;
}
variable_list AdaptiveMaxPool2DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveMaxPool2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list AdaptiveMaxPool2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list AdaptiveMaxPool3DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AdaptiveMaxPool3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list AdaptiveMaxPool3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list AvgPool2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool2d(grad, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AvgPool2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(count_include_pad);
    args.collect(divisor_override);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_info);
    args.collect(stride);
}
variable_list AvgPool2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(count_include_pad);
    saved.before(divisor_override);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_info);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(count_include_pad);
    saved.after(divisor_override);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_info);
    saved.after(stride);
    return result;
}
variable_list AvgPool3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool3d(grad, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AvgPool3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(count_include_pad);
    args.collect(divisor_override);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_info);
    args.collect(stride);
}
variable_list AvgPool3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(count_include_pad);
    saved.before(divisor_override);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_info);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(count_include_pad);
    saved.after(divisor_override);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_info);
    saved.after(stride);
    return result;
}
variable_list EluBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_or_result_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self_or_result = self_or_result_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, is_result, self_or_result)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_or_result_ix })) {
    auto grad_result = any_grad_defined ? (elu_double_backward(grad, grad_output, alpha, scale, input_scale, is_result, self_or_result)) : Tensor();
    copy_range(grad_inputs, self_or_result_ix, grad_result);
  }
  return grad_inputs;
}
void EluBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(grad_output_);
    args.collect(input_scale);
    args.collect(is_result);
    args.collect(scale);
    args.collect(self_or_result_);
}
variable_list EluBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(grad_output_);
    saved.before(input_scale);
    saved.before(is_result);
    saved.before(scale);
    saved.before(self_or_result_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(grad_output_);
    saved.after(input_scale);
    saved.after(is_result);
    saved.after(scale);
    saved.after(self_or_result_);
    return result;
}
variable_list FractionalMaxPool2DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FractionalMaxPool2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list FractionalMaxPool2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list FractionalMaxPool3DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FractionalMaxPool3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list FractionalMaxPool3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list GluBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (glu_double_backward_grad_output(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (glu_double_backward(grad, grad_output, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void GluBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(grad_output_);
    args.collect(self_);
}
variable_list GluBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(grad_output_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(grad_output_);
    saved.after(self_);
    return result;
}
variable_list HardtanhBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (hardtanh_backward(grad, self, min_val, max_val)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void HardtanhBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(max_val);
    args.collect(min_val);
    args.collect(self_);
}
variable_list HardtanhBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(max_val);
    saved.before(min_val);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(max_val);
    saved.after(min_val);
    saved.after(self_);
    return result;
}
variable_list LogSigmoidBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto buffer = buffer_.unpack();
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_backward(grad, self, buffer)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_double_backward(grad * grad_output, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LogSigmoidBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(buffer_);
    args.collect(grad_output_);
    args.collect(self_);
}
variable_list LogSigmoidBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(buffer_);
    saved.before(grad_output_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(buffer_);
    saved.after(grad_output_);
    saved.after(self_);
    return result;
}
variable_list LogSoftmaxBackwardDataBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto output = output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad.to(output.dtype()) - (grad.to(output.dtype()) * output.exp()).sum(dim, true)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? ((-grad_output.sum(dim, true) * output.exp() * grad.to(output.dtype())).to(output.dtype())) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
void LogSoftmaxBackwardDataBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(grad_output_);
    args.collect(output_);
}
variable_list LogSoftmaxBackwardDataBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(grad_output_);
    saved.before(output_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(grad_output_);
    saved.after(output_);
    return result;
}
variable_list LeakyReluBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, self, negative_slope, false)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LeakyReluBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(negative_slope);
    args.collect(self_);
}
variable_list LeakyReluBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(negative_slope);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(negative_slope);
    saved.after(self_);
    return result;
}
variable_list MaxPool2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (error_for_max_pool2d_double_backward()) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_info);
}
variable_list MaxPool2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(self_info);
    return result;
}
variable_list MaxPool2DWithIndicesBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool2DWithIndicesBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list MaxPool2DWithIndicesBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list MaxPool3DWithIndicesBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MaxPool3DWithIndicesBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(indices_);
    args.collect(self_info);
}
variable_list MaxPool3DWithIndicesBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(indices_);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(indices_);
    saved.after(self_info);
    return result;
}
variable_list MseLossBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_double_backward(grad * grad_output, self, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-mse_loss_double_backward(grad * grad_output, target, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void MseLossBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list MseLossBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list NllLossBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss_symint(grad, target, weight, reduction, ignore_index)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NllLossBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ignore_index);
    args.collect(reduction);
    args.collect(target_);
    args.collect(weight_);
}
variable_list NllLossBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ignore_index);
    saved.before(reduction);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(ignore_index);
    saved.after(reduction);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list NllLoss2DBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss2d_symint(grad, target, weight, reduction, ignore_index)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NllLoss2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ignore_index);
    args.collect(reduction);
    args.collect(target_);
    args.collect(weight_);
}
variable_list NllLoss2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ignore_index);
    saved.before(reduction);
    saved.before(target_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(ignore_index);
    saved.after(reduction);
    saved.after(target_);
    saved.after(weight_);
    return result;
}
variable_list RreluWithNoiseBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto noise = noise_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void RreluWithNoiseBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lower);
    args.collect(noise_);
    args.collect(self_);
    args.collect(training);
    args.collect(upper);
}
variable_list RreluWithNoiseBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lower);
    saved.before(noise_);
    saved.before(self_);
    saved.before(training);
    saved.before(upper);
    variable_list result = apply(variable_list(grads));
    saved.after(lower);
    saved.after(noise_);
    saved.after(self_);
    saved.after(training);
    saved.after(upper);
    return result;
}
variable_list ReflectionPad1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad1d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad1DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReflectionPad1DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list ReflectionPad2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad2d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReflectionPad2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list ReflectionPad3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad3d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReflectionPad3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReflectionPad3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list ReplicationPad1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad1d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad1DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReplicationPad1DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list ReplicationPad2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad2d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReplicationPad2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list ReplicationPad3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad3d_symint(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReplicationPad3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(padding);
    args.collect(self_info);
}
variable_list ReplicationPad3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(padding);
    saved.before(self_info);
    variable_list result = apply(variable_list(grads));
    saved.after(padding);
    saved.after(self_info);
    return result;
}
variable_list SparseSampledAddmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat1_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mat1 = mat1_.unpack();
  auto mat2 = mat2_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, mat1_ix, mat2_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ mat1_ix }),
        task_should_compute_output({ mat2_ix }),
      };
    auto grad_result = sparse_sampled_addmm_backward(grad, self, wrap_opt_if(mat1, grad_input_mask[2]), wrap_opt_if(mat2, grad_input_mask[1]), alpha, beta, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ mat1_ix })) {
        copy_range(grad_inputs, mat1_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ mat2_ix })) {
        copy_range(grad_inputs, mat2_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void SparseSampledAddmmBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(beta);
    args.collect(mat1_);
    args.collect(mat2_);
    args.collect(self_);
}
variable_list SparseSampledAddmmBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(beta);
    saved.before(mat1_);
    saved.before(mat2_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(beta);
    saved.after(mat1_);
    saved.after(mat2_);
    saved.after(self_);
    return result;
}
variable_list SparseMmReduceImplBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, other_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ other_ix }),
      };
    auto grad_result = grad.defined() ? _sparse_mm_reduce_impl_backward(self, grad, other, reduce, result1, grad_input_mask) :  std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ other_ix })) {
        copy_range(grad_inputs, other_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void SparseMmReduceImplBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(reduce);
    args.collect(self_);
    args.collect(result1_);
}
variable_list SparseMmReduceImplBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(reduce);
    saved.before(self_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(reduce);
    saved.after(self_);
    saved.after(result1_);
    return result;
}
variable_list SmoothL1LossBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_backward(grad, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_double_backward(grad * grad_output, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-smooth_l1_loss_double_backward(grad * grad_output, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void SmoothL1LossBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(beta);
    args.collect(grad_output_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list SmoothL1LossBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(beta);
    saved.before(grad_output_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(beta);
    saved.after(grad_output_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list HuberLossBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_double_backward_grad_output(grad, grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_double_backward(grad * grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-huber_loss_double_backward(grad * grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
void HuberLossBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(delta);
    args.collect(grad_output_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list HuberLossBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(delta);
    saved.before(grad_output_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(delta);
    saved.after(grad_output_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list SoftplusBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (softplus_backward(grad, self, beta, threshold)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softplus_double_backward(grad * grad_output, self, beta, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftplusBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(beta);
    args.collect(grad_output_);
    args.collect(self_);
    args.collect(threshold);
}
variable_list SoftplusBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(beta);
    saved.before(grad_output_);
    saved.before(self_);
    saved.before(threshold);
    variable_list result = apply(variable_list(grads));
    saved.after(beta);
    saved.after(grad_output_);
    saved.after(self_);
    saved.after(threshold);
    return result;
}
variable_list SoftmaxBackwardDataBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto output = output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_softmax_backward_data(grad.to(output.dtype()), output, dim, input_dtype)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? (softmax_double_backward(grad.to(output.dtype()), grad_output, dim, output).to(output.dtype())) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
void SoftmaxBackwardDataBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(grad_output_);
    args.collect(input_dtype);
    args.collect(output_);
}
variable_list SoftmaxBackwardDataBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(grad_output_);
    saved.before(input_dtype);
    saved.before(output_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(grad_output_);
    saved.after(input_dtype);
    saved.after(output_);
    return result;
}
variable_list SoftMarginLossBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_double_backward_grad_output(grad, grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_double_backward(grad * grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftMarginLossBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(reduction);
    args.collect(self_);
    args.collect(target_);
}
variable_list SoftMarginLossBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(reduction);
    saved.before(self_);
    saved.before(target_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(reduction);
    saved.after(self_);
    saved.after(target_);
    return result;
}
variable_list SoftshrinkBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (softshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SoftshrinkBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(lambd);
    args.collect(self_);
}
variable_list SoftshrinkBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(lambd);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(lambd);
    saved.after(self_);
    return result;
}
variable_list ThresholdBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, threshold)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ThresholdBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(threshold);
}
variable_list ThresholdBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(threshold);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(threshold);
    return result;
}
variable_list UpsampleLinear1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d_symint(grad, output_size, align_corners, scales)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleLinear1DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales);
}
variable_list UpsampleLinear1DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales);
    return result;
}
variable_list UpsampleBilinear2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d_symint(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBilinear2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleBilinear2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleBilinear2DAaBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_bilinear2d_aa_symint(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBilinear2DAaBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleBilinear2DAaBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleBicubic2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d_symint(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBicubic2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleBicubic2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleBicubic2DAaBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_bicubic2d_aa_symint(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleBicubic2DAaBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleBicubic2DAaBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleTrilinear3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d_symint(grad, output_size, align_corners, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleTrilinear3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(align_corners);
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleTrilinear3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(align_corners);
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(align_corners);
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleNearest1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d_symint(grad, output_size, scales)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest1DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales);
}
variable_list UpsampleNearest1DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales);
    return result;
}
variable_list UpsampleNearestExact1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact1d_symint(grad, output_size, scales)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact1DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales);
}
variable_list UpsampleNearestExact1DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales);
    return result;
}
variable_list UpsampleNearest2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d_symint(grad, output_size, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleNearest2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleNearestExact2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact2d_symint(grad, output_size, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact2DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleNearestExact2DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleNearest3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d_symint(grad, output_size, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearest3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleNearest3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list UpsampleNearestExact3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_upsample_nearest_exact3d_symint(grad, output_size, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
void UpsampleNearestExact3DBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(output_size);
    args.collect(scales_d);
    args.collect(scales_h);
    args.collect(scales_w);
}
variable_list UpsampleNearestExact3DBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(output_size);
    saved.before(scales_d);
    saved.before(scales_h);
    saved.before(scales_w);
    variable_list result = apply(variable_list(grads));
    saved.after(output_size);
    saved.after(scales_d);
    saved.after(scales_h);
    saved.after(scales_w);
    return result;
}
variable_list SigmoidBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto output = output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (sigmoid_backward(grad, output.conj())) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * grad_output * (-2 * output.conj() + 1)) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
void SigmoidBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(output_);
}
variable_list SigmoidBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(output_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(output_);
    return result;
}
variable_list TanhBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto output = output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (tanh_backward(grad, output.conj())) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * (-2 * output.conj() * grad_output)) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
void TanhBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grad_output_);
    args.collect(output_);
}
variable_list TanhBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grad_output_);
    saved.before(output_);
    variable_list result = apply(variable_list(grads));
    saved.after(grad_output_);
    saved.after(output_);
    return result;
}
variable_list CudnnCtcLossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_cudnn_ctc_loss_backward(grad, result0, result1, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
void CudnnCtcLossBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(zero_infinity);
    args.collect(result0_);
    args.collect(result1_);
}
variable_list CudnnCtcLossBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(zero_infinity);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(zero_infinity);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list CudnnCtcLossBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_cudnn_ctc_loss_backward(grad, result0, result1, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
void CudnnCtcLossBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(zero_infinity);
    args.collect(result0_);
    args.collect(result1_);
}
variable_list CudnnCtcLossBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(zero_infinity);
    saved.before(result0_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(zero_infinity);
    saved.after(result0_);
    saved.after(result1_);
    return result;
}
variable_list CudnnConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _cudnn_convolution_backward(self, grad, weight, padding, output_padding, stride, dilation, true, groups, {grad_input_mask[0], grad_input_mask[1]});
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void CudnnConvolutionTransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list CudnnConvolutionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MpsConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = grad.defined() ? mps_convolution_transpose_backward_symint(self, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void MpsConvolutionTransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MpsConvolutionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list CudnnConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _cudnn_convolution_backward(self, grad, weight, padding, std::vector<c10::SymInt>(padding.size(), 0), stride, dilation, false, groups, {grad_input_mask[0], grad_input_mask[1]});
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void CudnnConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dilation);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list CudnnConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dilation);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(dilation);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list CudnnGridSamplerBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto grid = grid_.unpack();
  auto self = self_.unpack();
  if (task_should_compute_output({ self_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? cudnn_grid_sampler_backward(self, grid, grad) : std::tuple<Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void CudnnGridSamplerBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(grid_);
    args.collect(self_);
}
variable_list CudnnGridSamplerBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(grid_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(grid_);
    saved.after(self_);
    return result;
}
variable_list CudnnAffineGridGeneratorBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto theta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ theta_ix })) {
    auto grad_result = any_grad_defined ? (cudnn_affine_grid_generator_backward(grad, N, C, H, W)) : Tensor();
    copy_range(grad_inputs, theta_ix, grad_result);
  }
  return grad_inputs;
}
void CudnnAffineGridGeneratorBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(C);
    args.collect(H);
    args.collect(N);
    args.collect(W);
}
variable_list CudnnAffineGridGeneratorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(C);
    saved.before(H);
    saved.before(N);
    saved.before(W);
    variable_list result = apply(variable_list(grads));
    saved.after(C);
    saved.after(H);
    saved.after(N);
    saved.after(W);
    return result;
}
variable_list CudnnBatchNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? (training ? cudnn_batch_norm_backward(input, grad.contiguous(input.suggest_memory_format()), weight, running_mean, running_var, result1, result2, epsilon, retain_variables ? result3.clone() : result3) : native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, epsilon, grad_input_mask)) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void CudnnBatchNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(epsilon);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(training);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
    args.collect(result3_);
}
variable_list CudnnBatchNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(epsilon);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(training);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    variable_list result = apply(variable_list(grads));
    saved.after(epsilon);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(training);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    return result;
}
variable_list CudnnBatchNormBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_var_ix = gen.range(1);
  auto reserveSpace_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto input = input_.unpack();
  auto reserveSpace = reserveSpace_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_mean = save_mean_.unpack();
  auto save_var = save_var_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, grad_output_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ grad_output_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_output, running_mean, running_var, true, epsilon, save_mean, save_var, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<2>(grad_result));
      }
  }
  if (task_should_compute_output({ reserveSpace_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward reserveSpace");
    copy_range(grad_inputs, reserveSpace_ix, grad_result);
  }
  if (task_should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  if (task_should_compute_output({ save_var_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward save_var");
    copy_range(grad_inputs, save_var_ix, grad_result);
  }
  return grad_inputs;
}
void CudnnBatchNormBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(epsilon);
    args.collect(grad_output_);
    args.collect(input_);
    args.collect(reserveSpace_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(save_mean_);
    args.collect(save_var_);
    args.collect(weight_);
}
variable_list CudnnBatchNormBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(epsilon);
    saved.before(grad_output_);
    saved.before(input_);
    saved.before(reserveSpace_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(save_mean_);
    saved.before(save_var_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(epsilon);
    saved.after(grad_output_);
    saved.after(input_);
    saved.after(reserveSpace_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(save_mean_);
    saved.after(save_var_);
    saved.after(weight_);
    return result;
}
variable_list NnpackSpatialConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, input, weight, bias_sym_sizes_opt, stride, padding, std::vector<c10::SymInt>(padding.size(), 1), false, std::vector<c10::SymInt>(padding.size(), 0), 1, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void NnpackSpatialConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(input_);
    args.collect(padding);
    args.collect(stride);
    args.collect(weight_);
}
variable_list NnpackSpatialConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(input_);
    saved.before(padding);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(input_);
    saved.after(padding);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list LstmMpsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!hx_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!params_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(hx_size_);
  auto params_ix = gen.range(params_size_);
  variable_list grad_inputs(gen.size());
  auto hx = unpack_list(hx_, nullptr);
  auto input = input_.unpack();
  auto params = unpack_list(params_, nullptr);
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, hx_ix, params_ix })) {
  
    auto grad_result = lstm_mps_backward(grads[0], grads[1], grads[2], result3, result4, input, result5, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ params_ix })) {
        copy_range(grad_inputs, params_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void LstmMpsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_first);
    args.collect(bidirectional);
    args.collect(dropout);
    args.collect(has_biases);
    args.collect(hx_);
    args.collect(input_);
    args.collect(num_layers);
    args.collect(params_);
    args.collect(train);
    args.collect(result3_);
    args.collect(result4_);
    args.collect(result5_);
}
variable_list LstmMpsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_first);
    saved.before(bidirectional);
    saved.before(dropout);
    saved.before(has_biases);
    saved.before(hx_);
    saved.before(input_);
    saved.before(num_layers);
    saved.before(params_);
    saved.before(train);
    saved.before(result3_);
    saved.before(result4_);
    saved.before(result5_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_first);
    saved.after(bidirectional);
    saved.after(dropout);
    saved.after(has_biases);
    saved.after(hx_);
    saved.after(input_);
    saved.after(num_layers);
    saved.after(params_);
    saved.after(train);
    saved.after(result3_);
    saved.after(result4_);
    saved.after(result5_);
    return result;
}
variable_list CudnnRnnBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!weight_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto cx = cx_.unpack();
  auto dropout_state = dropout_state_.unpack();
  auto hx = hx_.unpack();
  auto input = input_.unpack();
  auto weight = unpack_list(weight_, nullptr);
  auto result0 = result0_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, hx_ix, cx_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 4>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ hx_ix }),
        task_should_compute_output({ cx_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = _cudnn_rnn_backward_symint(input, weight, weight_stride0, result4, hx, cx, result0, grads[0], grads[1], grads[2], mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, retain_variables ? result3.clone() : result3, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void CudnnRnnBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_first);
    args.collect(batch_sizes);
    args.collect(bidirectional);
    args.collect(cx_);
    args.collect(dropout);
    args.collect(dropout_state_);
    args.collect(hidden_size);
    args.collect(hx_);
    args.collect(input_);
    args.collect(mode);
    args.collect(num_layers);
    args.collect(proj_size);
    args.collect(train);
    args.collect(weight_);
    args.collect(weight_stride0);
    args.collect(result0_);
    args.collect(result3_);
    args.collect(result4_);
}
variable_list CudnnRnnBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_first);
    saved.before(batch_sizes);
    saved.before(bidirectional);
    saved.before(cx_);
    saved.before(dropout);
    saved.before(dropout_state_);
    saved.before(hidden_size);
    saved.before(hx_);
    saved.before(input_);
    saved.before(mode);
    saved.before(num_layers);
    saved.before(proj_size);
    saved.before(train);
    saved.before(weight_);
    saved.before(weight_stride0);
    saved.before(result0_);
    saved.before(result3_);
    saved.before(result4_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_first);
    saved.after(batch_sizes);
    saved.after(bidirectional);
    saved.after(cx_);
    saved.after(dropout);
    saved.after(dropout_state_);
    saved.after(hidden_size);
    saved.after(hx_);
    saved.after(input_);
    saved.after(mode);
    saved.after(num_layers);
    saved.after(proj_size);
    saved.after(train);
    saved.after(weight_);
    saved.after(weight_stride0);
    saved.after(result0_);
    saved.after(result3_);
    saved.after(result4_);
    return result;
}
variable_list CudnnRnnBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  auto output_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto grad_hy_ix = gen.range(1);
  auto grad_cy_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ cx_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, cx_ix, grad_result);
  }
  if (task_should_compute_output({ grad_cy_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_cy_ix, grad_result);
  }
  if (task_should_compute_output({ grad_hy_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_hy_ix, grad_result);
  }
  if (task_should_compute_output({ grad_output_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (task_should_compute_output({ hx_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, hx_ix, grad_result);
  }
  if (task_should_compute_output({ input_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, input_ix, grad_result);
  }
  if (task_should_compute_output({ output_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, output_ix, grad_result);
  }
  if (task_should_compute_output({ weight_ix })) {
    auto grad_result = not_implemented_list("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
void CudnnRnnBackwardBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list CudnnRnnBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list MiopenConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, true, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MiopenConvolutionTransposeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(output_padding);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MiopenConvolutionTransposeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(output_padding);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(output_padding);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MiopenConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, false, std::vector<c10::SymInt>(padding.size(), 0), groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MiopenConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MiopenConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MiopenDepthwiseConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, false, std::vector<c10::SymInt>(padding.size(), 0), groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MiopenDepthwiseConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MiopenDepthwiseConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MiopenBatchNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? (training ? miopen_batch_norm_backward(input, grad.contiguous(), weight, running_mean, running_var, result1, result2, epsilon) : native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, epsilon, grad_input_mask)) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MiopenBatchNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(epsilon);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(training);
    args.collect(weight_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list MiopenBatchNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(epsilon);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(training);
    saved.before(weight_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(epsilon);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(training);
    saved.after(weight_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list MiopenBatchNormBackwardBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_var_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto input = input_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_mean = save_mean_.unpack();
  auto save_var = save_var_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ input_ix, weight_ix, grad_output_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ grad_output_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_output, running_mean, running_var, true, epsilon, save_mean, save_var, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<2>(grad_result));
      }
  }
  if (task_should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("miopen_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  if (task_should_compute_output({ save_var_ix })) {
    auto grad_result = not_implemented("miopen_batch_norm_backward save_var");
    copy_range(grad_inputs, save_var_ix, grad_result);
  }
  return grad_inputs;
}
void MiopenBatchNormBackwardBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(epsilon);
    args.collect(grad_output_);
    args.collect(input_);
    args.collect(running_mean_);
    args.collect(running_var_);
    args.collect(save_mean_);
    args.collect(save_var_);
    args.collect(weight_);
}
variable_list MiopenBatchNormBackwardBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(epsilon);
    saved.before(grad_output_);
    saved.before(input_);
    saved.before(running_mean_);
    saved.before(running_var_);
    saved.before(save_mean_);
    saved.before(save_var_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(epsilon);
    saved.after(grad_output_);
    saved.after(input_);
    saved.after(running_mean_);
    saved.after(running_var_);
    saved.after(save_mean_);
    saved.after(save_var_);
    saved.after(weight_);
    return result;
}
variable_list MiopenRnnBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!weight_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto cx = cx_.unpack();
  auto dropout_state = dropout_state_.unpack();
  auto hx = hx_.unpack();
  auto input = input_.unpack();
  auto weight = unpack_list(weight_, nullptr);
  auto result0 = result0_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, hx_ix, cx_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 4>{
        task_should_compute_output({ input_ix }),
        task_should_compute_output({ hx_ix }),
        task_should_compute_output({ cx_ix }),
        task_should_compute_output({ weight_ix }),
      };
    auto grad_result = miopen_rnn_backward(input, weight, weight_stride0, result4, hx, cx, result0, grads[0], grads[1], grads[2], mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, retain_variables ? result3.clone() : result3, grad_input_mask);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void MiopenRnnBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_first);
    args.collect(batch_sizes);
    args.collect(bidirectional);
    args.collect(cx_);
    args.collect(dropout);
    args.collect(dropout_state_);
    args.collect(hidden_size);
    args.collect(hx_);
    args.collect(input_);
    args.collect(mode);
    args.collect(num_layers);
    args.collect(train);
    args.collect(weight_);
    args.collect(weight_stride0);
    args.collect(result0_);
    args.collect(result3_);
    args.collect(result4_);
}
variable_list MiopenRnnBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_first);
    saved.before(batch_sizes);
    saved.before(bidirectional);
    saved.before(cx_);
    saved.before(dropout);
    saved.before(dropout_state_);
    saved.before(hidden_size);
    saved.before(hx_);
    saved.before(input_);
    saved.before(mode);
    saved.before(num_layers);
    saved.before(train);
    saved.before(weight_);
    saved.before(weight_stride0);
    saved.before(result0_);
    saved.before(result3_);
    saved.before(result4_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_first);
    saved.after(batch_sizes);
    saved.after(bidirectional);
    saved.after(cx_);
    saved.after(dropout);
    saved.after(dropout_state_);
    saved.after(hidden_size);
    saved.after(hx_);
    saved.after(input_);
    saved.after(mode);
    saved.after(num_layers);
    saved.after(train);
    saved.after(weight_);
    saved.after(weight_stride0);
    saved.after(result0_);
    saved.after(result3_);
    saved.after(result4_);
    return result;
}
variable_list MkldnnRnnLayerBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight0_ix = gen.range(1);
  auto weight1_ix = gen.range(1);
  auto weight2_ix = gen.range(1);
  auto weight3_ix = gen.range(1);
  auto hx__ix = gen.range(1);
  auto cx__ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto cx_ = cx__.unpack();
  auto hx_ = hx__.unpack();
  auto input = input_.unpack();
  auto weight0 = weight0_.unpack();
  auto weight1 = weight1_.unpack();
  auto weight2 = weight2_.unpack();
  auto weight3 = weight3_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (task_should_compute_output({ input_ix, weight0_ix, weight1_ix, weight2_ix, weight3_ix, hx__ix, cx__ix })) {
  
    auto grad_result = GradMode::is_enabled() ? mkldnn_rnn_layer_differentiable_backward(input, weight0, weight1, weight2, weight3, hx_, cx_, result0, result1, result2, grads[0], grads[1], grads[2], reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, result3) : mkldnn_rnn_layer_backward(input, weight0, weight1, weight2, weight3, hx_, cx_, result0, result1, result2, grads[0], grads[1], grads[2], reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, result3);
      if (task_should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight0_ix })) {
        copy_range(grad_inputs, weight0_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ weight1_ix })) {
        copy_range(grad_inputs, weight1_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ weight2_ix })) {
        copy_range(grad_inputs, weight2_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ weight3_ix })) {
        copy_range(grad_inputs, weight3_ix, std::get<4>(grad_result));
      }
      if (task_should_compute_output({ hx__ix })) {
        copy_range(grad_inputs, hx__ix, std::get<5>(grad_result));
      }
      if (task_should_compute_output({ cx__ix })) {
        copy_range(grad_inputs, cx__ix, std::get<6>(grad_result));
      }
  }
  return grad_inputs;
}
void MkldnnRnnLayerBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_first);
    args.collect(batch_sizes);
    args.collect(bidirectional);
    args.collect(cx__);
    args.collect(has_biases);
    args.collect(hidden_size);
    args.collect(hx__);
    args.collect(input_);
    args.collect(mode);
    args.collect(num_layers);
    args.collect(reverse);
    args.collect(train);
    args.collect(weight0_);
    args.collect(weight1_);
    args.collect(weight2_);
    args.collect(weight3_);
    args.collect(result0_);
    args.collect(result1_);
    args.collect(result2_);
    args.collect(result3_);
}
variable_list MkldnnRnnLayerBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_first);
    saved.before(batch_sizes);
    saved.before(bidirectional);
    saved.before(cx__);
    saved.before(has_biases);
    saved.before(hidden_size);
    saved.before(hx__);
    saved.before(input_);
    saved.before(mode);
    saved.before(num_layers);
    saved.before(reverse);
    saved.before(train);
    saved.before(weight0_);
    saved.before(weight1_);
    saved.before(weight2_);
    saved.before(weight3_);
    saved.before(result0_);
    saved.before(result1_);
    saved.before(result2_);
    saved.before(result3_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_first);
    saved.after(batch_sizes);
    saved.after(bidirectional);
    saved.after(cx__);
    saved.after(has_biases);
    saved.after(hidden_size);
    saved.after(hx__);
    saved.after(input_);
    saved.after(mode);
    saved.after(num_layers);
    saved.after(reverse);
    saved.after(train);
    saved.after(weight0_);
    saved.after(weight1_);
    saved.after(weight2_);
    saved.after(weight3_);
    saved.after(result0_);
    saved.after(result1_);
    saved.after(result2_);
    saved.after(result3_);
    return result;
}
variable_list MkldnnConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_symint(grad, self, weight, bias_sym_sizes_opt, stride, padding, dilation, /*transposed=*/ false, /*output_padding=*/ std::vector<c10::SymInt>(padding.size(), 0), groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MkldnnConvolutionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_sym_sizes_opt);
    args.collect(dilation);
    args.collect(groups);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(weight_);
}
variable_list MkldnnConvolutionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_sym_sizes_opt);
    saved.before(dilation);
    saved.before(groups);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_sym_sizes_opt);
    saved.after(dilation);
    saved.after(groups);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(weight_);
    return result;
}
variable_list MkldnnLinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (task_should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        task_should_compute_output({ self_ix }),
        task_should_compute_output({ weight_ix }),
        task_should_compute_output({ bias_ix }),
      };
    auto grad_result = mkldnn_linear_backward(self, grad, weight, grad_input_mask);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void MkldnnLinearBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(weight_);
}
variable_list MkldnnLinearBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(weight_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(weight_);
    return result;
}
variable_list MkldnnMaxPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_max_pool2d_backward(grad, result, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MkldnnMaxPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(result_);
}
variable_list MkldnnMaxPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(result_);
    return result;
}
variable_list MkldnnMaxPool3DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_max_pool3d_backward(grad, result, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MkldnnMaxPool3DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ceil_mode);
    args.collect(dilation);
    args.collect(kernel_size);
    args.collect(padding);
    args.collect(self_);
    args.collect(stride);
    args.collect(result_);
}
variable_list MkldnnMaxPool3DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ceil_mode);
    saved.before(dilation);
    saved.before(kernel_size);
    saved.before(padding);
    saved.before(self_);
    saved.before(stride);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(ceil_mode);
    saved.after(dilation);
    saved.after(kernel_size);
    saved.after(padding);
    saved.after(self_);
    saved.after(stride);
    saved.after(result_);
    return result;
}
variable_list MkldnnAdaptiveAvgPool2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_adaptive_avg_pool2d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MkldnnAdaptiveAvgPool2DBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list MkldnnAdaptiveAvgPool2DBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list MkldnnReshapeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void MkldnnReshapeBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list MkldnnReshapeBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list NestedTensorFromTensorListBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!list_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto list_ix = gen.range(list_size_);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto list = unpack_list(list_, nullptr);
  if (task_should_compute_output({ list_ix })) {
    auto grad_result = grad.defined()? at::unbind(grad) : std::vector<Tensor>(list.size());
    copy_range(grad_inputs, list_ix, grad_result);
  }
  return grad_inputs;
}
void NestedTensorFromTensorListBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(list_);
}
variable_list NestedTensorFromTensorListBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(list_);
    variable_list result = apply(variable_list(grads));
    saved.after(list_);
    return result;
}
variable_list NestedTensorFromMaskBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto t_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ t_ix })) {
    auto grad_result = any_grad_defined ? (grad.to_padded_tensor_symint(0, t_sym_sizes)) : Tensor();
    copy_range(grad_inputs, t_ix, grad_result);
  }
  return grad_inputs;
}
void NestedTensorFromMaskBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(t_sym_sizes);
}
variable_list NestedTensorFromMaskBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(t_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(t_sym_sizes);
    return result;
}
variable_list NestedFromPaddedBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto padded_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto padded = padded_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ padded_ix })) {
    auto grad_result = any_grad_defined ? (_nested_from_padded_backward(grad, padded, fuse_transform_0213)) : Tensor();
    copy_range(grad_inputs, padded_ix, grad_result);
  }
  return grad_inputs;
}
void NestedFromPaddedBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(fuse_transform_0213);
    args.collect(padded_);
}
variable_list NestedFromPaddedBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(fuse_transform_0213);
    saved.before(padded_);
    variable_list result = apply(variable_list(grads));
    saved.after(fuse_transform_0213);
    saved.after(padded_);
    return result;
}
variable_list ToPaddedTensorBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_nested_from_padded(grad, self._nested_tensor_size())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ToPaddedTensorBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ToPaddedTensorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list NestedViewFromBufferBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.values()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NestedViewFromBufferBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list NestedViewFromBufferBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ScaledDotProductEfficientAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto attn_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto attn_bias = attn_bias_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto log_sumexp = log_sumexp_.unpack(shared_from_this());
  auto output = output_.unpack(shared_from_this());
  auto philox_offset = philox_offset_.unpack(shared_from_this());
  auto philox_seed = philox_seed_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix, attn_bias_ix })) {
      auto grad_input_mask = std::array<bool, 4>{
        task_should_compute_output({ query_ix }),
        task_should_compute_output({ key_ix }),
        task_should_compute_output({ value_ix }),
        task_should_compute_output({ attn_bias_ix }),
      };
    auto grad_result = _scaled_dot_product_efficient_attention_backward(grad, query, key, value, attn_bias, output, log_sumexp, philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ attn_bias_ix })) {
        copy_range(grad_inputs, attn_bias_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void ScaledDotProductEfficientAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(attn_bias_);
    args.collect(dropout_p);
    args.collect(is_causal);
    args.collect(key_);
    args.collect(query_);
    args.collect(scale);
    args.collect(value_);
    args.collect(log_sumexp_);
    args.collect(output_);
    args.collect(philox_offset_);
    args.collect(philox_seed_);
}
variable_list ScaledDotProductEfficientAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(attn_bias_);
    saved.before(dropout_p);
    saved.before(is_causal);
    saved.before(key_);
    saved.before(query_);
    saved.before(scale);
    saved.before(value_);
    saved.before(log_sumexp_);
    saved.before(output_);
    saved.before(philox_offset_);
    saved.before(philox_seed_);
    variable_list result = apply(variable_list(grads));
    saved.after(attn_bias_);
    saved.after(dropout_p);
    saved.after(is_causal);
    saved.after(key_);
    saved.after(query_);
    saved.after(scale);
    saved.after(value_);
    saved.after(log_sumexp_);
    saved.after(output_);
    saved.after(philox_offset_);
    saved.after(philox_seed_);
    return result;
}
variable_list ScaledDotProductFlashAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto cum_seq_k = cum_seq_k_.unpack(shared_from_this());
  auto cum_seq_q = cum_seq_q_.unpack(shared_from_this());
  auto logsumexp = logsumexp_.unpack(shared_from_this());
  auto output = output_.unpack(shared_from_this());
  auto philox_offset = philox_offset_.unpack(shared_from_this());
  auto philox_seed = philox_seed_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix })) {
  
    auto grad_result = _scaled_dot_product_flash_attention_backward_symint(grad, query, key, value, output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, scale);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void ScaledDotProductFlashAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dropout_p);
    args.collect(is_causal);
    args.collect(key_);
    args.collect(query_);
    args.collect(scale);
    args.collect(value_);
    args.collect(cum_seq_k_);
    args.collect(cum_seq_q_);
    args.collect(logsumexp_);
    args.collect(max_k);
    args.collect(max_q);
    args.collect(output_);
    args.collect(philox_offset_);
    args.collect(philox_seed_);
}
variable_list ScaledDotProductFlashAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dropout_p);
    saved.before(is_causal);
    saved.before(key_);
    saved.before(query_);
    saved.before(scale);
    saved.before(value_);
    saved.before(cum_seq_k_);
    saved.before(cum_seq_q_);
    saved.before(logsumexp_);
    saved.before(max_k);
    saved.before(max_q);
    saved.before(output_);
    saved.before(philox_offset_);
    saved.before(philox_seed_);
    variable_list result = apply(variable_list(grads));
    saved.after(dropout_p);
    saved.after(is_causal);
    saved.after(key_);
    saved.after(query_);
    saved.after(scale);
    saved.after(value_);
    saved.after(cum_seq_k_);
    saved.after(cum_seq_q_);
    saved.after(logsumexp_);
    saved.after(max_k);
    saved.after(max_q);
    saved.after(output_);
    saved.after(philox_offset_);
    saved.after(philox_seed_);
    return result;
}
variable_list FlashAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto cum_seq_k = cum_seq_k_.unpack();
  auto cum_seq_q = cum_seq_q_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto output = output_.unpack(shared_from_this());
  auto philox_offset = philox_offset_.unpack(shared_from_this());
  auto philox_seed = philox_seed_.unpack(shared_from_this());
  auto softmax_logsumexp = softmax_logsumexp_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix })) {
  
    auto grad_result = _flash_attention_backward_symint(grad, query, key, value, output, softmax_logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, scale);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
void FlashAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(cum_seq_k_);
    args.collect(cum_seq_q_);
    args.collect(dropout_p);
    args.collect(is_causal);
    args.collect(key_);
    args.collect(max_k);
    args.collect(max_q);
    args.collect(query_);
    args.collect(scale);
    args.collect(value_);
    args.collect(output_);
    args.collect(philox_offset_);
    args.collect(philox_seed_);
    args.collect(softmax_logsumexp_);
}
variable_list FlashAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(cum_seq_k_);
    saved.before(cum_seq_q_);
    saved.before(dropout_p);
    saved.before(is_causal);
    saved.before(key_);
    saved.before(max_k);
    saved.before(max_q);
    saved.before(query_);
    saved.before(scale);
    saved.before(value_);
    saved.before(output_);
    saved.before(philox_offset_);
    saved.before(philox_seed_);
    saved.before(softmax_logsumexp_);
    variable_list result = apply(variable_list(grads));
    saved.after(cum_seq_k_);
    saved.after(cum_seq_q_);
    saved.after(dropout_p);
    saved.after(is_causal);
    saved.after(key_);
    saved.after(max_k);
    saved.after(max_q);
    saved.after(query_);
    saved.after(scale);
    saved.after(value_);
    saved.after(output_);
    saved.after(philox_offset_);
    saved.after(philox_seed_);
    saved.after(softmax_logsumexp_);
    return result;
}
variable_list EfficientAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto bias = bias_.unpack();
  auto cu_seqlens_k = cu_seqlens_k_.unpack();
  auto cu_seqlens_q = cu_seqlens_q_.unpack();
  auto key = key_.unpack();
  auto query = query_.unpack();
  auto value = value_.unpack();
  auto logsumexp = logsumexp_.unpack(shared_from_this());
  auto output = output_.unpack(shared_from_this());
  auto philox_offset = philox_offset_.unpack(shared_from_this());
  auto philox_seed = philox_seed_.unpack(shared_from_this());
  if (task_should_compute_output({ query_ix, key_ix, value_ix, bias_ix })) {
  
    auto grad_result = _efficient_attention_backward_symint(grad, query, key, value, bias, output, cu_seqlens_q, cu_seqlens_k, max_seqlen_batch_q, max_seqlen_batch_k, logsumexp, dropout_p, philox_seed, philox_offset, custom_mask_type, bias.requires_grad(), scale);
      if (task_should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
void EfficientAttentionBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(bias_);
    args.collect(cu_seqlens_k_);
    args.collect(cu_seqlens_q_);
    args.collect(custom_mask_type);
    args.collect(dropout_p);
    args.collect(key_);
    args.collect(query_);
    args.collect(scale);
    args.collect(value_);
    args.collect(logsumexp_);
    args.collect(max_seqlen_batch_k);
    args.collect(max_seqlen_batch_q);
    args.collect(output_);
    args.collect(philox_offset_);
    args.collect(philox_seed_);
}
variable_list EfficientAttentionBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(bias_);
    saved.before(cu_seqlens_k_);
    saved.before(cu_seqlens_q_);
    saved.before(custom_mask_type);
    saved.before(dropout_p);
    saved.before(key_);
    saved.before(query_);
    saved.before(scale);
    saved.before(value_);
    saved.before(logsumexp_);
    saved.before(max_seqlen_batch_k);
    saved.before(max_seqlen_batch_q);
    saved.before(output_);
    saved.before(philox_offset_);
    saved.before(philox_seed_);
    variable_list result = apply(variable_list(grads));
    saved.after(bias_);
    saved.after(cu_seqlens_k_);
    saved.after(cu_seqlens_q_);
    saved.after(custom_mask_type);
    saved.after(dropout_p);
    saved.after(key_);
    saved.after(query_);
    saved.after(scale);
    saved.after(value_);
    saved.after(logsumexp_);
    saved.after(max_seqlen_batch_k);
    saved.after(max_seqlen_batch_q);
    saved.after(output_);
    saved.after(philox_offset_);
    saved.after(philox_seed_);
    return result;
}
variable_list FftR2CBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_r2c_backward(grad, dim, normalization, onesided, self.sym_size(dim.back()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FftR2CBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(normalization);
    args.collect(onesided);
    args.collect(self_);
}
variable_list FftR2CBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(normalization);
    saved.before(onesided);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(normalization);
    saved.after(onesided);
    saved.after(self_);
    return result;
}
variable_list FftC2RBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_c2r_backward(grad, dim, normalization)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FftC2RBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(normalization);
}
variable_list FftC2RBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(normalization);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(normalization);
    return result;
}
variable_list FftC2CBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_fft_c2c_symint(grad, dim, normalization, !forward)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void FftC2CBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(forward);
    args.collect(normalization);
}
variable_list FftC2CBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(forward);
    saved.before(normalization);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(forward);
    saved.after(normalization);
    return result;
}
variable_list UnbindBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unbind_backward(grads, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnbindBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list UnbindBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list UnbindBackwardAutogradNestedTensor0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unbind_backward_nested(grads, at::native::get_nested_tensor_impl(self)->get_nested_sizes(), dim, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnbindBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(self_options);
}
variable_list UnbindBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(self_options);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(self_options);
    return result;
}
variable_list StackBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto tensors_ix = gen.range(tensors_size_);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  if (task_should_compute_output({ tensors_ix })) {
    auto grad_result = stack_tensors_backward(grad, dim, tensors_args_scalartypes);
    copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}
void StackBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(tensors_args_scalartypes);
}
variable_list StackBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(tensors_args_scalartypes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(tensors_args_scalartypes);
    return result;
}
variable_list ThnnFusedLstmCellBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_gates_ix = gen.range(1);
  auto hidden_gates_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  auto input_bias_ix = gen.range(1);
  auto hidden_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto cx = cx_.unpack();
  auto hidden_bias = hidden_bias_.unpack();
  auto hidden_gates = hidden_gates_.unpack();
  auto input_bias = input_bias_.unpack();
  auto input_gates = input_gates_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (task_should_compute_output({ input_gates_ix, hidden_gates_ix, cx_ix, input_bias_ix, hidden_bias_ix })) {
  
    auto grad_result = GradMode::is_enabled() ? _thnn_differentiable_lstm_cell_backward(grads[0], grads[1], input_gates, hidden_gates, input_bias, hidden_bias, cx, result1) : _thnn_fused_lstm_cell_backward(grads[0], grads[1], cx, result1, result2, input_bias.defined());
      if (task_should_compute_output({ input_gates_ix })) {
        copy_range(grad_inputs, input_gates_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ hidden_gates_ix })) {
        copy_range(grad_inputs, hidden_gates_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ input_bias_ix })) {
        copy_range(grad_inputs, input_bias_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ hidden_bias_ix })) {
        copy_range(grad_inputs, hidden_bias_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
void ThnnFusedLstmCellBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(cx_);
    args.collect(hidden_bias_);
    args.collect(hidden_gates_);
    args.collect(input_bias_);
    args.collect(input_gates_);
    args.collect(result1_);
    args.collect(result2_);
}
variable_list ThnnFusedLstmCellBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(cx_);
    saved.before(hidden_bias_);
    saved.before(hidden_gates_);
    saved.before(input_bias_);
    saved.before(input_gates_);
    saved.before(result1_);
    saved.before(result2_);
    variable_list result = apply(variable_list(grads));
    saved.after(cx_);
    saved.after(hidden_bias_);
    saved.after(hidden_gates_);
    saved.after(input_bias_);
    saved.after(input_gates_);
    saved.after(result1_);
    saved.after(result2_);
    return result;
}
variable_list ThnnFusedGruCellBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_gates_ix = gen.range(1);
  auto hidden_gates_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto input_bias_ix = gen.range(1);
  auto hidden_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto hidden_bias = hidden_bias_.unpack();
  auto hidden_gates = hidden_gates_.unpack();
  auto hx = hx_.unpack();
  auto input_bias = input_bias_.unpack();
  auto input_gates = input_gates_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (task_should_compute_output({ input_gates_ix, hidden_gates_ix, hx_ix, input_bias_ix, hidden_bias_ix })) {
  
    auto grad_result = grad.defined() ? (GradMode::is_enabled() ? _thnn_differentiable_gru_cell_backward(grad, input_gates, hidden_gates, hx, input_bias, hidden_bias) : _thnn_fused_gru_cell_backward(grad, result1, input_bias.defined())) : std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>();
      if (task_should_compute_output({ input_gates_ix })) {
        copy_range(grad_inputs, input_gates_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ hidden_gates_ix })) {
        copy_range(grad_inputs, hidden_gates_ix, std::get<1>(grad_result));
      }
      if (task_should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<2>(grad_result));
      }
      if (task_should_compute_output({ input_bias_ix })) {
        copy_range(grad_inputs, input_bias_ix, std::get<3>(grad_result));
      }
      if (task_should_compute_output({ hidden_bias_ix })) {
        copy_range(grad_inputs, hidden_bias_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
void ThnnFusedGruCellBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(hidden_bias_);
    args.collect(hidden_gates_);
    args.collect(hx_);
    args.collect(input_bias_);
    args.collect(input_gates_);
    args.collect(result1_);
}
variable_list ThnnFusedGruCellBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(hidden_bias_);
    saved.before(hidden_gates_);
    saved.before(hx_);
    saved.before(input_bias_);
    saved.before(input_gates_);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(hidden_bias_);
    saved.after(hidden_gates_);
    saved.after(hx_);
    saved.after(input_bias_);
    saved.after(input_gates_);
    saved.after(result1_);
    return result;
}
variable_list PackPaddedSequenceBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (_pack_padded_sequence_backward_symint(grad, input_sym_sizes, result1, batch_first)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
void PackPaddedSequenceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(batch_first);
    args.collect(input_sym_sizes);
    args.collect(result1_);
}
variable_list PackPaddedSequenceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(batch_first);
    saved.before(input_sym_sizes);
    saved.before(result1_);
    variable_list result = apply(variable_list(grads));
    saved.after(batch_first);
    saved.after(input_sym_sizes);
    saved.after(result1_);
    return result;
}
variable_list SegmentReduceBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto data_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto data = data_.unpack();
  auto lengths = lengths_.unpack();
  auto offsets = offsets_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ data_ix })) {
    auto grad_result = any_grad_defined ? (_segment_reduce_backward(grad, result, data, reduce, lengths, offsets, axis, initial)) : Tensor();
    copy_range(grad_inputs, data_ix, grad_result);
  }
  return grad_inputs;
}
void SegmentReduceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(axis);
    args.collect(data_);
    args.collect(initial);
    args.collect(lengths_);
    args.collect(offsets_);
    args.collect(reduce);
    args.collect(result_);
}
variable_list SegmentReduceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(axis);
    saved.before(data_);
    saved.before(initial);
    saved.before(lengths_);
    saved.before(offsets_);
    saved.before(reduce);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(axis);
    saved.after(data_);
    saved.after(initial);
    saved.after(lengths_);
    saved.after(offsets_);
    saved.after(reduce);
    saved.after(result_);
    return result;
}
variable_list PinMemoryBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PinMemoryBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list PinMemoryBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TestWarnInAutogradBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (warn_backwards(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestWarnInAutogradBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list TestWarnInAutogradBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TestAutogradMultipleDispatchBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand_symint(self_sym_sizes) + 1) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list TestAutogradMultipleDispatchBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list TestAutogradMultipleDispatchBackwardAutogradNestedTensor0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.mul(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchBackwardAutogradNestedTensor0::compiled_args(CompiledNodeArgs& args) {

}
variable_list TestAutogradMultipleDispatchBackwardAutogradNestedTensor0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TestAutogradMultipleDispatchBackwardAutogradCUDA0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand_symint(self_sym_sizes) * 2) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchBackwardAutogradCUDA0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list TestAutogradMultipleDispatchBackwardAutogradCUDA0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list TestAutogradMultipleDispatchBackwardAutogradNestedTensor1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.mul(grad).add(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchBackwardAutogradNestedTensor1::compiled_args(CompiledNodeArgs& args) {

}
variable_list TestAutogradMultipleDispatchBackwardAutogradNestedTensor1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TestAutogradMultipleDispatchViewBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchViewBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list TestAutogradMultipleDispatchViewBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list TestAutogradMultipleDispatchViewBackwardAutogradCUDA0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self) + 1) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchViewBackwardAutogradCUDA0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list TestAutogradMultipleDispatchViewBackwardAutogradCUDA0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ScatterReduceBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto index = index_.unpack();
  auto self = self_.unpack();
  auto src = src_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (task_should_compute_output({ self_ix, src_ix })) {
  
    auto grad_result = scatter_reduce_backward(grad, self, dim, index, src, reduce, include_self, result);
      if (task_should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (task_should_compute_output({ src_ix })) {
        copy_range(grad_inputs, src_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
void ScatterReduceBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(include_self);
    args.collect(index_);
    args.collect(reduce);
    args.collect(self_);
    args.collect(src_);
    args.collect(result_);
}
variable_list ScatterReduceBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(include_self);
    saved.before(index_);
    saved.before(reduce);
    saved.before(self_);
    saved.before(src_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(include_self);
    saved.after(index_);
    saved.after(reduce);
    saved.after(self_);
    saved.after(src_);
    saved.after(result_);
    return result;
}
variable_list ReshapeCopyBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReshapeCopyBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ReshapeCopyBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list ForeachDivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(div_tensor_other_backward(grads[i], self[i], other[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(div_tensor_self_backward(grads[i], other[i], self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachDivBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachDivBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachPowBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!exponent_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto exponent_ix = gen.range(exponent_size_);
  variable_list grad_inputs(gen.size());
  auto exponent = unpack_list(exponent_, nullptr);
  auto self = unpack_list(self_, nullptr);
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ exponent_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(pow_backward_exponent(grads[i], self[i], exponent[i], result[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(pow_backward_self(grads[i], self[i], exponent[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachPowBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent_);
    args.collect(self_);
    args.collect(result_);
}
variable_list ForeachPowBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent_);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent_);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list ForeachPowBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(pow_backward(grads[i], self[i], exponent[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachPowBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent);
    args.collect(self_);
}
variable_list ForeachPowBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent);
    saved.after(self_);
    return result;
}
variable_list ForeachPowBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!exponent_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto exponent_ix = gen.range(exponent_size_);
  variable_list grad_inputs(gen.size());
  auto exponent = unpack_list(exponent_, nullptr);
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ exponent_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(pow_backward_exponent(grads[i], self, exponent[i], result[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachPowBackward2::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent_);
    args.collect(self);
    args.collect(result_);
}
variable_list ForeachPowBackward2::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent_);
    saved.before(self);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent_);
    saved.after(self);
    saved.after(result_);
    return result;
}
variable_list ForeachMinimumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == scalar, grads[i] / 2, grads[i]).masked_fill_(self[i] > scalar, 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMinimumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachMinimumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachMinimumBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == scalars[i], grads[i] / 2, grads[i]).masked_fill_(self[i] > scalars[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMinimumBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachMinimumBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachMaximumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == scalar, grads[i] / 2, grads[i]).masked_fill_(self[i] < scalar, 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMaximumBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachMaximumBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachMaximumBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == scalars[i], grads[i] / 2, grads[i]).masked_fill_(self[i] < scalars[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMaximumBackward1::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachMaximumBackward1::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachNormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(norm_backward(grads[i], self[i], ord, result[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachNormBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(ord);
    args.collect(self_);
    args.collect(result_);
}
variable_list ForeachNormBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(ord);
    saved.before(self_);
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(ord);
    saved.after(self_);
    saved.after(result_);
    return result;
}
variable_list AliasBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AliasBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list AliasBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list AsStridedBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (as_strided_backward(grad, self_geometry, size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void AsStridedBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_geometry);
    args.collect(size);
    args.collect(storage_offset);
    args.collect(stride);
}
variable_list AsStridedBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_geometry);
    saved.before(size);
    saved.before(storage_offset);
    saved.before(stride);
    variable_list result = apply(variable_list(grads));
    saved.after(self_geometry);
    saved.after(size);
    saved.after(storage_offset);
    saved.after(stride);
    return result;
}
variable_list ConjBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ConjBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list ConjBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list NegViewBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.neg()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NegViewBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list NegViewBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list DiagonalBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (diagonal_backward_symint(grad, self_sym_sizes, offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void DiagonalBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim1);
    args.collect(dim2);
    args.collect(offset);
    args.collect(self_sym_sizes);
}
variable_list DiagonalBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim1);
    saved.before(dim2);
    saved.before(offset);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim1);
    saved.after(dim2);
    saved.after(offset);
    saved.after(self_sym_sizes);
    return result;
}
variable_list ExpandBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::sum_to(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ExpandBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ExpandBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list PermuteBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (permute_backwards(grad, dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void PermuteBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dims);
}
variable_list PermuteBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dims);
    variable_list result = apply(variable_list(grads));
    saved.after(dims);
    return result;
}
variable_list ReshapeAliasBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ReshapeAliasBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ReshapeAliasBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list SelectBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (select_backward_symint(grad, self_sym_sizes, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SelectBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
    args.collect(self_sym_sizes);
}
variable_list SelectBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SelectBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_nested_select_backward_symint(grad, self, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SelectBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(index);
    args.collect(self_);
}
variable_list SelectBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(index);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(index);
    saved.after(self_);
    return result;
}
variable_list SliceBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slice_backward_wrapper(grad, self_sym_sizes, dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SliceBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(end);
    args.collect(self_sym_sizes);
    args.collect(start);
    args.collect(step);
}
variable_list SliceBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(end);
    saved.before(self_sym_sizes);
    saved.before(start);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(end);
    saved.after(self_sym_sizes);
    saved.after(start);
    saved.after(step);
    return result;
}
variable_list SplitBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_backward(grads, split_size, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_size);
}
variable_list SplitBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_size);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_size);
    return result;
}
variable_list SplitWithSizesBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_with_sizes_backward(grads, split_sizes, dim, self_sym_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitWithSizesBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_options);
    args.collect(self_sym_sizes);
    args.collect(split_sizes);
}
variable_list SplitWithSizesBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_options);
    saved.before(self_sym_sizes);
    saved.before(split_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_options);
    saved.after(self_sym_sizes);
    saved.after(split_sizes);
    return result;
}
variable_list SplitWithSizesBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_nested_split_with_sizes_backward(grads, split_sizes, dim, at::native::get_nested_tensor_impl(self)->get_nested_sizes(), self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SplitWithSizesBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(self_options);
    args.collect(split_sizes);
}
variable_list SplitWithSizesBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(self_options);
    saved.before(split_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(self_options);
    saved.after(split_sizes);
    return result;
}
variable_list SqueezeBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackward1_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward1_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward1_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.unsqueeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list SqueezeBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list SqueezeBackward2_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackward2_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_sym_sizes);
}
variable_list SqueezeBackward2_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_sym_sizes);
    return result;
}
variable_list SqueezeBackwardAutogradNestedTensor1_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_multiple(grad, dim, self_dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void SqueezeBackwardAutogradNestedTensor1_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_dim);
}
variable_list SqueezeBackwardAutogradNestedTensor1_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_dim);
    return result;
}
variable_list TBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.t()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list TBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list TransposeBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.transpose(dim0, dim1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TransposeBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim0);
    args.collect(dim1);
}
variable_list TransposeBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim0);
    saved.before(dim1);
    variable_list result = apply(variable_list(grads));
    saved.after(dim0);
    saved.after(dim1);
    return result;
}
variable_list UnfoldBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unfold_backward_symint(grad, self_sym_sizes, dimension, size, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnfoldBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dimension);
    args.collect(self_sym_sizes);
    args.collect(size);
    args.collect(step);
}
variable_list UnfoldBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dimension);
    saved.before(self_sym_sizes);
    saved.before(size);
    saved.before(step);
    variable_list result = apply(variable_list(grads));
    saved.after(dimension);
    saved.after(self_sym_sizes);
    saved.after(size);
    saved.after(step);
    return result;
}
variable_list LiftFreshBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void LiftFreshBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list LiftFreshBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UnsqueezeBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.squeeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnsqueezeBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list UnsqueezeBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list ViewBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_symint(self_sym_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_sym_sizes);
}
variable_list ViewBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_sym_sizes);
    return result;
}
variable_list ViewBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ViewBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ViewAsRealBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_complex(grad.contiguous())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewAsRealBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list ViewAsRealBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ViewAsComplexBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_real(grad.contiguous().resolve_conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ViewAsComplexBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list ViewAsComplexBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ValuesBackward0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_sparse_coo_tensor_unsafe_symint(self.indices(), grad, self_sym_sizes)._coalesced_(true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ValuesBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(self_sym_sizes);
}
variable_list ValuesBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(self_sym_sizes);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(self_sym_sizes);
    return result;
}
variable_list ValuesBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_nested_view_from_buffer(grad.contiguous(), self._nested_tensor_size(), self._nested_tensor_strides(), self._nested_tensor_storage_offsets())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ValuesBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ValuesBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list NestedViewFromBufferBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.values()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void NestedViewFromBufferBackward0_copy::compiled_args(CompiledNodeArgs& args) {

}
variable_list NestedViewFromBufferBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list UnbindBackward0_copy::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unbind_backward(grads, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnbindBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
}
variable_list UnbindBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    return result;
}
variable_list UnbindBackwardAutogradNestedTensor0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unbind_backward_nested(grads, at::native::get_nested_tensor_impl(self)->get_nested_sizes(), dim, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void UnbindBackwardAutogradNestedTensor0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(dim);
    args.collect(self_);
    args.collect(self_options);
}
variable_list UnbindBackwardAutogradNestedTensor0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(dim);
    saved.before(self_);
    saved.before(self_options);
    variable_list result = apply(variable_list(grads));
    saved.after(dim);
    saved.after(self_);
    saved.after(self_options);
    return result;
}
variable_list TestAutogradMultipleDispatchViewBackward0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchViewBackward0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list TestAutogradMultipleDispatchViewBackward0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape_as(self) + 1) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAbsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * self[i].sgn());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAbsBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAbsBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAcosBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * -((-self[i] * self[i] + 1).rsqrt()).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAcosBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAcosBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAddBackward1Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddBackward1Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAddBackward1Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAddBackward0List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(other[i].scalar_type(), maybe_multiply(grads[i], alpha.conj())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddBackward0List::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachAddBackward0List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachAddBackward1ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddBackward1ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAddBackward1ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAddBackward0Tensor::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto other = other_.unpack();
  auto self = unpack_list(self_, nullptr);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(any_grad_defined ? (handle_r_to_c(other.scalar_type(), maybe_multiply(grads[i], alpha.conj()))) : Tensor());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddBackward0Tensor::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachAddBackward0Tensor::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachAddcdivBackward0Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor1_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor2_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensor1_ix = gen.range(tensor1_size_);
  auto tensor2_ix = gen.range(tensor2_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto tensor1 = unpack_list(tensor1_, nullptr);
  auto tensor2 = unpack_list(tensor2_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor1[i].scalar_type(), grads[i] * (value / tensor2[i]).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor2[i][i].scalar_type(), -grads[i] * (value * tensor1[i] / (tensor2[i] * tensor2[i])).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddcdivBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(tensor1_);
    args.collect(tensor2_);
    args.collect(value);
}
variable_list ForeachAddcdivBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(tensor1_);
    saved.before(tensor2_);
    saved.before(value);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(tensor1_);
    saved.after(tensor2_);
    saved.after(value);
    return result;
}
variable_list ForeachAddcdivBackward0ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor1_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor2_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensor1_ix = gen.range(tensor1_size_);
  auto tensor2_ix = gen.range(tensor2_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto tensor1 = unpack_list(tensor1_, nullptr);
  auto tensor2 = unpack_list(tensor2_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor1[i].scalar_type(), grads[i] * (scalars[i] / tensor2[i]).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor2[i][i].scalar_type(), -grads[i] * (scalars[i] * tensor1[i] / (tensor2[i] * tensor2[i])).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddcdivBackward0ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
    args.collect(tensor1_);
    args.collect(tensor2_);
}
variable_list ForeachAddcdivBackward0ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    saved.before(tensor1_);
    saved.before(tensor2_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    saved.after(tensor1_);
    saved.after(tensor2_);
    return result;
}
variable_list ForeachAddcmulBackward0Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor1_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor2_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensor1_ix = gen.range(tensor1_size_);
  auto tensor2_ix = gen.range(tensor2_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto tensor1 = unpack_list(tensor1_, nullptr);
  auto tensor2 = unpack_list(tensor2_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor1[i].scalar_type(), grads[i] * (tensor2[i] * value).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor2[i].scalar_type(), grads[i] * (tensor1[i] * value).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddcmulBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(tensor1_);
    args.collect(tensor2_);
    args.collect(value);
}
variable_list ForeachAddcmulBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(tensor1_);
    saved.before(tensor2_);
    saved.before(value);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(tensor1_);
    saved.after(tensor2_);
    saved.after(value);
    return result;
}
variable_list ForeachAddcmulBackward0ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor1_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensor2_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensor1_ix = gen.range(tensor1_size_);
  auto tensor2_ix = gen.range(tensor2_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto tensor1 = unpack_list(tensor1_, nullptr);
  auto tensor2 = unpack_list(tensor2_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ tensor1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor1[i].scalar_type(), grads[i] * (tensor2[i] * scalars[i]).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (task_should_compute_output({ tensor2_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(tensor2[i].scalar_type(), grads[i] * (tensor1[i] * scalars[i]).conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAddcmulBackward0ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
    args.collect(tensor1_);
    args.collect(tensor2_);
}
variable_list ForeachAddcmulBackward0ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    saved.before(tensor1_);
    saved.before(tensor2_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    saved.after(tensor1_);
    saved.after(tensor2_);
    return result;
}
variable_list ForeachAsinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * (-self[i] * self[i] + 1).rsqrt().conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAsinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAsinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachAtanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] / (self[i] * self[i] + 1).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachAtanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachAtanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachCeilBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(zeros_like(grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachCeilBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachCeilBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachClampMaxBackward0Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] <= scalar, grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMaxBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachClampMaxBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachClampMaxBackward1List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] > other[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] <= other[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMaxBackward1List::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachClampMaxBackward1List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachClampMaxBackward0ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] <= scalars[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMaxBackward0ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachClampMaxBackward0ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachClampMinBackward0Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] >= scalar, grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMinBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachClampMinBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachClampMinBackward1List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] < other[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] >= other[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMinBackward1List::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachClampMinBackward1List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachClampMinBackward0ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(where(self[i] >= scalars[i], grads[i], at::scalar_tensor(0., grads[i].options())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachClampMinBackward0ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachClampMinBackward0ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachCosBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * -self[i].sin().conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachCosBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachCosBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachCoshBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * self[i].sinh().conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachCoshBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachCoshBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachDivBackward1Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(div_tensor_self_backward(grads[i], scalar, self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachDivBackward1Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachDivBackward1Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachDivBackward1ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(div_tensor_self_backward(grads[i], scalars[i], self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachDivBackward1ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachDivBackward1ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachDivBackward0Tensor::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto other = other_.unpack();
  auto self = unpack_list(self_, nullptr);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(any_grad_defined ? (div_tensor_other_backward(grads[i], self[i], other)) : Tensor());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(div_tensor_self_backward(grads[i], other, self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachDivBackward0Tensor::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachDivBackward0Tensor::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachErfBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(2.0 / sqrt(M_PI) * exp(-(self[i].pow(2))) * grads[i]);
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachErfBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachErfBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachErfcBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(-2.0 / sqrt(M_PI) * exp(-(self[i].pow(2))) * grads[i]);
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachErfcBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachErfcBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachExpBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * result[i].conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachExpBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachExpBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachExpm1Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * (result[i].conj() + 1));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachExpm1Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachExpm1Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachFloorBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(zeros_like(grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachFloorBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachFloorBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachFracBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i]);
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachFracBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachFracBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachLerpBackward1List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!tensors1_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!weights_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensors1_ix = gen.range(tensors1_size_);
  auto weights_ix = gen.range(weights_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  auto tensors1 = unpack_list(tensors1_, nullptr);
  auto weights = unpack_list(weights_, nullptr);
  if (task_should_compute_output({ tensors1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * weights[i].conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensors1_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * (1 - weights[i]).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (task_should_compute_output({ weights_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * (tensors1[i] - self[i]).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, weights_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLerpBackward1List::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
    args.collect(tensors1_);
    args.collect(weights_);
}
variable_list ForeachLerpBackward1List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    saved.before(tensors1_);
    saved.before(weights_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    saved.after(tensors1_);
    saved.after(weights_);
    return result;
}
variable_list ForeachLerpBackward0Scalar::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto tensors1_ix = gen.range(tensors1_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ tensors1_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * weight.conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, tensors1_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(weight.isComplex() ? grads[i] * (1 - weight.conj().toComplexDouble()) : grads[i] * (1 - weight.toDouble()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLerpBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(weight);
}
variable_list ForeachLerpBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(weight);
    variable_list result = apply(variable_list(grads));
    saved.after(weight);
    return result;
}
variable_list ForeachLgammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * digamma(self[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLgammaBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachLgammaBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachLogBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i].div(self[i].conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLogBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachLogBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachLog10Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] / (self[i].conj() * 2.3025850929940456));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLog10Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachLog10Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachLog1PBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(log1p_backward(grads[i], self[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLog1PBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachLog1PBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachLog2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] / (self[i].conj() * 0.6931471805599453));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachLog2Backward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachLog2Backward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachMaximumBackward0List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == other[i], grads[i] / 2, grads[i]).masked_fill_(self[i] > other[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == other[i], grads[i] / 2, grads[i]).masked_fill_(self[i] < other[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMaximumBackward0List::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachMaximumBackward0List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachMinimumBackward0List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == other[i], grads[i] / 2, grads[i]).masked_fill_(self[i] < other[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(at::where(self[i] == other[i], grads[i] / 2, grads[i]).masked_fill_(self[i] > other[i], 0));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMinimumBackward0List::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachMinimumBackward0List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachMulBackward1Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(mul_tensor_backward(grads[i], scalar, self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMulBackward1Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalar);
    args.collect(self_);
}
variable_list ForeachMulBackward1Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalar);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalar);
    saved.after(self_);
    return result;
}
variable_list ForeachMulBackward0List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(mul_tensor_backward(grads[i], self[i], other[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(mul_tensor_backward(grads[i], other[i], self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMulBackward0List::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachMulBackward0List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachMulBackward1ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(mul_tensor_backward(grads[i], scalars[i], self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMulBackward1ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(scalars);
    args.collect(self_);
}
variable_list ForeachMulBackward1ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(scalars);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(scalars);
    saved.after(self_);
    return result;
}
variable_list ForeachMulBackward0Tensor::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto other = other_.unpack();
  auto self = unpack_list(self_, nullptr);
  bool any_grad_defined = any_variable_defined(grads);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(any_grad_defined ? (mul_tensor_backward(grads[i], self[i], other.scalar_type())) : Tensor());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(mul_tensor_backward(grads[i], other, self[i].scalar_type()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachMulBackward0Tensor::compiled_args(CompiledNodeArgs& args) {
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachMulBackward0Tensor::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachNegBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i].neg());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachNegBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachNegBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachPowBackward0Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(pow_backward(grads[i], self[i], exponent));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachPowBackward0Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(exponent);
    args.collect(self_);
}
variable_list ForeachPowBackward0Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(exponent);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(exponent);
    saved.after(self_);
    return result;
}
variable_list ForeachReciprocalBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(-grads[i] * (result[i] * result[i]).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachReciprocalBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachReciprocalBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachRoundBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(zeros_like(grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachRoundBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachRoundBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachSigmoidBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(sigmoid_backward(grads[i], result[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSigmoidBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachSigmoidBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachSignBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(zeros_like(grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSignBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachSignBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}
variable_list ForeachSinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * self[i].cos().conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSinBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachSinBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachSinhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * self[i].cosh().conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSinhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachSinhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachSqrtBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] / (2 * result[i].conj()));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSqrtBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachSqrtBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachSubBackward1Scalar::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSubBackward1Scalar::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachSubBackward1Scalar::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachSubBackward0List::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!other_released_, ERR_BACKWARD_TWICE);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  auto other_ix = gen.range(other_size_);
  variable_list grad_inputs(gen.size());
  auto other = unpack_list(other_, nullptr);
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ other_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(other[i].scalar_type(), maybe_multiply(-grads[i], alpha.conj())));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSubBackward0List::compiled_args(CompiledNodeArgs& args) {
    args.collect(alpha);
    args.collect(other_);
    args.collect(self_);
}
variable_list ForeachSubBackward0List::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(alpha);
    saved.before(other_);
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(alpha);
    saved.after(other_);
    saved.after(self_);
    return result;
}
variable_list ForeachSubBackward1ScalarList::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!self_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto self = unpack_list(self_, nullptr);
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(handle_r_to_c(self[i].scalar_type(), grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachSubBackward1ScalarList::compiled_args(CompiledNodeArgs& args) {
    args.collect(self_);
}
variable_list ForeachSubBackward1ScalarList::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(self_);
    variable_list result = apply(variable_list(grads));
    saved.after(self_);
    return result;
}
variable_list ForeachTanBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(grads[i] * (1 + result[i].pow(2)).conj());
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachTanBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachTanBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachTanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!result_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  auto result = unpack_list(result_, shared_from_this());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(tanh_backward(grads[i], result[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachTanhBackward0::compiled_args(CompiledNodeArgs& args) {
    args.collect(result_);
}
variable_list ForeachTanhBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
    saved.before(result_);
    variable_list result = apply(variable_list(grads));
    saved.after(result_);
    return result;
}
variable_list ForeachTruncBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(self_size_);
  variable_list grad_inputs(gen.size());
  if (task_should_compute_output({ self_ix })) {
    std::vector<Tensor> grad_result;
    grad_result.reserve(grads.size());
    for (const auto & i : c10::irange(grads.size())) {
      if (grads[i].defined()) {
        grad_result.emplace_back(zeros_like(grads[i]));
      } else {
        grad_result.emplace_back(Tensor());
      }
    }
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
void ForeachTruncBackward0::compiled_args(CompiledNodeArgs& args) {

}
variable_list ForeachTruncBackward0::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {

    variable_list result = apply(variable_list(grads));

    return result;
}

} // namespace torch::autograd::generated
