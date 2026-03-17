// torch/csrc/autograd/jin_helper.cpp
//
// JIN helper (JIN1 binary payload) + overwrite helpers
// + ReLU mask (lossless) support
// + MaxPool2d 2-bit index packing/unpacking support
//
// Notes
// -----
// 1) This file READS JIN1 payload into st.kv_cpu on Node B.
// 2) Node A should WRITE JIN1 payload including keys such as:
//      - conv2d:<i>:input
//      - conv2d:<i>:weight
//      - addmm:<i>:mat1
//      - addmm:<i>:mat2
//      - batchnorm:<i>:...
//      - maxpool2d:<i>:indices_2bit
//      - relu_mask:<i>
// 3) ReLU path:
//      - Node A: jin_capture_relu_mask(relu_out)
//      - Node B: jin_relu_backward_from_mask(grad)
//
// Build note
// ----------
// Make sure jin_helper.h declares at least:
//
//   extern "C" {
//     void jin_init_if_needed();
//     void jin_reset_counters();
//     bool jin_has_key(const char* key);
//     bool jin_overwrite_by_key(at::Tensor& t, const char* key);
//
//     void jin_overwrite_conv_input(at::Tensor& t);
//     void jin_overwrite_conv_weight(at::Tensor& t);
//
//     void jin_overwrite_relu_saved(at::Tensor& t);
//     void jin_overwrite_relu(at::Tensor& t);
//
//     void jin_overwrite_addmm_mat1(at::Tensor& t);
//     void jin_overwrite_addmm_mat2(at::Tensor& t);
//     void jin_advance_addmm();
//
//     void jin_overwrite_maxpool2d_input(at::Tensor& t);
//     void jin_overwrite_maxpool2d_indices(at::Tensor& t);
//
//     void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step);
//
//     void jin_overwrite_batchnorm_input(at::Tensor& t);
//     void jin_overwrite_batchnorm_running_mean(at::Tensor& t);
//     void jin_overwrite_batchnorm_running_var(at::Tensor& t);
//     void jin_overwrite_batchnorm_weight(at::Tensor& t);
//     void jin_overwrite_batchnorm_result1(at::Tensor& t);
//     void jin_overwrite_batchnorm_result2(at::Tensor& t);
//
//     void jin_capture_relu_mask(const at::Tensor& relu_out);
//     at::Tensor jin_relu_backward_from_mask(const at::Tensor& grad);
//
//     bool jin_is_role_B();
//   }
//
//   std::vector<uint8_t> jin_pack_maxpool2x2_flat_indices_to_2bit(
//       const at::Tensor& flat_indices,
//       int64_t input_h,
//       int64_t input_w);
//
//   at::Tensor jin_unpack_maxpool2x2_2bit_to_flat_indices(
//       const std::vector<uint8_t>& packed,
//       int64_t N,
//       int64_t C,
//       int64_t Hout,
//       int64_t Wout,
//       int64_t input_h,
//       int64_t input_w,
//       c10::Device device);
//
//   at::Tensor jin_make_maxpool2d_indices_2bit_tensor(
//       const at::Tensor& flat_indices);
//

#include "torch/csrc/autograd/jin_helper.h"

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_mu;

struct JINState {
  bool loaded = false;

  std::string role;          // "A" or "B"
  std::string payload_path;  // e.g. /tmp/jin_payload_step0.bin
  int64_t step = -1;

  // parsed payload tensors are kept on CPU
  std::unordered_map<std::string, at::Tensor> kv_cpu;

  // optional memory payload (JIN1 raw bytes)
  std::vector<uint8_t> mem_buf;
  int64_t mem_step = -1;
  bool mem_has = false;

  // op counters
  int64_t conv2d_i = 0;
  int64_t relu_i = 0;
  int64_t addmm_i = 0;
  int64_t maxpool2d_i = 0;
  int64_t batchnorm_i = 0;
};

JINState& S() {
  static JINState st;
  return st;
}

static inline const char* env_cstr(const char* k) {
  const char* v = std::getenv(k);
  return v ? v : "";
}

static inline int64_t env_i64(const char* k, int64_t defv) {
  const char* v = std::getenv(k);
  if (!v || !*v) return defv;
  char* end = nullptr;
  long long x = std::strtoll(v, &end, 10);
  if (!end || end == v) return defv;
  return static_cast<int64_t>(x);
}

static std::string make_key(const char* op, int64_t idx, const char* field) {
  std::string k;
  k.reserve(64);
  k.append(op);
  k.push_back(':');
  k.append(std::to_string(idx));
  k.push_back(':');
  k.append(field);
  return k;
}

static std::string relu_mask_key(int64_t i) {
  return "relu_mask:" + std::to_string(i);
}

static std::string sizes_str(const at::Tensor& t) {
  std::ostringstream oss;
  oss << "[";
  for (int64_t i = 0; i < t.dim(); ++i) {
    oss << t.size(i);
    if (i + 1 < t.dim()) oss << ",";
  }
  oss << "]";
  return oss.str();
}

static at::ScalarType dtype_from_u8(uint8_t code) {
  switch (code) {
    case 0: return at::kFloat;
    case 1: return at::kDouble;
    case 2: return at::kLong;
    case 3: return at::kInt;
    case 4: return at::kShort;
    case 5: return at::kChar;
    case 6: return at::kByte;
    case 7: return at::kBool;
    default:
      TORCH_CHECK(false, "[JIN] unknown dtype code=", static_cast<int>(code));
  }
}

// ============================================================
// JIN1 reader
// ============================================================

struct Reader {
  const uint8_t* p;
  const uint8_t* end;

  explicit Reader(const std::vector<uint8_t>& buf)
      : p(buf.data()), end(buf.data() + buf.size()) {}

  void need(size_t n) {
    TORCH_CHECK(static_cast<size_t>(end - p) >= n, "[JIN] payload truncated");
  }

  uint8_t u8() {
    need(1);
    return *p++;
  }

  uint32_t u32() {
    need(4);
    uint32_t x;
    std::memcpy(&x, p, 4);
    p += 4;
    return x;
  }

  uint64_t u64() {
    need(8);
    uint64_t x;
    std::memcpy(&x, p, 8);
    p += 8;
    return x;
  }

  int64_t i64() {
    need(8);
    int64_t x;
    std::memcpy(&x, p, 8);
    p += 8;
    return x;
  }

  std::string bytes_str(uint32_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
    p += n;
    return s;
  }

  const uint8_t* bytes_ptr(uint64_t n) {
    need(static_cast<size_t>(n));
    const uint8_t* r = p;
    p += static_cast<size_t>(n);
    return r;
  }
};

static void parse_jin1_into_state_locked(JINState& st, const std::vector<uint8_t>& buf) {
  st.kv_cpu.clear();

  Reader r(buf);
  r.need(4);
  TORCH_CHECK(
      r.u8() == 'J' && r.u8() == 'I' && r.u8() == 'N' && r.u8() == '1',
      "[JIN] bad magic (expected JIN1)");

  uint32_t n = r.u32();
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t key_len = r.u32();
    std::string key = r.bytes_str(key_len);

    uint8_t dtype_code = r.u8();
    at::ScalarType dtype = dtype_from_u8(dtype_code);

    uint8_t ndim = r.u8();
    std::vector<int64_t> sizes;
    sizes.reserve(ndim);
    for (uint8_t d = 0; d < ndim; ++d) {
      sizes.push_back(r.i64());
    }

    uint64_t nbytes = r.u64();
    const uint8_t* raw = r.bytes_ptr(nbytes);

    auto t = at::empty(
        sizes,
        at::TensorOptions().dtype(dtype).device(at::kCPU));

    TORCH_CHECK(
        static_cast<uint64_t>(t.nbytes()) == nbytes,
        "[JIN] nbytes mismatch key=", key,
        " expected=", static_cast<unsigned long long>(t.nbytes()),
        " got=", static_cast<unsigned long long>(nbytes));

    if (nbytes > 0) {
      std::memcpy(t.data_ptr(), raw, static_cast<size_t>(nbytes));
    }

    st.kv_cpu[key] = t;
  }

  st.loaded = true;
}

static void load_payload_locked(JINState& st) {
  st.kv_cpu.clear();

  std::ifstream f(st.payload_path, std::ios::binary);
  TORCH_CHECK(f.good(), "[JIN] cannot open payload: ", st.payload_path);

  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  TORCH_CHECK(sz > 0, "[JIN] payload empty: ", st.payload_path);
  f.seekg(0, std::ios::beg);

  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
  TORCH_CHECK(f.good() || f.eof(), "[JIN] failed reading payload: ", st.payload_path);

  parse_jin1_into_state_locked(st, buf);

  std::fprintf(
      stderr,
      "[JIN] LOADED(JIN1) payload=%s keys=%zu step=%lld\n",
      st.payload_path.c_str(),
      st.kv_cpu.size(),
      static_cast<long long>(st.step));
  std::fflush(stderr);
}

static void load_payload_from_mem_locked(JINState& st) {
  TORCH_CHECK(st.mem_has, "[JIN] mem payload not set");
  TORCH_CHECK(!st.mem_buf.empty(), "[JIN] mem payload empty");

  parse_jin1_into_state_locked(st, st.mem_buf);

  std::fprintf(
      stderr,
      "[JIN] LOADED(JIN1) payload=MEM keys=%zu step=%lld\n",
      st.kv_cpu.size(),
      static_cast<long long>(st.step));
  std::fflush(stderr);
}

static void reset_counters_locked(JINState& st) {
  st.conv2d_i = 0;
  st.relu_i = 0;
  st.addmm_i = 0;
  st.maxpool2d_i = 0;
  st.batchnorm_i = 0;
}

static void ensure_loaded_locked(JINState& st) {
  st.role = env_cstr("JIN_ROLE");
  st.payload_path = env_cstr("JIN_PAYLOAD_PATH");
  if (st.payload_path.empty()) st.payload_path = "/tmp/jin_payload.bin";

  // Only Node B needs to load payload.
  if (st.role != "B") {
    return;
  }

  int64_t cur_step = env_i64("JIN_STEP", -1);
  bool need_reload = (!st.loaded) || (cur_step >= 0 && cur_step != st.step);
  if (!need_reload) {
    return;
  }

  st.step = cur_step;

  if (st.mem_has && st.mem_step == cur_step) {
    load_payload_from_mem_locked(st);
  } else {
    load_payload_locked(st);
  }

  reset_counters_locked(st);
}

// ============================================================
// Generic overwrite helper
// ============================================================

static bool overwrite_tensor_locked(JINState& st, at::Tensor& target, const std::string& key) {
  auto it = st.kv_cpu.find(key);
  if (it == st.kv_cpu.end()) {
    std::fprintf(stderr, "[JIN] MISS key=%s\n", key.c_str());
    std::fflush(stderr);
    return false;
  }

  at::Tensor src = it->second;

  TORCH_CHECK(target.defined(), "[JIN] target undefined for key=", key);
  TORCH_CHECK(src.defined(), "[JIN] src undefined for key=", key);
  TORCH_CHECK(
      target.numel() == src.numel(),
      "[JIN] numel mismatch key=", key,
      " target_numel=", static_cast<long long>(target.numel()),
      " target_sizes=", sizes_str(target),
      " src_numel=", static_cast<long long>(src.numel()),
      " src_sizes=", sizes_str(src));
  TORCH_CHECK(
      target.scalar_type() == src.scalar_type(),
      "[JIN] dtype mismatch key=", key,
      " target_dtype=", c10::toString(target.scalar_type()),
      " src_dtype=", c10::toString(src.scalar_type()));

  auto stype = target.scalar_type();

  auto log_stats = [&](const char* tag, const at::Tensor& t) {
    if (at::isFloatingType(stype) || at::isComplexType(stype)) {
      double m = t.mean().item<double>();
      std::fprintf(stderr, "%s_mean=%f", tag, m);
    } else {
      auto mx = t.to(at::kLong).abs().max().item<int64_t>();
      std::fprintf(stderr, "%s_absmax=%lld", tag, static_cast<long long>(mx));
    }
  };

  std::fprintf(stderr, "[JIN] OK key=%s ", key.c_str());
  log_stats("before", target);

  target.copy_(src.to(target.device()));

  std::fprintf(stderr, " ");
  log_stats("after", target);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
  return true;
}

// ============================================================
// Small utilities
// ============================================================

static inline void bit_set(uint8_t* buf, int64_t idx) {
  buf[idx >> 3] |= static_cast<uint8_t>(1u << (idx & 7));
}

static inline bool bit_get(const uint8_t* buf, int64_t idx) {
  return ((buf[idx >> 3] >> (idx & 7)) & 1u) != 0;
}

static at::Tensor bytes_vec_to_u8_tensor_cpu(const std::vector<uint8_t>& v) {
  auto t = at::empty(
      {static_cast<int64_t>(v.size())},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  if (!v.empty()) {
    std::memcpy(t.data_ptr(), v.data(), v.size());
  }
  return t;
}

static std::vector<uint8_t> u8_tensor_cpu_to_bytes_vec(const at::Tensor& t) {
  TORCH_CHECK(t.defined(), "u8 tensor is undefined");
  TORCH_CHECK(t.scalar_type() == at::kByte, "u8 tensor must be uint8");

  auto cpu = t.contiguous().to(at::kCPU);
  const uint8_t* p = cpu.data_ptr<uint8_t>();
  return std::vector<uint8_t>(p, p + cpu.numel());
}

// ============================================================
// ReLU mask helpers
// ============================================================

static at::Tensor pack_relu_mask_bits_from_out(const at::Tensor& relu_out) {
  TORCH_CHECK(relu_out.defined(), "jin_capture_relu_mask: relu_out is undefined");

  at::Tensor out_cpu = relu_out.detach();
  if (!out_cpu.is_cpu()) out_cpu = out_cpu.to(at::kCPU);
  out_cpu = out_cpu.contiguous();

  const int64_t n = out_cpu.numel();
  const int64_t nbytes = (n + 7) / 8;

  at::Tensor mask_bytes = at::zeros(
      {nbytes},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));

  uint8_t* mb = mask_bytes.data_ptr<uint8_t>();
  const auto dt = out_cpu.scalar_type();

  if (dt == at::kFloat) {
    const float* p = out_cpu.data_ptr<float>();
    for (int64_t i = 0; i < n; ++i) if (p[i] > 0.0f) bit_set(mb, i);
  } else if (dt == at::kDouble) {
    const double* p = out_cpu.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) if (p[i] > 0.0) bit_set(mb, i);
  } else if (dt == at::kHalf) {
    const at::Half* p = out_cpu.data_ptr<at::Half>();
    for (int64_t i = 0; i < n; ++i) if (static_cast<float>(p[i]) > 0.0f) bit_set(mb, i);
  } else if (dt == at::kBFloat16) {
    const at::BFloat16* p = out_cpu.data_ptr<at::BFloat16>();
    for (int64_t i = 0; i < n; ++i) if (static_cast<float>(p[i]) > 0.0f) bit_set(mb, i);
  } else {
    TORCH_CHECK(false, "jin_capture_relu_mask: unsupported dtype: ", dt);
  }

  return mask_bytes;
}

// ============================================================
// MaxPool2d 2-bit helpers
// Assumption: kernel=2, stride=2, padding=0
// ============================================================

} // namespace

std::vector<uint8_t> jin_pack_maxpool2x2_flat_indices_to_2bit(
    const at::Tensor& flat_indices,
    int64_t input_h,
    int64_t input_w) {
  TORCH_CHECK(flat_indices.scalar_type() == at::kLong, "flat_indices must be int64");
  TORCH_CHECK(flat_indices.dim() == 4, "flat_indices must be [N,C,Hout,Wout]");
  TORCH_CHECK(input_h > 0 && input_w > 0, "invalid input_h/input_w");

  auto idx = flat_indices.contiguous().to(at::kCPU);

  const int64_t N = idx.size(0);
  const int64_t C = idx.size(1);
  const int64_t Hout = idx.size(2);
  const int64_t Wout = idx.size(3);

  const int64_t numel = idx.numel();
  const int64_t packed_size = (numel + 3) / 4;

  std::vector<uint8_t> packed(static_cast<size_t>(packed_size), 0);
  const int64_t* p = idx.data_ptr<int64_t>();

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < Hout; ++oh) {
        for (int64_t ow = 0; ow < Wout; ++ow) {
          const int64_t linear_idx = ((n * C + c) * Hout + oh) * Wout + ow;
          const int64_t flat = p[linear_idx];

          TORCH_CHECK(
              flat >= 0 && flat < input_h * input_w,
              "flat index out of range: flat=", flat,
              " input_h=", input_h,
              " input_w=", input_w);

          const int64_t ih = flat / input_w;
          const int64_t iw = flat % input_w;

          const int64_t h0 = oh * 2;
          const int64_t w0 = ow * 2;

          const int64_t local_h = ih - h0;
          const int64_t local_w = iw - w0;

          TORCH_CHECK(
              local_h >= 0 && local_h < 2 && local_w >= 0 && local_w < 2,
              "flat index does not belong to its 2x2 window: "
              "flat=", flat,
              " -> (ih,iw)=(", ih, ",", iw, ")",
              " but window start (h0,w0)=(", h0, ",", w0, ")");

          const int64_t local = local_h * 2 + local_w; // 0..3
          const int64_t byte_index = linear_idx >> 2;
          const int64_t shift = (linear_idx & 3) << 1;

          packed[static_cast<size_t>(byte_index)] |= static_cast<uint8_t>(local << shift);
        }
      }
    }
  }

  return packed;
}

at::Tensor jin_unpack_maxpool2x2_2bit_to_flat_indices(
    const std::vector<uint8_t>& packed,
    int64_t N,
    int64_t C,
    int64_t Hout,
    int64_t Wout,
    int64_t input_h,
    int64_t input_w,
    c10::Device device) {
  TORCH_CHECK(N > 0 && C > 0 && Hout > 0 && Wout > 0, "invalid output shape");
  TORCH_CHECK(input_h > 0 && input_w > 0, "invalid input_h/input_w");

  const int64_t numel = N * C * Hout * Wout;
  const int64_t need_bytes = (numel + 3) / 4;

  TORCH_CHECK(
      static_cast<int64_t>(packed.size()) == need_bytes,
      "packed size mismatch: got=", static_cast<int64_t>(packed.size()),
      " expected=", need_bytes);

  auto out = at::empty(
      {N, C, Hout, Wout},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));

  int64_t* out_ptr = out.data_ptr<int64_t>();

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < Hout; ++oh) {
        for (int64_t ow = 0; ow < Wout; ++ow) {
          const int64_t linear_idx = ((n * C + c) * Hout + oh) * Wout + ow;
          const int64_t byte_index = linear_idx >> 2;
          const int64_t shift = (linear_idx & 3) << 1;
          const uint8_t local =
              static_cast<uint8_t>((packed[static_cast<size_t>(byte_index)] >> shift) & 0x3);

          int64_t local_h = 0;
          int64_t local_w = 0;
          switch (local) {
            case 0: local_h = 0; local_w = 0; break;
            case 1: local_h = 0; local_w = 1; break;
            case 2: local_h = 1; local_w = 0; break;
            case 3: local_h = 1; local_w = 1; break;
            default:
              TORCH_CHECK(false, "invalid local code: ", static_cast<int>(local));
          }

          const int64_t h0 = oh * 2;
          const int64_t w0 = ow * 2;
          const int64_t ih = h0 + local_h;
          const int64_t iw = w0 + local_w;

          TORCH_CHECK(
              ih >= 0 && ih < input_h && iw >= 0 && iw < input_w,
              "reconstructed (ih,iw) out of range: (", ih, ",", iw,
              ") input_h=", input_h,
              " input_w=", input_w);

          out_ptr[linear_idx] = ih * input_w + iw;
        }
      }
    }
  }

  return out.to(device);
}

at::Tensor jin_make_maxpool2d_indices_2bit_tensor(const at::Tensor& flat_indices) {
  TORCH_CHECK(flat_indices.defined(), "flat_indices undefined");
  TORCH_CHECK(flat_indices.dim() == 4, "flat_indices must be [N,C,Hout,Wout]");
  TORCH_CHECK(flat_indices.scalar_type() == at::kLong, "flat_indices must be int64");

  const int64_t Hout = flat_indices.size(2);
  const int64_t Wout = flat_indices.size(3);

  // Assumption: kernel=2, stride=2, padding=0
  const int64_t Hin = Hout * 2;
  const int64_t Win = Wout * 2;

  auto packed = jin_pack_maxpool2x2_flat_indices_to_2bit(flat_indices, Hin, Win);
  return bytes_vec_to_u8_tensor_cpu(packed);
}

// ============================================================
// C API
// ============================================================

extern "C" {

void jin_init_if_needed() {
  std::lock_guard<std::mutex> lk(g_mu);
  ensure_loaded_locked(S());
}

void jin_reset_counters() {
  std::lock_guard<std::mutex> lk(g_mu);
  reset_counters_locked(S());
}

bool jin_has_key(const char* key) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  return st.kv_cpu.find(std::string(key)) != st.kv_cpu.end();
}

bool jin_overwrite_by_key(at::Tensor& t, const char* key) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return false;
  return overwrite_tensor_locked(st, t, std::string(key));
}

// -------------------------
// Conv2d
// -------------------------

void jin_overwrite_conv_input(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("conv2d", st.conv2d_i, "input");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_conv_weight(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("conv2d", st.conv2d_i, "weight");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
  st.conv2d_i += 1;
}

// -------------------------
// ReLU old overwrite path
// -------------------------

void jin_overwrite_relu_saved(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("relu", st.relu_i, "out");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
  st.relu_i += 1;
}

void jin_overwrite_relu(at::Tensor& t) {
  jin_overwrite_relu_saved(t);
}

// -------------------------
// Addmm
// -------------------------

void jin_overwrite_addmm_mat1(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("addmm", st.addmm_i, "mat1");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_addmm_mat2(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("addmm", st.addmm_i, "mat2");
  if (!t.defined()) {
    std::fprintf(stderr, "[JIN] SKIP key=%s (target undefined)\n", key.c_str());
    std::fflush(stderr);
    st.addmm_i += 1;
    return;
  }

  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
  st.addmm_i += 1;
}

void jin_advance_addmm() {
  std::lock_guard<std::mutex> lk(g_mu);
  S().addmm_i += 1;
}

// -------------------------
// MaxPool2d
// -------------------------

void jin_overwrite_maxpool2d_input(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("maxpool2d", st.maxpool2d_i, "input");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_maxpool2d_indices(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  std::fprintf(
      stderr,
      "[JIN] maxpool idx BEFORE overwrite shape=%s min=%lld max=%lld\n",
      sizes_str(t).c_str(),
      static_cast<long long>(t.min().item<int64_t>()),
      static_cast<long long>(t.max().item<int64_t>()));
  std::fflush(stderr);

  const std::string key = make_key("maxpool2d", st.maxpool2d_i, "indices_2bit");
  auto it = st.kv_cpu.find(key);
  TORCH_CHECK(it != st.kv_cpu.end(), "[JIN] missing key=", key);

  at::Tensor packed_u8 = it->second;
  TORCH_CHECK(packed_u8.defined(), "[JIN] packed tensor undefined for key=", key);
  TORCH_CHECK(packed_u8.scalar_type() == at::kByte, "[JIN] packed tensor must be uint8 for key=", key);

  auto sz = t.sizes();
  TORCH_CHECK(sz.size() == 4, "maxpool indices must be 4D");

  const int64_t N = sz[0];
  const int64_t C = sz[1];
  const int64_t Hout = sz[2];
  const int64_t Wout = sz[3];

  // Assumption: kernel=2, stride=2, padding=0
  const int64_t Hin = Hout * 2;
  const int64_t Win = Wout * 2;

  std::vector<uint8_t> packed = u8_tensor_cpu_to_bytes_vec(packed_u8);

  at::Tensor restored = jin_unpack_maxpool2x2_2bit_to_flat_indices(
      packed, N, C, Hout, Wout, Hin, Win, t.device());

  TORCH_CHECK(restored.numel() == t.numel(), "[JIN] restored numel mismatch for key=", key);
  TORCH_CHECK(restored.scalar_type() == t.scalar_type(), "[JIN] restored dtype mismatch for key=", key);

  t.copy_(restored);

  std::fprintf(
      stderr,
      "[JIN] maxpool idx AFTER overwrite shape=%s min=%lld max=%lld\n",
      sizes_str(t).c_str(),
      static_cast<long long>(t.min().item<int64_t>()),
      static_cast<long long>(t.max().item<int64_t>()));
  std::fflush(stderr);

  std::fprintf(
      stderr,
      "[JIN][APPLY] key=%s packed_bytes=%lld restored_bytes=%lld\n",
      key.c_str(),
      static_cast<long long>(packed.size()),
      static_cast<long long>(t.numel() * t.element_size()));
  std::fflush(stderr);

  st.maxpool2d_i += 1;
}

// -------------------------
// Memory payload
// -------------------------

void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();

  TORCH_CHECK(data != nullptr, "[JIN] data is null");
  TORCH_CHECK(nbytes > 0, "[JIN] nbytes=0");
  TORCH_CHECK(nbytes < (1ULL << 32), "[JIN] payload too large");

  st.mem_buf.resize(static_cast<size_t>(nbytes));
  std::memcpy(st.mem_buf.data(), data, static_cast<size_t>(nbytes));
  st.mem_step = step;
  st.mem_has = true;

  // Force reload
  st.loaded = false;

  // Eager load now
  st.step = step;
  load_payload_from_mem_locked(st);
  reset_counters_locked(st);
}

// -------------------------
// BatchNorm
// -------------------------

void jin_overwrite_batchnorm_input(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "input");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_batchnorm_running_mean(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "running_mean");
  if (!t.defined()) {
    std::fprintf(stderr, "[JIN] SKIP key=%s (target undefined)\n", key.c_str());
    std::fflush(stderr);
    return;
  }
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_batchnorm_running_var(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "running_var");
  if (!t.defined()) {
    std::fprintf(stderr, "[JIN] SKIP key=%s (target undefined)\n", key.c_str());
    std::fflush(stderr);
    return;
  }
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_batchnorm_weight(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "weight");
  if (!t.defined()) {
    std::fprintf(stderr, "[JIN] SKIP key=%s (target undefined)\n", key.c_str());
    std::fflush(stderr);
    return;
  }
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_batchnorm_result1(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "result1");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
}

void jin_overwrite_batchnorm_result2(at::Tensor& t) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);
  if (st.role != "B") return;

  const std::string key = make_key("batchnorm", st.batchnorm_i, "result2");
  TORCH_CHECK(overwrite_tensor_locked(st, t, key), "[JIN] missing key=", key);
  st.batchnorm_i += 1;
}

// -------------------------
// ReLU mask APIs
// -------------------------

void jin_capture_relu_mask(const at::Tensor& relu_out) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();

  // Refresh role from env. On A we do not load payload.
  ensure_loaded_locked(st);
  if (st.role != "A") return;

  const int64_t i = st.relu_i++;
  const std::string key = relu_mask_key(i);

  at::Tensor mask_bytes = pack_relu_mask_bits_from_out(relu_out);
  st.kv_cpu[key] = mask_bytes;
}

at::Tensor jin_relu_backward_from_mask(const at::Tensor& grad) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  ensure_loaded_locked(st);

  // On baseline / Node A / role unset, just pass through
  if (st.role != "B") {
    return grad;
  }

  TORCH_CHECK(grad.defined(), "jin_relu_backward_from_mask: grad is undefined");

  const int64_t i = st.relu_i++;
  const std::string key = relu_mask_key(i);

  auto it = st.kv_cpu.find(key);
  if (it == st.kv_cpu.end()) {
    // safer fallback for debugging order mismatch
    return grad;
  }

  at::Tensor mask_bytes = it->second;
  TORCH_CHECK(mask_bytes.defined(), "jin_relu_backward_from_mask: mask undefined");
  TORCH_CHECK(mask_bytes.is_cpu(), "jin_relu_backward_from_mask: mask must be CPU");
  TORCH_CHECK(mask_bytes.scalar_type() == at::kByte, "jin_relu_backward_from_mask: mask must be uint8");

  mask_bytes = mask_bytes.contiguous();
  const uint8_t* mb = mask_bytes.data_ptr<uint8_t>();

  at::Tensor g = grad;
  if (!g.is_contiguous()) g = g.contiguous();

  const int64_t n = g.numel();
  TORCH_CHECK(mask_bytes.numel() * 8 >= n,
              "jin_relu_backward_from_mask: mask too small. mask_bits=",
              mask_bytes.numel() * 8, " n=", n);

  // CUDA path: unpack on CPU -> move -> multiply
  if (g.is_cuda()) {
    at::Tensor mask_u8_cpu = at::empty(
        {n},
        at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    uint8_t* mp = mask_u8_cpu.data_ptr<uint8_t>();
    for (int64_t j = 0; j < n; ++j) {
      mp[j] = bit_get(mb, j) ? 1 : 0;
    }

    at::Tensor mask_u8 = mask_u8_cpu.to(g.device(), /*non_blocking=*/false);
    at::Tensor mask_f = mask_u8.to(g.scalar_type()).view_as(g);
    return g * mask_f;
  }

  // CPU path: direct copy-or-zero
  at::Tensor out = at::empty_like(g);
  const auto dt = g.scalar_type();

  if (dt == at::kFloat) {
    const float* gp = g.data_ptr<float>();
    float* op = out.data_ptr<float>();
    for (int64_t j = 0; j < n; ++j) op[j] = bit_get(mb, j) ? gp[j] : 0.0f;
  } else if (dt == at::kDouble) {
    const double* gp = g.data_ptr<double>();
    double* op = out.data_ptr<double>();
    for (int64_t j = 0; j < n; ++j) op[j] = bit_get(mb, j) ? gp[j] : 0.0;
  } else if (dt == at::kHalf) {
    const at::Half* gp = g.data_ptr<at::Half>();
    at::Half* op = out.data_ptr<at::Half>();
    const at::Half z = static_cast<at::Half>(0);
    for (int64_t j = 0; j < n; ++j) op[j] = bit_get(mb, j) ? gp[j] : z;
  } else if (dt == at::kBFloat16) {
    const at::BFloat16* gp = g.data_ptr<at::BFloat16>();
    at::BFloat16* op = out.data_ptr<at::BFloat16>();
    const at::BFloat16 z = static_cast<at::BFloat16>(0);
    for (int64_t j = 0; j < n; ++j) op[j] = bit_get(mb, j) ? gp[j] : z;
  } else {
    TORCH_CHECK(false, "jin_relu_backward_from_mask: unsupported grad dtype: ", dt);
  }

  return out;
}

bool jin_is_role_B() {
  std::lock_guard<std::mutex> lk(g_mu);
  auto& st = S();
  st.role = env_cstr("JIN_ROLE");
  return st.role == "B";
}

} // extern "C"