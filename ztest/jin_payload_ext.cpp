#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cstdint>

extern "C" void jin_set_payload_bytes(const void* data, uint64_t nbytes, int64_t step);

namespace py = pybind11;

PYBIND11_MODULE(jin_payload_ext, m) {
  m.def("set_payload", [](py::bytes b, int64_t step) {
    std::string s = b; // copies
    jin_set_payload_bytes((const void*)s.data(), (uint64_t)s.size(), step);
  }, py::arg("payload_bytes"), py::arg("step"));
}