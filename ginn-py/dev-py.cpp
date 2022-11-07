//#include <pybind11/operators.h>
// NOTE: Using py::self way of operator binding yields template type deduction
// errors in the particular nvcc I'm using.
#include <pybind11/pybind11.h>

#include <ginn/dev.h>

#include "dev-py.h"

namespace ginn {
namespace python {

namespace py = pybind11;

void bind_dev(py::module_& m) {
  using namespace py::literals;

  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", CPU)
      .value("GPU", GPU)
      .value("NULL_DEV", NULL_DEV);

  py::class_<DeviceId>(m, "DeviceId")
      .def("__repr__",
           [](const DeviceId& i) -> std::string {
             return std::string("<DeviceId with type: ") +
                    (i.type == 0     ? "CPU"
                     : (i.type == 1) ? "GPU"
                                     : "NULL_DEV") +
                    ", idx: " + std::to_string(i.idx) + ">";
           })
      .def(
          "__eq__",
          [](const DeviceId& a, const DeviceId& b) { return a == b; },
          py::is_operator());

  py::class_<Device, std::shared_ptr<Device>>(m, "Device")
      .def("type", &Device::type)
      .def("id", &Device::id)
      .def("precedence", &Device::precedence);

  py::class_<CpuDevice, Device, std::shared_ptr<CpuDevice>>(m, "CpuDevice");

  m.def("Cpu", &Cpu, "");
  m.def("cpu", &cpu, "");

  py::class_<PreallocCpuDevice, Device, std::shared_ptr<PreallocCpuDevice>>(
      m, "PreallocCpuDevice")
      .def("reset", &PreallocCpuDevice::reset)
      .def("size", &PreallocCpuDevice::size)
      .def("used", &PreallocCpuDevice::used);

  m.def("PreallocCpu", &PreallocCpu, "");

#ifdef GINN_ENABLE_GPU
  py::class_<GpuDevice, Device, std::shared_ptr<GpuDevice>>(m, "GpuDevice");

  m.def("Gpu", &Gpu, py::arg("gpu_idx") = 0);
  m.def("gpu", &gpu, py::arg("gpu_idx") = 0);

  py::class_<PreallocGpuDevice, Device, std::shared_ptr<PreallocGpuDevice>>(
      m, "PreallocGpuDevice")
      .def("reset", &PreallocGpuDevice::reset)
      .def("size", &PreallocGpuDevice::size)
      .def("used", &PreallocGpuDevice::used);

  m.def("PreallocGpu",
        py::overload_cast<size_t, size_t>(&PreallocGpu),
        "idx"_a,
        "size"_a);

#endif

  m.def("gpus", &gpus);
}

} // namespace python
} // namespace ginn
