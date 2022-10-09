#ifndef GINN_PY_DEV_PY_H
#define GINN_PY_DEV_PY_H

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <ginn/dev.h>

namespace ginn {
namespace python {

namespace py = pybind11;

inline void bind_dev(py::module_& m) {
  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", CPU)
      .value("GPU", GPU)
      .value("NULL_DEV", NULL_DEV);

  py::class_<DeviceId>(m, "DeviceId")
      .def("__repr__",
           [](const DeviceId& i) {
             return std::string("<DeviceId with type: ") +
                    (i.type == 0     ? "CPU"
                     : (i.type == 1) ? "GPU"
                                     : "NULL_DEV") +
                    ", idx: " + std::to_string(i.idx) + ">";
           })
      .def(py::self == py::self);

  py::class_<Device>(m, "Device")
      .def("type", &Device::type)
      .def("id", &Device::id)
      .def("precedence", &Device::precedence);

  py::class_<Cpu, Device>(m, "Cpu").def(py::init<>());

  py::class_<PreallocCpu, Device>(m, "PreallocCpu")
      .def(py::init<size_t>())
      .def("reset", &PreallocCpu::reset)
      .def("used", &PreallocCpu::used);
}

} // namespace python
} // namespace ginn

#endif
