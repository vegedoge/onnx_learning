#include <torch/torch.h>

torch::Tensor my_add(torch::Tensor x, torch::Tensor y) {
  return 2 * x + y;
}

PYBIND11_MODULE(my_lib, m){
  m.def("my_add", my_add);
}

