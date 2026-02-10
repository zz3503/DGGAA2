// knn_api.cpp: Python bindings (pybind11)
#include <torch/extension.h>
#include "knn_cuda.h"

// Expose wrapper to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_cuda_wrapper", &knn_cuda_wrapper, "knn_cuda_wrapper");
    m.def("ball_dist_wrapper", &ball_dist_wrapper, "ball_dist_wrapper");
    m.def("dynamic_graph_topk_wrapper", &dynamic_graph_topk_wrapper, "dynamic_graph_topk_wrapper");
}
