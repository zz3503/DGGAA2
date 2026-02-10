from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn_api',
    ext_modules=[
        CUDAExtension(
            name='knn_api',
            sources=['src/knn_api.cpp', 'src/knn_cuda.cpp', 'src/knn_gpu.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
