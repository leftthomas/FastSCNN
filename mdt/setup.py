from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

# requirements = ["torch", "torchvision"]
#
# def get_extensions():
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#
#     main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
#     source_cuda = glob.glob(os.path.join(this_dir, "*.cu"))
#
#     sources = main_file
#     extension = CppExtension
#     extra_compile_args = {"cxx": ["-D_MWAITXINTRIN_H_INCLUDED"]}
#     define_macros = []
#
#     if torch.cuda.is_available() and CUDA_HOME is not None:
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]
#     else:
#         raise NotImplementedError('Cuda is not availabel')
#
#     sources = [os.path.join(this_dir, s) for s in sources]
#     include_dirs = [this_dir]
#     ext_modules = [
#         extension(
#             "mdconv",
#             sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ]
#     return ext_modules
#
# setup(
#     name="mdconv",
#     version="1.0",
#     author="wzj",
#     url="https://github.com/wantsjean",
#     description="md conv",
#     packages=find_packages(exclude=("configs", "tests",)),
#     # install_requires=requirements,
#     ext_modules=get_extensions(),
#     cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},)


setup(
    name='mdconv',
    version="1.0",
    ext_modules=[
        CUDAExtension('mdconv', [
            'md_conv_cuda.cpp',
            'md_conv_cuda_kernel.cu',
        ]),
    ],
    author="wzj",
    cmdclass={
        'build_ext': BuildExtension
    })
