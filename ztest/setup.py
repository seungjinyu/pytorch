from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="jin_payload_ext",
    ext_modules=[
        CppExtension(
            name="jin_payload_ext",
            sources=["jin_payload_ext.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)