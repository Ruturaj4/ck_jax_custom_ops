mkdir -p build
pybind_include_path=$(python -c "import pybind11; print(pybind11.get_include())")
python_executable=$(python -c 'import sys; print(sys.executable)')

# Compile the kernel.
hipcc -std=c++17 -D__HIP_PLATFORM_AMD__ -Wno-return-type -I/opt/rocm/include\
  -I${CK_HOME}/library/include -I${CK_HOME}/include -L${CK_HOME}/build/lib \
  --offload-arch=gfx942 -c kernel/wrapper.cpp -fPIC -Wall -o  build/gemm.o

# Build gpu ops. 
c++ -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -I$pybind_include_path $(python-config --cflags) \
  -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -o build/gpu_ops.o \
  -c gpu_ops/gpu_ops.cpp

# Link and build complete library.
c++ -D__HIP_PLATFORM_AMD__ -fPIC -O3 -DNDEBUG -O3 -flto -shared -o build/gpu_ops$(python-config --extension-suffix) \
  build/gpu_ops.o build/gemm.o -L/opt/rocm/lib  -lhiprtc -lamdhip64 -lrt -lpthread -ldl -lutility
