make clean
mkdir build
make build/RayTracing_cuda
cd build
nvprof ./RayTracing_cuda 3840 2160 60

