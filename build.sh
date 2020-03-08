cd ..
make clean
mkdir build
make build/RayTracing_cuda
cd build
nvprof ./RayTracing_cuda 800 600 60