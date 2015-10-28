#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <cmath>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "util.hpp"
#include "matrix.hpp"

#define WIDTH_A (16)
#define HEIGHT_A (16)

void runMatrixVectorMul() {
  auto mat = matrix::randmat<WIDTH_A, HEIGHT_A>();
  auto vec = matrix::randvec<WIDTH_A>();

  mat.print();
  vec.print();

  auto result = matrix::op::multiply(mat, vec);
  result.print();
}

void runMatrixMatrixMul() {
  auto matA = matrix::randmat<WIDTH_A, HEIGHT_A>();
  auto matB = matrix::randmat<WIDTH_A, HEIGHT_A>();

  matA.print();
  matB.print();

  auto result = matrix::op::multiply(matA, matB);
  result.print();
}

inline double estimated_performance_of(const int Mdim, const int Ndim, const int Pdim, const double run_time, const int iters = 1) {
  return iters * 2.0 * Mdim * Ndim * Pdim/(1000000.0f * run_time);
}

void benchmark(const unsigned int iters) {
  util::Timer timer;

  auto mat = matrix::randmat<1024, 1024>();
  for (int i=0; i<iters; ++i) {
    mat = matrix::op::multiply(mat, matrix::randmat<1024, 1024>());
  }

  const double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
  printf(" %.2f seconds at %.1f MFLOPS \n",  run_time, estimated_performance_of(mat.getWidth(), mat.getHeight(), mat.getWidth(), run_time, iters));
}

int main(int argc, char** argv) {
  switch (DEVICE) {
  case CL_DEVICE_TYPE_DEFAULT: printf("DEVICE=DEFAULT\n"); break;
  case CL_DEVICE_TYPE_CPU:     printf("DEVICE=CPU\n"); break;
  case CL_DEVICE_TYPE_GPU:     printf("DEVICE=GPU\n"); break;
  default:                     printf("DEVICE=%d\n", DEVICE); break;
  }

  //runMatrixVectorMul();
  //runMatrixMatrixMul();
  benchmark(10);
  return 0;
}
