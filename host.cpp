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

  matrix::op::multiply(mat, vec);
}

void runMatrixMatrixMul() {
  auto matA = matrix::randmat<WIDTH_A, HEIGHT_A>();
  auto matB = matrix::randmat<WIDTH_A, HEIGHT_A>();

  matA.print();
  matB.print();

  matrix::op::multiply(matA, matB);
}

int main(int argc, char** argv) {
  switch (DEVICE) {
  case CL_DEVICE_TYPE_DEFAULT: printf("DEVICE=DEFAULT\n"); break;
  case CL_DEVICE_TYPE_CPU:     printf("DEVICE=CPU\n"); break;
  case CL_DEVICE_TYPE_GPU:     printf("DEVICE=GPU\n"); break;
  default:                     printf("DEVICE=%d\n", DEVICE); break;
  }

  //runMatrixVectorMul();
  runMatrixMatrixMul();
  return 0;
}
