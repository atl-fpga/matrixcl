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

void buildProgram(cl::Context *context, cl::Program* program) {
  try {
    program->build();
  } catch (cl::Error error) {
    if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
      std::vector<cl::Device> devices;
      devices = context->getInfo<CL_CONTEXT_DEVICES>();
      std::string built = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cerr << built << "\n";
    }
    throw error;
  }
}

void runMatrixVectorMul() {
  std::unique_ptr<matrix::Matrix> mat(new matrix::Matrix(WIDTH_A, HEIGHT_A));
  std::unique_ptr<matrix::Matrix> vec(new matrix::Matrix(WIDTH_A, 1));
  matrix::initRandoms(*mat);
  matrix::initRandoms(*vec);

  mat->print();
  vec->print();

  switch (DEVICE) {
  case CL_DEVICE_TYPE_DEFAULT: printf("DEVICE=DEFAULT\n"); break;
  case CL_DEVICE_TYPE_CPU:     printf("DEVICE=CPU\n"); break;
  case CL_DEVICE_TYPE_GPU:     printf("DEVICE=GPU\n"); break;
  default:                     printf("DEVICE=%d\n", DEVICE); break;
  }

  matrix::matrixVectorMultiplation(*mat, *vec);
}

void runMatrixMatrixMul() {
  std::unique_ptr<matrix::Matrix> matA(new matrix::Matrix(WIDTH_A, HEIGHT_A));
  std::unique_ptr<matrix::Matrix> matB(new matrix::Matrix(WIDTH_A, HEIGHT_A));
  matrix::initRandoms(*matA);
  matrix::initRandoms(*matB);

  matA->print();
  matB->print();

  switch (DEVICE) {
  case CL_DEVICE_TYPE_DEFAULT: printf("DEVICE=DEFAULT\n"); break;
  case CL_DEVICE_TYPE_CPU:     printf("DEVICE=CPU\n"); break;
  case CL_DEVICE_TYPE_GPU:     printf("DEVICE=GPU\n"); break;
  default:                     printf("DEVICE=%d\n", DEVICE); break;
  }

  matrix::matrixMultiplation(*matA, *matB);
}

int main(int argc, char** argv) {
  //runMatrixVectorMul();
  runMatrixMatrixMul();
  return 0;
}
