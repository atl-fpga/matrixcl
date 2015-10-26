#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

namespace matrix {

  class Matrix {
  public:
    Matrix(const unsigned int w, const unsigned int h);
    float* get();
    cl::Buffer* createBuffer(cl::Context* context, cl_mem_flags flags);
    unsigned int getHeight();
    unsigned int getWidth();
    unsigned int size();
    void print(); 
  private:
    float *matrix;
    const unsigned int height;
    const unsigned int width;
  };

  Matrix::Matrix(const unsigned int w, const unsigned int h) : width(w), height(h) {
    // todo: assert width and height > 0
    matrix = new float[width*height];
  }

  cl::Buffer* Matrix::createBuffer(cl::Context* context, cl_mem_flags flags) {
    if (flags != CL_MEM_WRITE_ONLY) {
      return new cl::Buffer(*context, flags, this->size() * sizeof(float), matrix);
    } else {
      return new cl::Buffer(*context, flags, this->size() * sizeof(float));
    }
  }

  unsigned int Matrix::getHeight() {
    return height;
  }

  unsigned int Matrix::getWidth() {
    return width;
  }

  unsigned int Matrix::size() {
    return width*height;
  }

  float* Matrix::get() {
    return matrix;
  }

  void Matrix::print() {
    for (int i = 0; i < (this->height*this->width); i++){
      std::cout << this->matrix[i] << " ";
      if ((i != 0) && ((i-1) % this->width == 0)) {
	std::cout << "\n";
      }
    }
    std::cout << std::endl; 
  }

  void initRandoms(Matrix::Matrix* mat) {
    float* content = mat->get();
    for (unsigned int i = 0; i < mat->size(); ++i) {
      content[i] = rand() / (float) RAND_MAX;
    }
  }

  void initZeros(Matrix::Matrix* mat) {
    float* content = mat->get();
    std::fill_n(content, mat->size(), 0);
  }

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

  void matrixMultiplation(matrix::Matrix *matA, matrix::Matrix *matB) {
    matrix::Matrix *result = new matrix::Matrix(matA->getWidth(), matA->getHeight());
    try {
      cl::Context context(DEVICE);
      cl::Program program = cl::Program(context, "matmul_kernel.cl");
      buildProgram(&context, &program);

      auto mmul = cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer,
				  cl::LocalSpaceArg, cl::LocalSpaceArg>(program, "mmul");
      
      cl::Buffer *cl_matA = matA->createBuffer(&context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
      cl::Buffer *cl_matB = matB->createBuffer(&context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
      cl::Buffer *cl_result = result->createBuffer(&context, CL_MEM_WRITE_ONLY);

      int blocksize = 16;
      cl::LocalSpaceArg A_block = cl::Local(sizeof(float) * blocksize*blocksize);
      cl::LocalSpaceArg B_block = cl::Local(sizeof(float) * blocksize*blocksize);

      cl::CommandQueue queue(context);
      mmul(
	   cl::EnqueueArgs(queue,
			   cl::NDRange(matA->getWidth(),
				       matA->getHeight()),
			   cl::NDRange(blocksize, blocksize)),
	   matA->getWidth(),
	   *cl_matA,
	   *cl_matB,
	   *cl_result,
	   A_block,
	   B_block);
      
      queue.enqueueReadBuffer(*cl_result, CL_TRUE, 0,
                              matA->size() * sizeof(float), result->get());
      result->print();
    } catch (cl::Error err) {
      std::cout << "Exception\n";
      std::cerr 
	<< "ERROR: "
	<< err.what()
    << "(" << err.err() << ")"
	<< std::endl;
    }

    delete[](result);
  }
  
  void matrixVectorMultiplation(matrix::Matrix *mat, matrix::Matrix *vec) {
    matrix::Matrix *result_vector = new matrix::Matrix(mat->getWidth(), 1);
    try {
      cl::Context context(DEVICE);
      cl::Program program = cl::Program(context, "matvec_mul.cl");
      buildProgram(&context, &program);

      auto mmul =
	cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "matrixVectorMul");
       
      cl::Buffer *cl_mat = mat->createBuffer(&context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
      cl::Buffer *cl_vec = vec->createBuffer(&context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
      cl::Buffer *cl_result_vector = result_vector->createBuffer(&context, CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR);

      cl::CommandQueue queue(context);
      mmul(
	   cl::EnqueueArgs(queue, cl::NDRange(mat->getHeight())),
	   *cl_result_vector,
	   *cl_mat,
	   *cl_vec,
	   mat->getWidth()
	   );

      queue.enqueueReadBuffer(*cl_result_vector, CL_TRUE, 0, mat->getWidth() * sizeof(float), result_vector->get());      
      result_vector->print();
    } catch (cl::Error err) {
      std::cout << "Exception\n";
      std::cerr 
	<< "ERROR: "
	<< err.what()
	<< std::endl;
    }

    delete[](result_vector);
  }

} // namespace matrix
  
#endif // MATRIX_HPP_
