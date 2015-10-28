#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

namespace matrix {

  template <const unsigned int W, const unsigned int H>
  class Matrix {
  public:
    Matrix<W, H>() {
      // todo: assert width and height > 0
      matrix = new float[W*H];
    }

    Matrix(const Matrix<W, H>&) = delete;
    Matrix& operator=(const Matrix<W, H>&) = delete;

    Matrix(const Matrix<W, H>&& rhs): matrix(rhs.matrix) {
    }

    cl::Buffer createBuffer(cl::Context& context, cl_mem_flags flags) const {
      if (flags != CL_MEM_WRITE_ONLY) {
        return cl::Buffer(context, flags, this->size() * sizeof(float), matrix);
      } else {
        return cl::Buffer(context, flags, this->size() * sizeof(float));
      }
    }

    float* get() {
      return matrix;
    }

    unsigned int getHeight() const {
      return H;
    }

    unsigned int getWidth() const {
      return W;
    }

    unsigned int size() const {
      return W*H;
    }

    void print() const {
      for (int i = 0; i < H*W; i++){
        std::cout << this->matrix[i] << " ";
        if ((i != 0) && ((i-1) % W == 0)) {
          std::cout << "\n";
        }
      }
      std::cout << std::endl;
    }

  private:
    float *matrix;
  };

  template <const unsigned int W, const unsigned int H>
  inline Matrix<W, H> randmat() {
    Matrix<W, H> m;
    float* const __restrict__ p = m.get();
    for (unsigned int i = 0; i < m.size(); ++i) {
      p[i] = ::rand() / (float) RAND_MAX;
    }
    return m;
  }

  template <const unsigned int W, const unsigned int H>
  inline Matrix<W, H> zeromat() {
    Matrix<W, H> m;
    std::fill_n(m.get(), m.size(), 0);
    return m;
  }

  template <const unsigned int DIM>
  inline Matrix<DIM, 1> randvec() {
    return matrix::randmat<DIM, 1>();
  }

  template <const unsigned int DIM>
  inline Matrix<DIM, 1> zerovec() {
    return matrix::zeromat<DIM, 1>();
  }

  namespace op {

    namespace {
      void buildProgram(cl::Context& context, cl::Program& program) {
        try {
          program.build();
        } catch (cl::Error error) {
          if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::vector<cl::Device> devices;
            devices = context.getInfo<CL_CONTEXT_DEVICES>();
            std::string built = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cerr << built << "\n";
          }
          throw error;
        }
      }
    }

    template<const unsigned int AW, const unsigned int AH, const unsigned int BW, const unsigned int BH>
    void multiply(matrix::Matrix<AW, AH>& matA, matrix::Matrix<BW, BH>& matB) {
      try {
        auto result = matrix::zeromat<AW, BH>();

        cl::Context context(DEVICE);
        cl::Program program = cl::Program(context, "matmul_kernel.cl");
        buildProgram(context, program);

        auto mmul = cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer,
                                    cl::LocalSpaceArg, cl::LocalSpaceArg>(program, "mmul");

        cl::Buffer cl_matA = matA.createBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        cl::Buffer cl_matB = matB.createBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        cl::Buffer cl_result = result.createBuffer(context, CL_MEM_WRITE_ONLY);

        int blocksize = 16;
        cl::LocalSpaceArg A_block = cl::Local(sizeof(float) * blocksize*blocksize);
        cl::LocalSpaceArg B_block = cl::Local(sizeof(float) * blocksize*blocksize);

        cl::CommandQueue queue(context);
        mmul(
          cl::EnqueueArgs(queue,
                          cl::NDRange(matA.getWidth(),
                                      matA.getHeight()),
                          cl::NDRange(blocksize, blocksize)),
          matA.getWidth(),
          cl_matA,
          cl_matB,
          cl_result,
          A_block,
          B_block);

        queue.enqueueReadBuffer(cl_result, CL_TRUE, 0,
                                matA.size() * sizeof(float), result.get());
        result.print();
      } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
          << "ERROR: "
          << err.what()
          << "(" << err.err() << ")"
          << std::endl;
      }
    }

    template<const unsigned int AW, const unsigned int AH, const unsigned int BDIM>
    void multiply(matrix::Matrix<AW, AH>& mat, matrix::Matrix<BDIM, 1>& vec) {
      try {
        auto result_vector = matrix::zerovec<AW>();

        cl::Context context(DEVICE);
        cl::Program program = cl::Program(context, "matvec_mul.cl");
        buildProgram(context, program);

        auto mmul =
          cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "matrixVectorMul");

        cl::Buffer cl_mat = mat.createBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        cl::Buffer cl_vec = vec.createBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
        cl::Buffer cl_result_vector = result_vector.createBuffer(context, CL_MEM_WRITE_ONLY);

        cl::CommandQueue queue(context);
        mmul(
          cl::EnqueueArgs(queue, cl::NDRange(mat.getHeight())),
          cl_result_vector,
          cl_mat,
          cl_vec,
          mat.getWidth()
          );

        queue.enqueueReadBuffer(cl_result_vector, CL_TRUE, 0, mat.getWidth() * sizeof(float), result_vector.get());
        result_vector.print();
      } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
          << "ERROR: "
          << err.what()
          << std::endl;
      }
    }
  }

} // namespace matrix

#endif // MATRIX_HPP_
