__kernel void matrixVectorMul(__global float* resultVector,
			      __global float* matrixA,
			      __global float* vectorB,
			      const int width)
{
  int tx = get_global_id(0);
  float value = 0;
  for (unsigned int k = 0; k < width; ++k) {
    value += matrixA[tx * width + k] * vectorB[k];
  }
  resultVector[tx] = value;
}
