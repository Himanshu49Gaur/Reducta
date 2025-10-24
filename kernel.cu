#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size) {
    __shared__ float sdata[BLOCK_SIZE];

    // Global and thread indices
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load elements into shared memory (handling boundary conditions)
    float sum = 0.0f;
    if (index < size) sum += in[index];  // First half
    if (index + blockDim.x < size) sum += in[index + blockDim.x];  // Second half

    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block’s partial sum to output
    if (tid == 0) {
        printf("Block %d sum: %f\n", blockIdx.x, sdata[0]);  // DEBUG
        out[blockIdx.x] = sdata[0];
    }
}
