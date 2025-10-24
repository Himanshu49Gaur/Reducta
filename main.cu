#include <stdio.h>
#include <stdlib.h>
#include "support.cu"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    int i;

    // Allocate and initialize host memory
    if(argc == 1) {
        in_elements = 1000000;
    } else if(argc == 2) {
        in_elements = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./reduction          # Input of size 1,000,000 is used"
               "\n    Usage: ./reduction <m>      # Input of size m is used"
               "\n");
        exit(0);
    }

    initVector(&in_h, in_elements);

    out_elements = in_elements / (BLOCK_SIZE << 1);
    if (in_elements % (BLOCK_SIZE << 1)) out_elements++;

    out_h = (float*)malloc(out_elements * sizeof(float));
    if (out_h == NULL) FATAL("Unable to allocate host memory");

    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", in_elements);

    // Allocate device variables ----------------------------------------------
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, in_elements * sizeof(float));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cuda_ret = cudaMalloc((void**)&out_d, out_elements * sizeof(float));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, in_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemset(out_d, 0, out_elements * sizeof(float));
    if (cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x  = out_elements; dim_grid.y = dim_grid.z = 1;

    reduction<<<dim_grid, dim_block>>>(out_d, in_d, in_elements);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
        FATAL("Unable to launch/execute kernel");

    stopTime(&timer);
    printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(out_h, out_d, out_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
        FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Debug: Print Partial Sums ----------------------------------------------
    printf("\nPartial sums from GPU:\n");
    for (i = 0; i < out_elements; i++) {
        printf("%f ", out_h[i]);  // DEBUG OUTPUT
    }
    printf("\n");

    // Final CPU accumulation -------------------------------------------------
    float final_sum = 0.0f;
    for (i = 0; i < out_elements; i++) {
        final_sum += out_h[i];
    }

    printf("Final CPU sum after reduction: %f\n", final_sum);  // DEBUG OUTPUT

    // Verify correctness -----------------------------------------------------
    printf("Verifying results...\n");
    verify(in_h, in_elements, final_sum);

    // Free memory ------------------------------------------------------------
    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}
