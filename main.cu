#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iostream>

__global__ void initializeMatrix(float* matrix, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < size && idy < size && idz < size) {
        int index = idz * size * size + idy * size + idx;
        curandStatePhilox4_32_10_t localState;
        curand_init(0, index, 0, &localState);
        matrix[index] = curand_uniform4(&localState).x;
    }
}

__global__ void initializeFilter(float* filter, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx < size && idy < size && idz < size) {
        int index = idz * size * size + idy * size + idx;
	if (idx == idy || idx == (size - idy - 1)) {
            filter[index] = 1;
        } else {
            filter[index] = 0;
        }
    }
}


__global__ void convolutionParallel(float* matrix, float* filter, float* result, int size, int m){
    int filterSize = m;
    int halfFilter = filterSize / 2;

    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Realizar la convolución
    if (z < size && y < size && x < size) {
        float sum = 0.0f;
        for (int fz = 0; fz < filterSize; fz++) {
            for (int fy = 0; fy < filterSize; fy++) {
                for (int fx = 0; fx < filterSize; fx++) {
                    int nz = z + fz - halfFilter;
                    int ny = y + fy - halfFilter;
                    int nx = x + fx - halfFilter;
                    if (nz >= 0 && nz < size && ny >= 0 && ny < size && nx >= 0 && nx < size) {
                        int filterIndex = fz * filterSize * filterSize + fy * filterSize + fx;
                        sum += matrix[nz * size * size + ny * size + nx] * filter[filterIndex];
                    }
                }
            }
        }
        result[z * size * size + y * size + x] = sum;
    }
}


void convolutionSecuencial(float* matrix, float* filter, float* result, int size, int m) {
    int filterSize = m;
    int halfFilter = filterSize / 2;

    // Realizar la convolución
    for (int z = 0; z < size; z++) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                float sum = 0.0f;
                for (int fz = 0; fz < filterSize; fz++) {
                    for (int fy = 0; fy < filterSize; fy++) {
                        for (int fx = 0; fx < filterSize; fx++) {
                            int nz = z + fz - halfFilter;
                            int ny = y + fy - halfFilter;
                            int nx = x + fx - halfFilter;
                            if (nz >= 0 && nz < size && ny >= 0 && ny < size && nx >= 0 && nx < size) {
                                int filterIndex = fz * filterSize * filterSize + fy * filterSize + fx;
                                sum += matrix[nz * size * size + ny * size + nx] * filter[filterIndex];
			    }
                        }
                    }
                }
                result[z * size * size + y * size + x] = sum;
            }
        }
    }
}

void printMatrix(float* matrix, int size) {
    // Imprimir la matriz
    for (int z = 0; z < size; z++) {
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                printf("%.3f ", matrix[z * size * size + y * size + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Ejecutar como ./prog <n> <m> <mode>\n");
        return 1;
    }

    int size = atoi(argv[1]);
    int mode = atoi(argv[3]);
    int m = atoi(argv[2]);
    
    if (m > size) {
        printf("<m> no puede ser mayor a <n>\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t matrixSize = size * size * size * sizeof(float);

    // Alojar memoria en el dispositivo
    float* deviceMatrix;
    float* deviceResult;
    float* deviceFilter;
    size_t resultSize = size * size * size * sizeof(float);
    size_t filterDeviceSize = m * m * m * sizeof(float);


    cudaMalloc((void**)&deviceMatrix, matrixSize);
    cudaMalloc((void**)&deviceResult, resultSize);
    cudaMalloc((void**)&deviceFilter, filterDeviceSize);

    // Configurar las dimensiones del bloque y de la cuadrícula
    dim3 blockDim(64, 4, 4);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
                 (size + blockDim.y - 1) / blockDim.y,
                 (size + blockDim.z - 1) / blockDim.z);

    dim3 gridDimF((m + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y,
                 (m + blockDim.z - 1) / blockDim.z);
   
    
    // Lanzar el kernel en el dispositivo
    initializeMatrix<<<gridDim, blockDim>>>(deviceMatrix, size);

    // Esperar a que todos los hilos terminen
    cudaDeviceSynchronize();

    // Copiar la matriz desde el dispositivo a la memoria del host
    float* hostMatrix = (float*)malloc(matrixSize);
    cudaMemcpy(hostMatrix, deviceMatrix, matrixSize, cudaMemcpyDeviceToHost);

    // Imprimir la matriz
    if (size < 3) {
        printMatrix(hostMatrix, size);
    }

    
    // Lanzar el kernel en el dispositivo
    initializeFilter<<<gridDimF, blockDim>>>(deviceFilter, m);

    // Esperar a que todos los hilos terminen
    cudaDeviceSynchronize();

    // Copiar el filtro desde el dispositivo a la memoria del host
    float* hostFilter = (float*)malloc(filterDeviceSize);
    cudaMemcpy(hostFilter, deviceFilter, filterDeviceSize, cudaMemcpyDeviceToHost);

    // Imprimir el filtro
    if (m < 33 && size < 3){
        printMatrix(hostFilter, m);
    }    
    
    float* hostResult = (float*)malloc(resultSize);

    if(mode == 1){
        // Realizar la convolució secuencia
        cudaEventRecord(start);
        convolutionSecuencial(hostMatrix, hostFilter, hostResult, size, m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);
        std::cout << "Tiempo de ejecución: " << elapsed << " ms" << std::endl;

    }if(mode == 2){
        cudaEventRecord(start);
        convolutionParallel<<<gridDim, blockDim>>>(deviceMatrix, deviceFilter, deviceResult, size, m);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);
        std::cout << "Tiempo de ejecución: " << elapsed << " ms" << std::endl;
        cudaMemcpy(hostResult, deviceResult, matrixSize, cudaMemcpyDeviceToHost);
    }

    // Imprimir el resultado
    if (size < 3) {
        printf("Convolucion:\n");
        printMatrix(hostResult, size);
    }

    // Liberar la memoria
    cudaFree(deviceMatrix);
    cudaFree(deviceResult);
    cudaFree(deviceFilter);
    free(hostMatrix);
    free(hostResult);
    free(hostFilter);

    return 0;
}
