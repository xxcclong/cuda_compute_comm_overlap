/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <vector>
#include <thread>
#include <assert.h>
#include <functional>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

// 1: element wise sqrt 2: element wise mul others: cublas gemm
#define TEST_ALGO 2
#define WAIT_EVENT

/* Matrix size */
#define N (32 * 4)
#define M (32 * 4)

dim3 cuda_threads(64);
dim3 cuda_grid(N / 64);

#define GPUS (4)
#define ITERATIONS (200)
#define PRERUN_ITER (100)
#define COMPUTE_TIME 20
#define COMM_TIME 1
//float *h_C_ref;
float* d_A[GPUS];
float* d_B[GPUS];
float* d_C[GPUS];
float* d_D[GPUS];
float alpha = 1.0f;
float beta = 0.0f;
int n2 = N * N;
int i;

enum NCCL_MODE {
    ASYNC = 0,
    SYNC = 1,
    ONE_STREAM = 2,
    NO_COMM = 3,
    NO_COMPUTE = 4
};


std::unique_ptr<ncclComm_t[]> comms = nullptr;
std::unique_ptr<cudaStream_t[]> nccl_streams = nullptr;
std::unique_ptr<cudaStream_t[]> blas_streams = nullptr;
size_t timestamp() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               high_resolution_clock::now().time_since_epoch()).count();
}

void init_nccl() {
    comms.reset(new ncclComm_t[GPUS]);
    nccl_streams.reset(new cudaStream_t[GPUS]);
    blas_streams.reset(new cudaStream_t[GPUS]);
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    ncclGroupStart();
    for (size_t i = 0; i < GPUS; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(nccl_streams.get() + i);
        ncclCommInitRank(comms.get() + i, GPUS, nccl_id, i);
        cudaStreamCreate(blas_streams.get() + i);
    }
    ncclGroupEnd();
}

int init_data(int dev) {
    float* ha;
    float* hb;
    float* hc;
    float* hd;
    //float *h_C_ref;
    d_A[dev] = 0;
    d_B[dev] = 0;
    d_C[dev] = 0;
    d_D[dev] = 0;
    //float *da = *d_A[dev] = 0;
    //float *db = *d_B[dev] = 0;
    //float *dc = *d_C[dev] = 0;
    cudaSetDevice(dev);
    /* Allocate host memory for the matrices */
    ha = reinterpret_cast<float*>(malloc(n2 * sizeof(ha[0])));
    hb = reinterpret_cast<float*>(malloc(n2 * sizeof(hb[0])));
    hc = reinterpret_cast<float*>(malloc(n2 * sizeof(hc[0])));
    hd = reinterpret_cast<float*>(malloc(M * sizeof(hd[0])));

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++) {
        ha[i] = rand() / static_cast<float>(RAND_MAX);
        hb[i] = rand() / static_cast<float>(RAND_MAX);
        hc[i] = rand() / static_cast<float>(RAND_MAX);
    }
    for (i = 0; i < M; i++)
        hd[i] = rand() / static_cast<float>(RAND_MAX);


    /* Allocate device memory for the matrices */
    if (cudaMalloc(reinterpret_cast<void**>(&d_A[dev]), n2 * sizeof(d_A[dev][0])) !=
            cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_B[dev]), n2 * sizeof(d_B[dev][0])) !=
            cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_C[dev]), n2 * sizeof(d_C[dev][0])) !=
            cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_D[dev]), M * sizeof(d_D[dev][0])) !=
            cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate D)\n");
        return EXIT_FAILURE;
    }


    /* Initialize the device matrices with the host matrices */
    cublasSetVector(n2, sizeof(ha[0]), ha, 1, d_A[dev], 1);
    cublasSetVector(n2, sizeof(hb[0]), hb, 1, d_B[dev], 1);
    cublasSetVector(n2, sizeof(hc[0]), hc, 1, d_C[dev], 1);
    return 0;
}

int destroy_data(int dev) {
    //float *h_C_ref;
    float* da = d_A[dev];
    float* db = d_B[dev];
    float* dc = d_C[dev];
    float* dd = d_D[dev];
    /* Memory clean up */

    if (cudaFree(da) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(db) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(dc) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(dd) != cudaSuccess) {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }
    return 0;
}

__global__ void eleSqrt(float *a, float* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] += sqrtf(fabsf(a[idx]));
}

__global__ void eleMul(float *a, float* b, float* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] += a[idx] * b[idx];
}


inline void print_compute_func()
{
#ifndef TEST_ALGO
    fprintf(stderr, "algo: cublasSgemm\n");
#elif TEST_ALGO == 1 
    fprintf(stderr, "algo: Sqrt\n");
#elif TEST_ALGO == 2 
    fprintf(stderr, "algo: Element Wise Mul\n");
#else
    fprintf(stderr, "algo: cublasSgemm\n");
#endif
}


inline void compute_func(int dev, cublasHandle_t* handle, cudaStream_t* s)
{
#ifndef TEST_ALGO
    cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A[dev],
                             N, d_B[dev], N, &beta, d_C[dev], N);
#elif TEST_ALGO == 1 
    eleSqrt<<<cuda_grid, cuda_threads, 0, *s>>>(d_A[dev], d_C[dev]);
#elif TEST_ALGO == 2 
    eleMul<<<cuda_grid, cuda_threads, 0, *s>>>(d_A[dev], d_B[dev], d_C[dev]);
#else
    cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A[dev],
                             N, d_B[dev], N, &beta, d_C[dev], N);
#endif
}

int prerun(int dev = 0) {
    cublasStatus_t status;
    cublasHandle_t handle;

    auto& blas_stream = *(blas_streams.get() + dev);
    cudaSetDevice(dev);

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    cublasSetStream(handle, blas_stream);

    /* Performs operation using cublas */
    auto& nccl_stream = *(nccl_streams.get() + dev);
    cudaEvent_t start_event, stop_event;
    float compute_time = 0, comm_time = 0, temp_time = 0;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // run computation
    for (int i = 0; i < PRERUN_ITER; ++i) {
        cudaEventRecord(start_event, 0);
        compute_func(dev, &handle, &blas_stream);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&temp_time, start_event, stop_event);
        compute_time += temp_time;
    }

    // run comm
    for (int i = 0; i < PRERUN_ITER; ++i) {
        cudaEventRecord(start_event, 0);
        ncclAllReduce(d_D[dev], d_D[dev], M, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&temp_time, start_event, stop_event);
        comm_time += temp_time;
    }
    if (dev == 0)
        fprintf(stderr, "compute kernel time %fms\ncomm kernel time %fms\nin theory\nall compute time %fms\nall comm time %fms\ncompute / comm %f\n\n",
                compute_time / PRERUN_ITER,
                comm_time / PRERUN_ITER,
                compute_time / PRERUN_ITER * COMPUTE_TIME * ITERATIONS,
                comm_time / PRERUN_ITER * COMM_TIME * ITERATIONS,
                compute_time * COMPUTE_TIME / (comm_time * COMM_TIME));

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}


/* Main */
int worker(int dev, int nccl_mode) {
    cublasStatus_t status;

    cublasHandle_t handle;
    auto& blas_stream = *(blas_streams.get() + dev);
    cudaSetDevice(dev);

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    cublasSetStream(handle, blas_stream);

    /* Performs operation using cublas */
    auto& nccl_stream = *(nccl_streams.get() + dev);
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    size_t start = timestamp();
    cudaEventRecord(start_event, 0);
    if (nccl_mode == NCCL_MODE::ONE_STREAM) {
        for (size_t i = 0; i < ITERATIONS; ++i) {
            for (int temp = 0; temp < COMPUTE_TIME; ++temp)
                compute_func(dev, &handle, &blas_stream);
            for (int temp = 0; temp < COMM_TIME; ++temp)
                ncclAllReduce(d_D[dev], d_D[dev], M, ncclFloat, ncclSum, *(comms.get() + dev), blas_stream);
        }
        //cudaStreamSynchronize(blas_stream);
    } else {
        // nccl_mode is ASYNC_NCCL or SYNC_NCCL
        for (size_t i = 0; i < ITERATIONS; ++i) {
            if (nccl_mode != NO_COMPUTE) {
                for (int temp = 0; temp < COMPUTE_TIME; ++temp)
                    compute_func(dev, &handle, &blas_stream);
            }



            if (nccl_mode != NO_COMM) {
                for (int temp = 0; temp < COMM_TIME; ++temp) {
                    ncclAllReduce(d_D[dev], d_D[dev], M, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
                    if (nccl_mode == SYNC) {
                        cudaStreamSynchronize(nccl_stream);
                    }
                }
            } else {
                for (int temp = 0; temp < COMM_TIME; ++temp) {
                    ncclAllReduce(d_D[dev], d_D[dev], 1, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
                    if (nccl_mode == SYNC) {
                        cudaStreamSynchronize(nccl_stream);
                    }
                }
            }
        }
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float event_overall_time = 0;
    cudaEventElapsedTime(&event_overall_time, start_event, stop_event);
    fprintf(stderr, "device: [%d], %d iterations spent: cputime [%.2f ms] eventtime [%.2f ms] \n", dev, ITERATIONS, (timestamp() - start) / 1000.0, event_overall_time);
    //fprintf(stderr, "device: [%d], %d iterations spent: [%d ms]\n", dev, ITERATIONS, (timestamp()-start)/1000);


    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}

int worker_with_wait(int dev, int nccl_mode) {
    cublasStatus_t status;

    cublasHandle_t handle;
    auto& blas_stream = *(blas_streams.get() + dev);
    cudaSetDevice(dev);

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    cublasSetStream(handle, blas_stream);

    /* Performs operation using cublas */
    auto& nccl_stream = *(nccl_streams.get() + dev);
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    std::vector<cudaEvent_t> compute_events;
    compute_events.reserve(ITERATIONS);
    for(int i = 0; i < ITERATIONS; ++i) {
        cudaEventCreateWithFlags(&compute_events[i], cudaEventDisableTiming);
    }
    size_t start = timestamp();
    cudaEventRecord(start_event, 0);

    for (size_t i = 0; i < ITERATIONS; ++i) {
        for (int temp = 0; temp < COMPUTE_TIME; ++temp){
            compute_func(dev, &handle, &blas_stream);
        }
        cudaEventRecord(compute_events[i], blas_stream);
        if(i > 0) cudaStreamWaitEvent(nccl_stream, compute_events[i - 1], 0);
        for (int temp = 0; temp < COMM_TIME; ++temp) {
            ncclAllReduce(d_D[dev], d_D[dev], M, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
        }
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    float event_overall_time = 0;
    cudaEventElapsedTime(&event_overall_time, start_event, stop_event);
    fprintf(stderr, "[wait mode] device: [%d], %d iterations spent: cputime [%.2f ms] eventtime [%.2f ms] \n", dev, ITERATIONS, (timestamp() - start) / 1000.0, event_overall_time);


    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("USAGE: ./a.out 2 # 0 sync, 1 async, 2 one stream, 3 only compute, 4 only comm, 5 wait mode\n");
        return -1;
    }

    init_nccl();
    for (int i = 0; i < GPUS; ++i) {
        init_data(i);
    }
    std::vector<std::thread> threads;
    std::vector<std::thread> prerun_threads;
    int nccl_mode = atoi(argv[1]);
    printf("nccl mode %d\n", nccl_mode);
    print_compute_func();
    for (int i = 0; i < GPUS; ++i) {
        std::thread t(std::bind(&prerun, i));
        prerun_threads.push_back(std::move(t));
    }
    for (auto& t : prerun_threads) {
        t.join();
    }
    size_t start = timestamp();
    if(nccl_mode != 5){
        for (int i = 0; i < GPUS; ++i) {
            std::thread t(std::bind(&worker, i, nccl_mode));
            threads.push_back(std::move(t));
        }
        for (auto& t : threads) {
            t.join();
        }
    }
    else {
        for (int i = 0; i < GPUS; ++i) {
            std::thread t(std::bind(&worker_with_wait, i, nccl_mode));
            threads.push_back(std::move(t));
        }
        for (auto& t : threads) {
            t.join();
        }   
    }
    fprintf(stderr, "nccl mode: [%d], spent: [%.2f ms]\n", nccl_mode, (timestamp() - start) / 1000.0);


    for (int i = 0; i < GPUS; ++i) {
        destroy_data(i);
    }
    return 0;
}
