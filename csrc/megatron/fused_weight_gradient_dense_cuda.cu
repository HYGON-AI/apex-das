// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
// Source provenance is documented in THIRD_PARTY_NOTICES.md.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

// #include <ATen/cuda/tunable/GemmHipblaslt.h>

#include "type_shim.h"
#include "utils.h"
// BF16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::BFloat16* A,
    int lda,
    at::BFloat16* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
#ifdef USE_ROCM
//#ifdef __HIP_PLATFORM_HCC__
if (std::getenv("USE_HIPBLASLT") && std::string(std::getenv("USE_HIPBLASLT")) == "1") {
  hipblasLtMatrixLayout_t matA, matB, matC;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA,
                                                    HIP_R_16BF,
                                                    transa == HIPBLAS_OP_N ? m : k,
                                                    transa == HIPBLAS_OP_N ? k : m,
                                                    lda));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB,
                                                    HIP_R_16BF,
                                                    transb == HIPBLAS_OP_N ? k : n,
                                                    transb == HIPBLAS_OP_N ? n : k,
                                                    ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC,
                                                    HIP_R_32F,
                                                    m,
                                                    n,
                                                    ldc));

  hipblasLtMatmulDesc_t matmulDesc;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmulDesc,
                                                  HIPBLAS_COMPUTE_32F,
                                                  HIP_R_32F));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa,
                                                        sizeof(transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb,
                                                        sizeof(transb)));
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(matmulDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));

  hipblasLtMatmulPreference_t pref;
  size_t workspaceSize = 1 << 22;
  auto lt_workspace = at::empty(workspaceSize, at::ScalarType::BFloat16);
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref,
                                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize,
                                                              sizeof(workspaceSize)));

  int device_id;
  CHECK_HIP_ERROR(hipGetDevice(&device_id));

  hipblasLtHandle_t lt_handle = cached_handles.get(device_id);
  if (lt_handle == nullptr)
    lt_handle = cached_handles.obtain(device_id);

  hipStream_t stream;
  CHECK_HIP_ERROR(hipblasGetStream(lt_handle, &stream));

  const int request_solutions = 1;
  hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
  int returnedAlgoCount = 0;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(lt_handle,
                                                        matmulDesc,
                                                        matA,
                                                        matB,
                                                        matC,
                                                        matC,
                                                        pref,
                                                        request_solutions,
                                                        heuristicResult,
                                                        &returnedAlgoCount));

  if (returnedAlgoCount == 0) {
    std::cerr << "No valid solution found!" << std::endl;
    return;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(lt_handle,
                                        matmulDesc,
                                        alpha,
                                        A,
                                        matA,
                                        B,
                                        matB,
                                        beta,
                                        C,
                                        matC,
                                        C,
                                        matC,
                                        &heuristicResult[0].algo,
                                        (void*) (lt_workspace.data_ptr<at::BFloat16>()),
                                        workspaceSize,
                                        stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmulDesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
  return;
  }else{
//#elif defined(USE_ROCM)
    // hipBLAS
    TORCH_CUDABLAS_CHECK((hipblasGemmEx(handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        A,
        HIP_R_16BF,
        lda,
        B,
        HIP_R_16BF,
        ldb,
        beta,
        C,
        HIP_R_32F,
        ldc,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT)));
  }
#else
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16BF,
      lda,
      B,
      CUDA_R_16BF,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

// FP16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
#ifdef USE_ROCM
  TORCH_CUDABLAS_CHECK((hipblasGemmEx(handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      HIP_R_16F,
      lda,
      B,
      HIP_R_16F,
      ldb,
      beta,
      C,
      HIP_R_32F,
      ldc,
      HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT)));
#else
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

// FP32 wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc) {
#ifdef USE_ROCM
  TORCH_CUDABLAS_CHECK((hipblasGemmEx(handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      HIP_R_32F,
      lda,
      B,
      HIP_R_32F,
      ldb,
      beta,
      C,
      HIP_R_32F,
      ldc,
      HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT)));
#else
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

template <typename T>
void wgrad_gemm_accum_fp32_cuda(T *input, T *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha = 1.0;
    const float beta  = 1.0;

    gemmex_wrapper(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        in_dim,
        out_dim,
        hidden_dim,
        &alpha,
        input,
        in_dim,
        d_output,
        out_dim,
        &beta,
        d_weight,
        in_dim);
}

template void wgrad_gemm_accum_fp32_cuda<at::Half>(at::Half *input, at::Half *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<float>(float *input, float *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);


void wgrad_gemm_accum_fp32_cuda_stub(
  at::Tensor &input,
  at::Tensor &d_output,
  at::Tensor &d_weight
) {
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    const int hidden_dim = input_2d.size(0);
    const int in_dim = input_2d.size(1);
    const int out_dim = d_weight.size(0);

    DISPATCH_FLOAT_HALF_AND_BFLOAT(input_2d.scalar_type(), 0, "wgrad_gemm_accum_fp32",
        wgrad_gemm_accum_fp32_cuda<scalar_t_0>(
            input_2d.data_ptr<scalar_t_0>(),
            d_output_2d.data_ptr<scalar_t_0>(),
            d_weight.data_ptr<float>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}
