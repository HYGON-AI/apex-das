// SPDX-License-Identifier: BSD-3-Clause
// Source provenance is documented in THIRD_PARTY_NOTICES.md.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <rocblas/rocblas.h>

#include <hipblaslt/hipblaslt.h>

#include "type_shim.h"

#ifdef __HIP_PLATFORM_HCC__
#include "utils.h"
#include <hipblaslt/hipblaslt-ext.hpp>
#endif



hipDataType get_dtype(at::Tensor A, bool use_blas) {
    switch (A.scalar_type()) {
        case at::ScalarType::Float:    return use_blas ? HIPBLAS_R_32F : HIP_R_32F;
        case at::ScalarType::Double:   return use_blas ? HIPBLAS_R_64F : HIP_R_64F;
        case at::ScalarType::Half:     return use_blas ? HIPBLAS_R_16F : HIP_R_16F;
        case at::ScalarType::BFloat16:     return use_blas ? HIPBLAS_R_16B : HIP_R_16BF;
        default: throw std::runtime_error("Unsupported pointer dtype");
    }
}

cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Tensor A, int lda,
    at::Tensor B, int ldb,
    const float* beta,
    at::Tensor C, int ldc) 
    {
      const void *d_a = static_cast<const void *>(A.data_ptr());
      const void *d_b = static_cast<const void *>(B.data_ptr());
      void       *d_c = static_cast<void *>(C.data_ptr());
      hipDataType dtype_a = get_dtype(A, false);
      hipDataType dtype_b = get_dtype(B, false);
      hipDataType dtype_c = get_dtype(C, false);

      return hipblasGemmEx(
          handle,
          transa,
          transb,
          m,
          n,
          k,
          alpha,
          d_a,
          dtype_a,
          lda,
          d_b,
          dtype_b,
          ldb,
          beta,
          d_c,
          dtype_c,
          ldc,
          (dtype_a == HIP_R_64F) ? CUBLAS_COMPUTE_64F : CUBLAS_COMPUTE_32F,
          (dtype_a == HIP_R_16F || dtype_a == HIP_R_16BF) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);
}

#ifdef __HIP_PLATFORM_HCC__
int gemm_bias_lt(
    hipblasLtHandle_t ltHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Tensor A, int lda,
    at::Tensor B, int ldb,
    const float *beta, /* host pointer */
    at::Tensor C, int ldc,
    void *workspace,
    size_t workspaceSize,
    hipStream_t stream,
    bool use_bias,
    at::Tensor bias) {
  hipDataType dtype_a = get_dtype(A, true);
  hipDataType dtype_b = get_dtype(B, true);
  hipDataType dtype_c = get_dtype(C, true);

  hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Adesc,
                                                    dtype_a,
                                                    transa == HIPBLAS_OP_N ? m : k,
                                                    transa == HIPBLAS_OP_N ? k : m,
                                                    lda));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Bdesc,
                                                    dtype_b,
                                                    transb == HIPBLAS_OP_N ? k : n,
                                                    transb == HIPBLAS_OP_N ? n : k,
                                                    ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Cdesc,
                                                    dtype_c,
                                                    m,
                                                    n,
                                                    ldc));

  hipblasLtMatmulDesc_t operationDesc = nullptr;
  hipblasComputeType_t desc_computeType = HIPBLAS_COMPUTE_32F;
  hipDataType desc_dataType = HIPBLAS_R_32F;

  if (A.scalar_type() == at::ScalarType::Double)
  {
    desc_computeType = HIPBLAS_COMPUTE_64F;
    desc_dataType = HIPBLAS_R_64F;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&operationDesc,
                                                  desc_computeType,
                                                  desc_dataType));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa,
                                                        sizeof(transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb,
                                                        sizeof(transb)));

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  hipDataType dtype_bias = get_dtype(bias, true);

  auto d_bias = static_cast<void *>(bias.data_ptr());

  if (use_bias) {
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          &d_bias,
                                                          sizeof(d_bias)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, 
                                                          &dtype_bias, 
                                                          sizeof(dtype_bias)));
    epilogue = HIPBLASLT_EPILOGUE_BIAS;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));

  hipblasLtMatmulPreference_t preference;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize,
                                                              sizeof(workspaceSize)));

  int returnedResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(ltHandle,
                                                        operationDesc,
                                                        Adesc,
                                                        Bdesc,
                                                        Cdesc,
                                                        Cdesc,
                                                        preference,
                                                        1,
                                                        &heuristicResult,
                                                        &returnedResults));

  if (returnedResults == 0) {
    std::cerr << "No Solution found! request 1 at least" << std::endl;
    return 1;
  }

  const void *d_a = static_cast<const void *>(A.data_ptr());
  const void *d_b = static_cast<const void *>(B.data_ptr());
  void       *d_c = static_cast<void *>(C.data_ptr());

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(ltHandle,
                                        operationDesc,
                                        alpha,
                                        d_a,
                                        Adesc,
                                        d_b,
                                        Bdesc,
                                        beta,
                                        static_cast<const void *>(d_c),
                                        Cdesc,
                                        d_c,
                                        Cdesc,
                                        &heuristicResult.algo,
                                        workspace,
                                        workspaceSize,
                                        stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Adesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Bdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Cdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(operationDesc));
  return 0;
}

int gemm_bias_gelu_lt(
    hipblasLtHandle_t ltHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Tensor A, int lda,
    at::Tensor B, int ldb,
    const float *beta, /* host pointer */
    at::Tensor C, int ldc,
    void *workspace,
    size_t workspaceSize,
    hipStream_t stream,
    bool use_bias,
    at::Tensor gelu_in,
    at::Tensor bias){
  hipDataType dtype_a = get_dtype(A, true);
  hipDataType dtype_b = get_dtype(B, true);
  hipDataType dtype_c = get_dtype(C, true);

  hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Adesc,
                                                    dtype_a,
                                                    transa == HIPBLAS_OP_N ? m : k,
                                                    transa == HIPBLAS_OP_N ? k : m,
                                                    lda));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Bdesc,
                                                    dtype_b,
                                                    transb == HIPBLAS_OP_N ? k : n,
                                                    transb == HIPBLAS_OP_N ? n : k,
                                                    ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Cdesc,
                                                    dtype_c,
                                                    m,
                                                    n,
                                                    ldc));

  hipblasLtMatmulDesc_t operationDesc = nullptr;
  hipblasComputeType_t desc_computeType = HIPBLAS_COMPUTE_32F;
  hipDataType desc_dataType = HIPBLAS_R_32F;

  if (A.scalar_type() == at::ScalarType::Double)
  {
    desc_computeType = HIPBLAS_COMPUTE_64F;
    desc_dataType = HIPBLAS_R_64F;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&operationDesc,
                                                  desc_computeType,
                                                  desc_dataType));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa,
                                                        sizeof(transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb,
                                                        sizeof(transb)));

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_GELU_AUX;

  hipDataType dtype_bias = get_dtype(bias, true);

  auto d_bias = static_cast<void *>(bias.data_ptr());
  auto d_gelu_in = static_cast<void *>(gelu_in.data_ptr());
  int64_t ld_gelu = static_cast<int64_t>(gelu_in.stride(0));

  if (use_bias) {
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          &d_bias,
                                                          sizeof(d_bias)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                          &dtype_bias,
                                                          sizeof(dtype_bias)));
    epilogue = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                        &d_gelu_in,
                                                        sizeof(d_gelu_in)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                                        &ld_gelu,
                                                        sizeof(ld_gelu)));


  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));

  hipblasLtMatmulPreference_t preference;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize,
                                                              sizeof(workspaceSize)));

  int returnedResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(ltHandle,
                                                        operationDesc,
                                                        Adesc,
                                                        Bdesc,
                                                        Cdesc,
                                                        Cdesc,
                                                        preference,
                                                        1,
                                                        &heuristicResult,
                                                        &returnedResults));

  if (returnedResults == 0) {
    std::cerr << "No Solution found! request 1 at least" << std::endl;
    return 1;
  }

  const void *d_a = static_cast<const void *>(A.data_ptr());
  const void *d_b = static_cast<const void *>(B.data_ptr());
  void       *d_c = static_cast<void *>(C.data_ptr());

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(ltHandle,
                                        operationDesc,
                                        alpha,
                                        d_a,
                                        Adesc,
                                        d_b,
                                        Bdesc,
                                        beta,
                                        static_cast<const void *>(d_c),
                                        Cdesc,
                                        d_c,
                                        Cdesc,
                                        &heuristicResult.algo,
                                        workspace,
                                        workspaceSize,
                                        stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Adesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Bdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Cdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(operationDesc));
  return 0;
}

int gemm_dgelu_bgradb_lt(
    hipblasLtHandle_t ltHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Tensor A, int lda,
    at::Tensor B, int ldb,
    const float *beta, /* host pointer */
    at::Tensor C, int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    at::Tensor gelu_in,
    at::Tensor bgrad){
  hipDataType dtype_a = get_dtype(A, true);
  hipDataType dtype_b = get_dtype(B, true);
  hipDataType dtype_c = get_dtype(C, true);

  hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Adesc,
                                                    dtype_a,
                                                    transa == HIPBLAS_OP_N ? m : k,
                                                    transa == HIPBLAS_OP_N ? k : m,
                                                    lda));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Bdesc,
                                                    dtype_b,
                                                    transb == HIPBLAS_OP_N ? k : n,
                                                    transb == HIPBLAS_OP_N ? n : k,
                                                    ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Cdesc,
                                                    dtype_c,
                                                    m,
                                                    n,
                                                    ldc));

  hipblasLtMatmulDesc_t operationDesc = nullptr;
  hipblasComputeType_t desc_computeType = HIPBLAS_COMPUTE_32F;
  hipDataType desc_dataType = HIPBLAS_R_32F;

  if (A.scalar_type() == at::ScalarType::Double)
  {
    desc_computeType = HIPBLAS_COMPUTE_64F;
    desc_dataType = HIPBLAS_R_64F;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&operationDesc,
                                                  desc_computeType,
                                                  desc_dataType));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa,
                                                        sizeof(transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb,
                                                        sizeof(transb)));

  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DGELU_BGRAD;

  hipDataType dtype_bgrad = get_dtype(bgrad, true);

  auto d_bgrad = static_cast<void *>(bgrad.data_ptr());
  auto d_gelu_in = static_cast<void *>(gelu_in.data_ptr());
  int64_t ld_gelu = static_cast<int64_t>(gelu_in.stride(0));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
                                                        &d_bgrad, 
                                                        sizeof(d_bgrad)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                        HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, 
                                                        &dtype_bgrad, 
                                                        sizeof(dtype_bgrad)));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, 
                                                        &d_gelu_in, 
                                                        sizeof(d_gelu_in)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, 
                                                        &ld_gelu, 
                                                        sizeof(ld_gelu)));


  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));

  hipblasLtMatmulPreference_t preference;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize,
                                                              sizeof(workspaceSize)));

  int returnedResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(ltHandle,
                                                        operationDesc,
                                                        Adesc,
                                                        Bdesc,
                                                        Cdesc,
                                                        Cdesc,
                                                        preference,
                                                        1,
                                                        &heuristicResult,
                                                        &returnedResults));

  if (returnedResults == 0) {
    std::cerr << "No Solution found! request 1 at least" << std::endl;
    return 1;
  }

  const void *d_a = static_cast<const void *>(A.data_ptr());
  const void *d_b = static_cast<const void *>(B.data_ptr());
  void       *d_c = static_cast<void *>(C.data_ptr());

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(ltHandle,
                                        operationDesc,
                                        alpha,
                                        d_a,
                                        Adesc,
                                        d_b,
                                        Bdesc,
                                        beta,
                                        static_cast<const void *>(d_c),
                                        Cdesc,
                                        d_c,
                                        Cdesc,
                                        &heuristicResult.algo,
                                        workspace,
                                        workspaceSize,
                                        stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Adesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Bdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Cdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(operationDesc));
  return 0;
}

int gemm_bgradb_lt(
    hipblasLtHandle_t ltHandle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha, /* host pointer */
    at::Tensor A, int lda,
    at::Tensor B, int ldb,
    const float *beta, /* host pointer */
    at::Tensor C, int ldc,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream,
    bool use_bias,
    at::Tensor bgrad) {
  hipDataType dtype_a = get_dtype(A, true);
  hipDataType dtype_b = get_dtype(B, true);
  hipDataType dtype_c = get_dtype(C, true);

  hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Adesc,
                                                    dtype_a,
                                                    transa == HIPBLAS_OP_N ? m : k,
                                                    transa == HIPBLAS_OP_N ? k : m,
                                                    lda));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Bdesc,
                                                    dtype_b,
                                                    transb == HIPBLAS_OP_N ? k : n,
                                                    transb == HIPBLAS_OP_N ? n : k,
                                                    ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&Cdesc,
                                                    dtype_c,
                                                    m,
                                                    n,
                                                    ldc));

  hipblasLtMatmulDesc_t operationDesc = nullptr;
  hipblasComputeType_t desc_computeType = HIPBLAS_COMPUTE_32F;
  hipDataType desc_dataType = HIPBLAS_R_32F;

  if (A.scalar_type() == at::ScalarType::Double)
  {
    desc_computeType = HIPBLAS_COMPUTE_64F;
    desc_dataType = HIPBLAS_R_64F;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&operationDesc,
                                                  desc_computeType,
                                                  desc_dataType));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSA,
                                                        &transa,
                                                        sizeof(transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_TRANSB,
                                                        &transb,
                                                        sizeof(transb)));
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

  hipDataType dtype_bgrad = get_dtype(bgrad, true);

  auto d_bgrad = static_cast<void *>(bgrad.data_ptr());

  if (use_bias) {
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                          HIPBLASLT_MATMUL_DESC_BIAS_POINTER, 
                                                          &d_bgrad, 
                                                          sizeof(d_bgrad)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc, 
                                                          HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, 
                                                          &dtype_bgrad, 
                                                          sizeof(dtype_bgrad)));
    epilogue = HIPBLASLT_EPILOGUE_BGRADB;
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(operationDesc,
                                                        HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                        &epilogue,
                                                        sizeof(epilogue)));

  hipblasLtMatmulPreference_t preference;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspaceSize,
                                                              sizeof(workspaceSize)));

  int returnedResults = 0;
  hipblasLtMatmulHeuristicResult_t heuristicResult;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(ltHandle,
                                                        operationDesc,
                                                        Adesc,
                                                        Bdesc,
                                                        Cdesc,
                                                        Cdesc,
                                                        preference,
                                                        1,
                                                        &heuristicResult,
                                                        &returnedResults));

  if (returnedResults == 0) {
    std::cerr << "No Solution found! request 1 at least" << std::endl;
    return 1;
  }

  const void *d_a = static_cast<const void *>(A.data_ptr());
  const void *d_b = static_cast<const void *>(B.data_ptr());
  void       *d_c = static_cast<void *>(C.data_ptr());

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(ltHandle,
                                        operationDesc,
                                        alpha,
                                        d_a,
                                        Adesc,
                                        d_b,
                                        Bdesc,
                                        beta,
                                        static_cast<const void *>(d_c),
                                        Cdesc,
                                        d_c,
                                        Cdesc,
                                        &heuristicResult.algo,
                                        workspace,
                                        workspaceSize,
                                        stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Adesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Bdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(Cdesc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(operationDesc));
  return 0;
}
#endif

int linear_bias_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, int in_features, int batch_size, int out_features, at::Tensor output, void *lt_workspace) {
  int device_id;
  CHECK_HIP_ERROR(hipGetDevice(&device_id));

  hipblasLtHandle_t handle = cached_handles.get(device_id);
  if (handle == nullptr)
    handle = cached_handles.obtain(device_id);
  
  hipStream_t stream;
  CHECK_HIP_ERROR(hipblasGetStream(handle, &stream));

  const float alpha          = 1.0;
  const float beta_zero       = 0.0;
  const float beta_one       = 1.0;
  int status = 1;
  size_t workspace_size = 1 << 22;
#ifdef __HIP_PLATFORM_HCC__
  status = gemm_bias_lt(handle,
                   HIPBLAS_OP_T,
                   HIPBLAS_OP_N,
                   out_features,
                   batch_size,
                   in_features,
                   &alpha,
                   weight, in_features,
                   input, in_features,
                   &beta_zero,
                   output, out_features,
                   lt_workspace,
                   workspace_size,
                   stream,
                   true,
                   bias);
#endif
  if (status != 0){
    std::cerr << "gemm_bias_lt error" << std::endl;
    return status;
  }
  return status;
}


int linear_bias_backward_cuda(at::Tensor input, at::Tensor weight, at::Tensor d_output, int in_features, int batch_size, int out_features, at::Tensor d_weight, at::Tensor d_bias, at::Tensor d_input,  void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
#ifdef __HIP_PLATFORM_HCC__
    // d_bias is reduced separately in fused_dense.cpp on ROCm.  Using the
    // BGRADB hipBLASLt epilogue here is therefore redundant and is not
    // supported for all shapes by every hipBLASLt implementation.
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      in_features,
      out_features,
      batch_size,
      &alpha,
      input, in_features,
      d_output, out_features,
      &beta_zero,
      d_weight, in_features);
    if (status == 0) {
      status = gemm_bias(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        in_features,
        batch_size,
        out_features,
        &alpha,
        weight, in_features,
        d_output, out_features,
        &beta_zero,
        d_input, in_features);
    }
#endif
  if (status != 0){
    std::cerr << "linear_bias_backward GEMM error" << std::endl;
    return status;
  }
  return status;
}

int linear_gelu_linear_forward_cuda(at::Tensor input, at::Tensor weight1, at::Tensor bias1, at::Tensor weight2, at::Tensor bias2, int in_features, int hidden_features, int batch_size, int out_features, at::Tensor output1, at::Tensor output2, at::Tensor gelu_in, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    int status = 1;
#ifdef __HIP_PLATFORM_HCC__
    status = gemm_bias_gelu_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    hidden_features, 
    batch_size, 
    in_features,
    &alpha, /* host pointer */
    weight1, in_features,
    input, in_features,
    &beta_zero, /* host pointer */
    output1, hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    gelu_in,
    bias1);
    status = gemm_bias_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_T,
    CUBLAS_OP_N,
    out_features,
    batch_size,
    hidden_features,
    &alpha, /* host pointer */
    weight2, hidden_features,
    output1, hidden_features,
    &beta_zero, /* host pointer */
    output2, out_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    bias2);
    return status;
#else 
    return 1;
#endif
}

int linear_gelu_linear_backward_cuda(at::Tensor input, at::Tensor gelu_in, at::Tensor output1, 
                                     at::Tensor weight1, at::Tensor weight2, at::Tensor d_output1, 
                                     at::Tensor d_output2, int in_features, int batch_size, int hidden_features, int out_features, 
                                     at::Tensor d_weight1, at::Tensor d_weight2, at::Tensor d_bias1, at::Tensor d_bias2, at::Tensor d_input, void *lt_workspace) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Get the stream from cublas handle to reuse for biasReLU kernel.
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha          = 1.0;
    const float beta_zero       = 0.0;
    const float beta_one       = 1.0;
    int status = 1;
#ifdef __HIP_PLATFORM_HCC__
//wgrad for first gemm
    status = gemm_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    hidden_features,
    out_features,
    batch_size,
    &alpha, /* host pointer */
    output1, hidden_features,
    d_output2, out_features,
    &beta_zero, /* host pointer */
    d_weight2, hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    true,
    d_bias2);
//dgrad for second GEMM
    status = gemm_dgelu_bgradb_lt(
    (cublasLtHandle_t)handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    hidden_features,
    batch_size,
    out_features,
    &alpha, /* host pointer */
    weight2, hidden_features,
    d_output2, out_features,
    &beta_zero, /* host pointer */
    d_output1, hidden_features,
    lt_workspace,
    1 << 22,
    stream,
    gelu_in,
    d_bias1);
//wgrad for the first GEMM
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_T,
      in_features,
      hidden_features,
      batch_size,
      &alpha,
      input, in_features,
      d_output1, hidden_features,
      &beta_zero,
      d_weight1, in_features);

//dgrad for the first GEMM
    status = gemm_bias(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      in_features,
      batch_size,
      hidden_features,
      &alpha,
      weight1, in_features,
      d_output1, hidden_features,
      &beta_zero,
      d_input, in_features);
#endif
    return status;

}
