// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <gds.h>

// torch
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

// cuda
#include <cuda_runtime.h>
#if defined(APEX_USE_HIPFILE)
#include <hipfile.h>
#elif defined(APEX_USE_CUFILE)
#include <cufile.h>
#endif

// file io
#include <fcntl.h>

namespace apex::contrib::gds {

#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
// POSIX
template <class T, typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status)) : std::string(std::strerror(errno));
}

// CUfileError_t
template <class T, typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
std::string cuFileGetErrorString(T status) {
  std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status)) errStr.append(".").append(cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}
#endif

File::File() : is_open(false) {};

File::File(const std::string& filename, const std::string& mode) : filename(filename), mode(mode), is_open(false) {
  open(filename, mode);
}

File::~File() {
  if (is_open) {
    close();
  }
}

void File::open(const std::string& other_filename, const std::string& other_mode) {
  TORCH_CHECK(is_open == false, "file", filename, "is already open");
  if (!filename.empty()) {
    TORCH_CHECK(other_filename == filename, "file", filename, "is already open with mode", mode);
  }
  if (!mode.empty()) {
    TORCH_CHECK(other_mode == mode, "file", filename, "is already open with mode", mode);
  }

  maybe_register =
#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
      true;
#else
      false;
#endif
  // Open the binary file
  if (mode == "r") {
    // for reading
    fd = ::open(filename.c_str(), O_RDONLY | (maybe_register ? O_DIRECT : 0));
  } else if (mode == "w") {
    // for writing
    fd = ::open(filename.c_str(), O_CREAT | O_WRONLY | (maybe_register ? O_DIRECT : 0), 0664);
  } else if (mode == "rn") {
    // for reading
    fd = ::open(filename.c_str(), O_RDONLY);
    maybe_register = false;
  } else if (mode == "wn") {
    // for writing
    fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    maybe_register = false;
  } else {
    TORCH_CHECK(false, "only r and w modes are currently supported, but got:", mode);
  }
  TORCH_CHECK(fd >= 0, "fcntl cannot open file: ", filename);

  // Register cuFile handle
#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
  if (maybe_register) {
    memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
      TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuFileGetErrorString(status));
    }
  }
#endif
  is_open = true;
}

void File::close() {
  // Deregister cuFile handle and close the file
  if (is_open) {
#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
    if (maybe_register) {
      cuFileHandleDeregister(cf_handle);
    }
#endif
    ::close(fd);
    fd = -1;
  }
  is_open = false;
}

void File::load_data(const torch::Tensor& tensor) {
  TORCH_CHECK(mode == "r", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Read the binary file
#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, 0, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuFileGetErrorString(ret));
#else
  void* dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  const ssize_t nbytes_read = pread(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_read == nbytes || nbytes_read == 0, "fcntl pread failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCPU, nbytes, cudaMemcpyHostToDevice));
  free(dataPtrCPU);
#endif
}

void File::save_data(const torch::Tensor& tensor) {
  TORCH_CHECK(mode == "w", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Register device memory
#if defined(APEX_USE_HIPFILE) || defined(APEX_USE_CUFILE)
  status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed: ", cuFileGetErrorString(status));

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, 0, 0);
  status = cuFileBufDeregister(dataPtr);

  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuFileGetErrorString(ret));
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed:", cuFileGetErrorString(status));
#else
  void* dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtrCPU, dataPtr, nbytes, cudaMemcpyDeviceToHost));
  const ssize_t nbytes_written = pwrite(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_written == nbytes, "fcntl pwrite failed");
  free(dataPtrCPU);
#endif
}

// Just for benchmarking purposes

void File::load_data_no_gds(const torch::Tensor& tensor) {
  TORCH_CHECK(mode == "rn", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();
  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");

  const ssize_t nbytes_read = pread(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_read == nbytes || nbytes_read == 0, "fcntl pread failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCPU, nbytes, cudaMemcpyHostToDevice));
  free(dataPtrCPU);
}

void File::save_data_no_gds(const torch::Tensor& tensor) {
  TORCH_CHECK(mode == "wn", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCPU = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();
  dataPtrCPU = malloc(nbytes);
  TORCH_CHECK(dataPtrCPU != nullptr, "malloc failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtrCPU, dataPtr, nbytes, cudaMemcpyDeviceToHost));

  const ssize_t nbytes_written = pwrite(fd, dataPtrCPU, nbytes, 0);
  TORCH_CHECK(nbytes_written == nbytes, "fcntl pwrite failed");
  free(dataPtrCPU);
}

}  // namespace apex::contrib::gds
