// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if defined(USE_ROCM) && __has_include(<hipfile.h>)
#include <hipfile.h>
#define APEX_USE_HIPFILE 1
#elif !defined(USE_ROCM)
#include <cufile.h>
#define APEX_USE_CUFILE 1
#endif
#include <torch/torch.h>

#include <string>

namespace apex::contrib::gds {
class File {
 public:
  File();
  File(const std::string& filename, const std::string& mode);
  ~File();

  void open(const std::string& filename, const std::string& mode);
  void close();

  void load_data(const torch::Tensor& tensor);
  void save_data(const torch::Tensor& tensor);
  void load_data_no_gds(const torch::Tensor& tensor);
  void save_data_no_gds(const torch::Tensor& tensor);

 private:
  std::string filename;
  std::string mode;

#if defined(APEX_USE_HIPFILE)
  HIPfileDescr_t cf_descr;
  HIPfileHandle_t cf_handle;
  HIPfileError_t status;
#elif defined(APEX_USE_CUFILE)
  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;
  CUfileError_t status;
#endif

  int fd = -1;
  bool is_open = false;
  bool maybe_register = true;
};
}  // namespace apex::contrib::gds
