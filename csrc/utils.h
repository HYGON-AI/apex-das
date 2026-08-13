// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdlib>
#include <torch/extension.h>

#ifdef __HIP_PLATFORM_HCC__
#include <vector>
#include <forward_list>
#include <mutex>
#include <hipblaslt/hipblaslt.h>

#ifndef CHECK_ERROR
#define CHECK_ERROR(status)                \
    if(status != 0)                        \
    {                                      \
        fprintf(stderr, "EXIT_FAILURE");   \
        fprintf(stderr, "\n");             \
        exit(EXIT_FAILURE);                \
    }
#endif

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: Err=(%d) at %s:%d\n", \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

// static class WorkspacePool {
// public:
//   WorkspacePool() {
//     CHECK_HIP_ERROR(hipMalloc(&workspace, workspaceSize));
//   }
//   ~WorkspacePool() {
//     CHECK_HIP_ERROR(hipFree(workspace));
//   }

//   void* get() {
//     std::lock_guard<std::mutex> lock(mt);
//     return workspace;
//   }
// private:
//   std::mutex mt;
//   void* workspace = nullptr;
//   static size_t workspaceSize;
// } workspace_pool;

// size_t WorkspacePool::workspaceSize = 256 * 1024 * 1024;

static class HandlePool {
public:
  hipblasLtHandle_t get(int device_id) 
  {
    std::lock_guard<std::mutex> lock(mt);

    if (pool.empty())
    {
      int device_count = 0; 
      CHECK_HIP_ERROR(hipGetDeviceCount(&device_count));
      pool.resize(device_count);
      return nullptr;
    }

    if (!pool[device_id].empty())
    {
      hipblasLtHandle_t h = pool[device_id].front();
      pool[device_id].pop_front();
      return h;
    }

    return nullptr;
  }

  hipblasLtHandle_t obtain(int device_id) 
  {
    hipblasLtHandle_t h = get(device_id);
    if (h == nullptr)
    {
      CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&h));
    }
    return h;
  }

  void store(const std::vector<hipblasLtHandle_t>& handles)
  {
    std::lock_guard<std::mutex> lock(mt);
    if (pool.empty())
    {
      std::cout << "[ERROR] Attempt to store handles to invalid pool" << std::endl;
    }
    for (unsigned int i=0; i<pool.size(); i++)
    {
      if (handles[i] != nullptr)
      {
        pool[i].push_front(handles[i]);
      }
    }
  }

  ~HandlePool() {
#if DESTROY_HIPBLASLT_HANDLES_POOL
    std::lock_guard<std::mutex> lock(mt);
    for (auto & hlist : pool)
    {
      for (auto & h : hlist)
      {
        CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(h));
      }
    }
    pool.clear();
#endif
  }

  inline size_t get_size() const
  {
    return pool.size();
  }

private:
  std::mutex mt;
  using Pool = std::vector<std::forward_list<hipblasLtHandle_t>>;
#if DESTROY_HIPBLASLT_HANDLES_POOL
  Pool pool;
#else
  Pool &pool = *new Pool();
#endif
} handle_pool;


thread_local static class HandleCache {
public:
  hipblasLtHandle_t get(int device_id) const
  {
    return d.empty() ? nullptr : d[device_id];
  }

  hipblasLtHandle_t obtain(int device_id)
  {
    hipblasLtHandle_t h = get(device_id);
    if (h)
    {
      return h;
    }
    h = handle_pool.obtain(device_id);
    set(device_id, h);
    return h;
  }

  void set(int device_id, hipblasLtHandle_t h) 
  { 
    if (d.empty())
    {
      d.resize(handle_pool.get_size());
    }
    d[device_id] = h;
  }

  ~HandleCache()
  {
    if (!d.empty())
    {
      handle_pool.store(d);
    }
  }

private:
  std::vector<hipblasLtHandle_t> d;
} cached_handles;
#endif // __HIP_PLATFORM_HCC__

inline bool parseEnvVarFlag(const char* envVarName) {
  char* stringValue = std::getenv(envVarName);
  if (stringValue != nullptr) {
    int val;
    try {
      val = std::stoi(stringValue);
    } catch (std::exception& e) {
      TORCH_CHECK(false,
          "Invalid value for environment variable: " + std::string(envVarName));
    }
    if (val == 1) {
      return true;
    } else if (val == 0) {
      return false;
    } else {
      TORCH_CHECK(false,
          "Invalid value for environment variable: " + std::string(envVarName));
    }
  }
  return false;
}
