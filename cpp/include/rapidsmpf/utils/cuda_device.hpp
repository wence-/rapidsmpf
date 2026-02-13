/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cuda/devices>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf::detail {

/**
 * @brief @return Obtain a reference to current CUDA Device.
 */
inline cuda::device_ref current_device() {
    int dev;
    RAPIDSMPF_CUDA_TRY(cudaGetDevice(&dev));
    return cuda::device_ref{dev};
}
}  // namespace rapidsmpf::detail
