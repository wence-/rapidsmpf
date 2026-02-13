/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/integrations/cudf/bloom_filter.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/message.hpp>
#include <rapidsmpf/streaming/cudf/bloom_filter.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "rapidsmpf/memory/buffer_resource.hpp"
#include "rapidsmpf/streaming/core/channel.hpp"
#include "rapidsmpf/streaming/core/context.hpp"
#include "rapidsmpf/streaming/core/node.hpp"

namespace rapidsmpf::streaming {

Node approx_distinct_count(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::int32_t precision,
    cudf::null_policy null_handling,
    cudf::nan_policy nan_handling,
    OpID tag
) {
    ShutdownAtExit c{ch_in, ch_out};

    co_await ctx->executor()->schedule();
    auto stream = ctx->br()->stream_pool().get_stream();
    auto counter = cudf::approx_distinct_count(
        cudf::table_view(), precision, null_handling, nan_handling, stream
    );
    CudaEvent event;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        auto chunk = co_await msg.release<TableChunk>().make_available(ctx, 0);
        cuda_stream_join(stream, chunk.stream(), &event);
        counter.add(chunk.table_view(), stream);
        cuda_stream_join(chunk.stream(), stream, &event);
    }
    if (ctx->comm()->nranks() > 1) {
        auto metadata = std::make_unique<std::vector<std::uint8_t>>(1);
        auto sketch = counter.sketch();
        auto [res, _] = ctx->br()->reserve(
            MemoryType::DEVICE, sketch.size_bytes(), AllowOverbooking::YES
        );
        auto buf = ctx->br()->allocate(stream, std::move(res));
        buf->write_access([&](std::byte* data, rmm::cuda_stream_view stream) {
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                data, sketch.data(), sketch.size_bytes(), cudaMemcpyDefault, stream
            ));
        });
        // TODO: allreduce with preallocated buffers
        auto allgather = streaming::AllGather(ctx, tag);
        allgather.insert(0, {std::move(metadata), std::move(buf)});
        allgather.insert_finished();
        auto per_rank = co_await allgather.extract_all(streaming::AllGather::Ordered::NO);
        auto other = cudf::approx_distinct_count(
            cudf::table_view(), precision, null_handling, nan_handling, stream
        );
        auto sketch_other = other.sketch();
        for (auto&& data : per_rank) {
            cuda_stream_join(data.data->stream(), stream, &event);
            RAPIDSMPF_CUDA_TRY(cudaMemcpyAsync(
                sketch_other.data(),
                data.data->data(),
                sketch_other.size_bytes(),
                cudaMemcpyDefault,
                stream
            ));
            cuda_stream_join(stream, data.data->stream(), &event);
            counter.merge(other, stream);
        }
    }
    co_await ch_out->send(
        Message{0, std::make_unique<std::size_t>(counter.estimate(stream)), {}, {}}
    );
    co_await ch_out->drain(ctx->executor());
}


  std::optional<std::size_t> estimate_cardinality(std::size_t S, std::size_t n) {
    // Want to invert for M in S = M (1 - (1 - 1/M)^n)
    double observed_fraction = static_cast<double>(S) / static_cast<double>(n);
    if (S >  ) {
    }
  }
Node estimate_cardinality(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_in,
    std::shared_ptr<Channel> ch_out,
    std::int32_t precision,
    cudf::null_policy null_handling,
    cudf::nan_policy nan_handling,
    OpID tag
) {
    
}

}  // namespace rapidsmpf::streaming
