/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/event>

#include <cudf/table/table_view.hpp>

#include <rapidsmpf/streaming/cudf/broadcast.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming {
coro::task<streaming::Message> broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    OpID tag,
    streaming::AllGather::Ordered ordered
) {
    ShutdownAtExit c{ch_in};
    co_await ctx->executor()->schedule();
    cuda::event event{ctx->br()->device()};
    if (ctx->comm()->nranks() == 1) {
        std::vector<TableChunk> chunks;
        std::vector<cudf::table_view> views;
        auto gather_stream = ctx->br()->stream_pool().get_stream();
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            auto chunk =
                co_await msg.release<streaming::TableChunk>().make_available(ctx);
            cuda_stream_join(gather_stream, chunk.stream(), &event);
            views.push_back(chunk.table_view());
            chunks.push_back(std::move(chunk));
        }
        if (chunks.size() == 1) {
            co_return streaming::to_message(
                0, std::make_unique<streaming::TableChunk>(std::move(chunks[0]))
            );
        } else {
            RAPIDSMPF_EXPECTS(chunks.size() > 0, "No chunks in broadcast");
            auto result = cudf::concatenate(views, gather_stream, ctx->br()->device_mr());
            // So that deallocation of the consitutent tables is stream-ordered wrt the
            // concatenation.
            cuda_stream_join(
                chunks
                    | std::views::transform([](auto&& chunk) { return chunk.stream(); }),
                std::ranges::single_view(gather_stream),
                &event
            );
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(std::move(result), gather_stream)
            );
        }
    } else {
        streaming::AllGather gatherer{ctx, tag};
        while (true) {
            auto msg = co_await ch_in->receive();
            if (msg.empty()) {
                break;
            }
            // TODO: If this chunk is already in pack form, this is unnecessary.
            auto chunk =
                co_await msg.release<streaming::TableChunk>().make_available(ctx);
            auto pack =
                cudf::pack(chunk.table_view(), chunk.stream(), ctx->br()->device_mr());
            auto packed_data = PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), chunk.stream())
            );
            gatherer.insert(msg.sequence_number(), {std::move(packed_data)});
        }
        gatherer.insert_finished();
        auto result = co_await gatherer.extract_all(ordered);
        if (result.size() == 1) {
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    std::make_unique<PackedData>(std::move(result[0]))
                )
            );
        } else {
            auto stream = ctx->br()->stream_pool().get_stream();
            co_return streaming::to_message(
                0,
                std::make_unique<streaming::TableChunk>(
                    unpack_and_concat(
                        unspill_partitions(
                            std::move(result),
                            ctx->br().get(),
                            AllowOverbooking::YES,
                            ctx->statistics()
                        ),
                        stream,
                        ctx->br().get(),
                        ctx->statistics()
                    ),
                    stream
                )
            );
        }
    }
}
}  // namespace rapidsmpf::streaming
