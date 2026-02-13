#pragma once

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <coro/coro.hpp>

namespace rapidsmpf::streaming {

/**
 * @brief Broadcast the concatenation of all input messages to all ranks.
 *
 * @note Receives all input chunks, gathers from all ranks, and then provides concatenated
 * output.
 *
 * @param ctx Streaming context
 * @param ch_in Input channel of `TableChunk`s
 * @param tag Disambiguating tag for allgather
 * @param ordered Should the concatenated output be ordered
 *
 * @return Message containing the concatenation of all the input table chunks.
 */
[[nodiscard]] coro::task<Message> broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    OpID tag,
    streaming::AllGather::Ordered ordered
);

}  // namespace rapidsmpf::streaming
