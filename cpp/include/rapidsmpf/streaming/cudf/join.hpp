#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#pragma once

namespace rapidsmpf::streaming::join {

/// @brief Treatment of keys in the result of a join
enum class KeepKeys : bool {
    NO,  ///< Key columns do not appear in the output
    YES,  ///< Key columns do appear in the output
};

/// @brief Which input is the broadcasted side for a join.
enum class BroadcastSide : bool {
    LEFT,  ///< Broadcast the left input
    RIGHT,  ///< Broadcast the right input
};

namespace detail {
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
 * @return Coroutine that produces a message containing the concatenation of all the input table chunks.
 */
[[nodiscard]] coro::task<streaming::Message> broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    OpID tag,
    streaming::AllGather::Ordered ordered = streaming::AllGather::Ordered::YES
);

/**
 * @brief Broadcast the concatenation of all input messages to all ranks.
 *
 * @note Receives all input chunks, gathers from all ranks, and then provides concatenated
 * output.
 *
 * @param ctx Streaming context
 * @param ch_in Input channel of `TableChunk`s
 * @param ch_out Input channel of a single `TableChunk`
 * @param tag Disambiguating tag for allgather
 * @param ordered Should the concatenated output be ordered
 *
 * @return Coroutine representing the broadcast
 */
[[nodiscard]] streaming::Node broadcast(
    std::shared_ptr<streaming::Context> ctx,
    std::shared_ptr<streaming::Channel> ch_in,
    std::shared_ptr<streaming::Channel> ch_out,
    OpID tag,
    streaming::AllGather::Ordered ordered = streaming::AllGather::Ordered::YES
);
}  // namespace detail

}  // namespace rapidsmpf::streaming::join
