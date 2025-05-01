/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once


#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/partition.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing and retrieving data chunks by partition and
 * chunk ID.
 * @tparam KeyType The type of keys identifying chunks in the postbox.
 */
template <typename KeyType>
class PostBox {
  public:
    using key_type = KeyType;  ///< The type of keys identifying chunks in the postbox.

    /**
     * @brief Constructor for the postbox.
     *
     * @tparam F Function type.
     * @param key_map Mapping function from `PartID` to `key_type`.
     */
    template <typename F>
    PostBox(F&& key_map) : key_map_{std::move(key_map)} {}

    /**
     * @brief Inserts a chunk into the PostBox.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk) {
        std::lock_guard const lock(mutex_);
        auto [_, inserted] =
            pigeonhole_[key_map_(chunk.pid)].insert({chunk.cid, std::move(chunk)});
        RAPIDSMPF_EXPECTS(inserted, "PostBox.insert(): chunk already exist");
    }

    /**
     * @brief Extracts a specific chunk from the PostBox.
     *
     * @param pid The ID of the partition containing the chunk.
     * @param cid The ID of the chunk to be accessed.
     * @return The extracted chunk.
     *
     * @throws std::out_of_range If the chunk is not found.
     */
    [[nodiscard]] Chunk extract(key_type pid, ChunkID cid) {
        std::lock_guard const lock(mutex_);
        return extract_item(pigeonhole_.at(pid), cid).second;
    }

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @return A map of chunk IDs to chunks for the specified partition.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    std::unordered_map<ChunkID, Chunk> extract(key_type pid) {
        std::lock_guard const lock(mutex_);
        return extract_value(pigeonhole_, pid);
    }

    /**
     * @brief Extracts all chunks from the PostBox.
     *
     * @return A vector of all chunks in the PostBox.
     */
    std::vector<Chunk> extract_all() {
        std::lock_guard const lock(mutex_);
        std::vector<Chunk> ret;
        for (auto& [_, chunks] : pigeonhole_) {
            for (auto& [_, chunk] : chunks) {
                ret.push_back(std::move(chunk));
            }
        }
        pigeonhole_.clear();
        return ret;
    }

    /**
     * @brief Checks if the PostBox is empty.
     *
     * @return `true` if the PostBox is empty, `false` otherwise.
     */
    [[nodiscard]] bool empty() const {
        return pigeonhole_.empty();
    }

    /**
     * @brief Searches for chunks of the specified memory type.
     *
     * @param mem_type The type of memory to search within.
     * @return A vector of tuples, where each tuple contains: PartID, ChunkID, and the
     * size of the chunk.
     */
    [[nodiscard]] std::vector<std::tuple<key_type, ChunkID, std::size_t>> search(
        MemoryType mem_type
    ) const {
        std::lock_guard const lock(mutex_);
        std::vector<std::tuple<KeyType, ChunkID, std::size_t>> ret;
        for (auto& [pid, chunks] : pigeonhole_) {
            for (auto& [cid, chunk] : chunks) {
                if (chunk.gpu_data && chunk.gpu_data->mem_type() == mem_type) {
                    ret.emplace_back(pid, cid, chunk.gpu_data->size);
                }
            }
        }
        return ret;
    }

    /**
     * @brief Returns a description of this instance.
     * @return The description.
     */
    [[nodiscard]] std::string str() const {
        if (empty()) {
            return "PostBox()";
        }
        std::stringstream ss;
        ss << "PostBox(";
        for (auto const& [pid, chunks] : pigeonhole_) {
            ss << "p" << pid << ": [";
            for (auto const& [cid, chunk] : chunks) {
                assert(cid == chunk.cid);
                if (chunk.expected_num_chunks) {
                    ss << "EOP" << chunk.expected_num_chunks << ", ";
                } else {
                    ss << cid << ", ";
                }
            }
            ss << "\b\b], ";
        }
        ss << "\b\b)";
        return ss.str();
    }

  private:
    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;
    std::unordered_map<key_type, std::unordered_map<ChunkID, Chunk>>
        pigeonhole_;  ///< Storage for chunks, organized by partition and chunk ID.
    std::function<key_type(PartID)>
        key_map_;  ///< Mapper from chunk partition ids to key_type.
};

/**
 * @brief Overloads the stream insertion operator for the PostBox class.
 *
 * This function allows a description of a PostBox to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, PostBox<T> const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
