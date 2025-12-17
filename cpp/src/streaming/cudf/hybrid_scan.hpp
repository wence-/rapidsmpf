#include <algorithm>
#include <ranges>
#include <vector>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/error.hpp>

#include "expression_converter.hpp"

namespace {

std::unique_ptr<cudf::io::datasource> make_datasource_from_source_info(
    cudf::io::source_info const& source_info,
    std::size_t offset = 0,
    std::size_t max_size_estimate = 0
) {
    switch (source_info.type()) {
    case cudf::io::io_type::FILEPATH:
        {
            return cudf::io::datasource::create(
                source_info.filepaths().front(), offset, max_size_estimate
            );
        }
    case cudf::io::io_type::HOST_BUFFER:
        return std::move(
            cudf::io::datasource::create(source_info.host_buffers()).front()
        );
    case cudf::io::io_type::DEVICE_BUFFER:
        return std::move(
            cudf::io::datasource::create(source_info.device_buffers()).front()
        );
    case cudf::io::io_type::USER_IMPLEMENTED:
        return std::move(
            cudf::io::datasource::create(source_info.user_sources()).front()
        );
    default:
        CUDF_FAIL("Unsupported source type");
    }
}

std::unique_ptr<cudf::io::datasource::buffer> fetch_footer_bytes(
    std::reference_wrapper<cudf::io::datasource> datasource_ref
) {
    CUDF_FUNC_RANGE();

    using namespace cudf::io::parquet;

    auto& datasource = datasource_ref.get();

    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len = sizeof(file_ender_s);
    std::size_t const len = datasource.size();

    auto const header_buffer = datasource.host_read(0, header_len);
    auto const ender_buffer = datasource.host_read(len - ender_len, ender_len);
    auto const header = reinterpret_cast<file_header_s const*>(header_buffer->data());
    auto const ender = reinterpret_cast<const file_ender_s*>(ender_buffer->data());
    RAPIDSMPF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    constexpr uint32_t parquet_magic =
        (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
    RAPIDSMPF_EXPECTS(
        header->magic == parquet_magic && ender->magic == parquet_magic,
        "Corrupted header or footer"
    );
    RAPIDSMPF_EXPECTS(
        ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
        "Incorrect footer length"
    );

    return datasource.host_read(len - ender->footer_len - ender_len, ender->footer_len);
}

/**
 * @brief Fetches a host span of Parquet PageIndexbytes from the input buffer span
 *
 * @param buffer Input buffer span
 * @param page_index_bytes Byte range of `PageIndex` to fetch
 * @return A host span of the PageIndex bytes
 */
std::unique_ptr<cudf::io::datasource::buffer> fetch_page_index_bytes(
    std::reference_wrapper<cudf::io::datasource> datasource_ref,
    cudf::io::text::byte_range_info const page_index_bytes
) {
    return datasource_ref.get().host_read(
        static_cast<std::size_t>(page_index_bytes.offset()),
        static_cast<std::size_t>(page_index_bytes.size())
    );
}

/**
 * @brief Fetches a list of byte ranges from a host buffer into a vector of device buffers
 *
 * @param host_buffer Host buffer span
 * @param byte_ranges Byte ranges to fetch
 * @param stream CUDA stream
 * @param mr Device memory resource to create device buffers with
 *
 * @return Vector of device buffers
 */
std::vector<rmm::device_buffer> fetch_byte_ranges(
    std::reference_wrapper<cudf::io::datasource> datasource_ref,
    cudf::host_span<cudf::io::text::byte_range_info const> byte_ranges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    static std::mutex mutex;

    CUDF_FUNC_RANGE();

    auto& datasource = datasource_ref.get();
    std::vector<std::future<std::size_t>> ioFutures{};
    ioFutures.reserve(byte_ranges.size());
    std::vector<rmm::device_buffer> buffers(byte_ranges.size());
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::ranges::for_each(
            std::views::iota(std::size_t{0}, byte_ranges.size()), [&](auto const idx) {
                const auto& byte_range = byte_ranges[idx];
                auto const byte_range_offset =
                    static_cast<std::size_t>(byte_range.offset());
                auto const byte_range_size = static_cast<std::size_t>(byte_range.size());

                auto& buffer = buffers[idx];

                // Pad the buffer size to be a multiple of 8 bytes
                constexpr std::size_t buffer_padding_multiple = 8;
                buffer = rmm::device_buffer(
                    cudf::util::round_up_safe<std::size_t>(
                        byte_range_size, buffer_padding_multiple
                    ),
                    stream,
                    mr
                );

                // Directly read the column chunk data to the device buffer if
                // supported
                if (datasource.supports_device_read()
                    and datasource.is_device_read_preferred(byte_range_size))
                {
                    ioFutures.emplace_back(datasource.device_read_async(
                        byte_range_offset,
                        byte_range_size,
                        static_cast<uint8_t*>(buffer.data()),
                        stream
                    ));
                } else {
                    // Read the column chunk data to the host buffer and copy it to
                    // the device buffer
                    auto hostBuffer =
                        datasource.host_read(byte_range_offset, byte_range_size);

                    cudf::detail::cuda_memcpy_async(
                        cudf::device_span<uint8_t>{
                            static_cast<uint8_t*>(buffer.data()), byte_range_size
                        },
                        cudf::host_span<uint8_t const>{
                            static_cast<uint8_t const*>(hostBuffer->data()),
                            byte_range_size
                        },
                        stream
                    );
                }
            }
        );
    }

    // Wait for all IO futures to complete
    std::ranges::for_each(ioFutures, [](auto& future) { future.get(); });

    return buffers;
}

/**
 * @brief Combine columns from filter and payload tables into a single table
 *
 * @param filter_table Filter table
 * @param payload_table Payload table
 * @return Combined table
 */
std::unique_ptr<cudf::table> combine_tables(
    std::unique_ptr<cudf::table> filter_table, std::unique_ptr<cudf::table> payload_table
) {
    auto filter_columns = filter_table->release();
    auto payload_columns = payload_table->release();

    auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
    all_columns.reserve(filter_columns.size() + payload_columns.size());
    std::ranges::move(filter_columns, std::back_inserter(all_columns));
    std::ranges::move(payload_columns, std::back_inserter(all_columns));
    auto table = std::make_unique<cudf::table>(std::move(all_columns));

    return table;
}

}  // namespace

/**
 * @brief Read parquet file with the next-gen parquet reader
 *
 * @param io_source io source to read
 * @param filter_expression Filter expression
 * @param filters Set of parquet filters to apply
 * @param stream CUDA stream for hybrid scan reader
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and
 * the final row validity column
 */
std::unique_ptr<cudf::table> hybrid_scan(
    cudf::io::parquet_reader_options const& options,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr
) {
    CUDF_FUNC_RANGE();

    RAPIDSMPF_EXPECTS(
        options.get_source().num_sources() == 1,
        "Hybrid scan expects exactly one parquet source"
    );

    auto datasource = make_datasource_from_source_info(options.get_source());

    // Fetch footer bytes and setup reader
    auto const footer_buffer = fetch_footer_bytes(std::ref(*datasource));
    auto const reader =
        std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
            cudf::host_span<uint8_t const>{footer_buffer->data(), footer_buffer->size()},
            options
        );

    // Get page index byte range from the reader
    auto const has_page_index = [&]() {
        auto const page_index_byte_range = reader->page_index_byte_range();
        if (page_index_byte_range.is_empty()) {
            return false;
        }
        auto const page_index_buffer =
            fetch_page_index_bytes(std::ref(*datasource), page_index_byte_range);
        reader->setup_page_index(
            cudf::host_span<uint8_t const>{
                page_index_buffer->data(), page_index_buffer->size()
            }
        );
        return true;
    }();

    // Get all row groups from the reader
    auto input_row_group_indices = reader->all_row_groups(options);
    auto current_row_group_indices =
        cudf::host_span<cudf::size_type>(input_row_group_indices);

    // Filter row groups with stats
    auto stats_filtered_row_group_indices =
        reader->filter_row_groups_with_stats(current_row_group_indices, options, stream);
    // Update current row group indices
    current_row_group_indices = stats_filtered_row_group_indices;

    // TODO: Deliberately skipping dictionary and bloom filter based row group pruning
    // here since they usually cost more than the speed benefit they provide.


    // Number of rows in the final set of row groups
    auto const num_rows = reader->total_rows_in_row_groups(current_row_group_indices);

    // Construct a row mask column with all rows needed
    // TODO: Deliberately skipping column index based page pruning here since it usually
    // costs more than the speed benefit it provides.
    auto row_mask = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::BOOL8},
        num_rows,
        rmm::device_buffer{},
        0,
        stream,
        mr
    );

    // Check if we have page index in the parquet source. If so, use two-step hybrid scan
    if (has_page_index) {
        // Get column chunk byte ranges from the reader
        auto const filter_column_chunk_byte_ranges =
            reader->filter_column_chunks_byte_ranges(current_row_group_indices, options);
        auto filter_column_chunk_buffers = fetch_byte_ranges(
            std::ref(*datasource), filter_column_chunk_byte_ranges, stream, mr
        );

        // Materialize the table with only the filter columns
        auto row_mask_mutable_view = row_mask->mutable_view();
        auto filter_table =
            reader
                ->materialize_filter_columns(
                    current_row_group_indices,
                    std::move(filter_column_chunk_buffers),
                    row_mask_mutable_view,
                    cudf::io::parquet::experimental::use_data_page_mask::NO,
                    options,
                    stream
                )
                .tbl;

        // Get column chunk byte ranges from the reader
        auto const payload_column_chunk_byte_ranges =
            reader->payload_column_chunks_byte_ranges(current_row_group_indices, options);
        auto payload_column_chunk_buffers = fetch_byte_ranges(
            std::ref(*datasource), payload_column_chunk_byte_ranges, stream, mr
        );

        // Materialize the table with only the payload columns
        auto payload_table =
            reader
                ->materialize_payload_columns(
                    current_row_group_indices,
                    std::move(payload_column_chunk_buffers),
                    row_mask->view(),
                    cudf::io::parquet::experimental::use_data_page_mask::YES,
                    options,
                    stream
                )
                .tbl;

        return combine_tables(std::move(filter_table), std::move(payload_table));
    }
    // Otherwise, use single-step hybrid scan which is equivalent to
    // `cudf::io::read_parquet`
    else
    {
        // Workaround: Create temporary options and set a dummy filter expression to avoid
        // erroneous assertion in `payload_column_chunks_byte_ranges`. Remove this once PR
        // https://github.com/rapidsai/cudf/pull/20604 is merged
        auto temporary_options = options;
        auto scalar = cudf::numeric_scalar<int32_t>(0, false, stream);
        auto literal = cudf::ast::literal(scalar);
        auto filter = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);
        temporary_options.set_filter(filter);


        auto const single_step_reader =
            std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
                reader->parquet_metadata(), temporary_options
            );

        // Get column chunk byte ranges from the reader
        auto const all_column_chunk_byte_ranges =
            single_step_reader->payload_column_chunks_byte_ranges(
                current_row_group_indices, temporary_options
            );

        // Materialize the table with all payload columns
        auto [read_table, metadata] = single_step_reader->materialize_payload_columns(
            current_row_group_indices,
            fetch_byte_ranges(
                std::ref(*datasource), all_column_chunk_byte_ranges, stream, mr
            ),
            row_mask->view(),
            cudf::io::parquet::experimental::use_data_page_mask::YES,
            temporary_options,
            stream
        );

        auto const expression_converter =
            named_to_reference_converter(options.get_filter(), metadata);

        if (expression_converter.get_converted_expr().has_value()) {
            auto predicate = cudf::compute_column(
                *read_table,
                expression_converter.get_converted_expr().value().get(),
                stream,
                mr
            );
            RAPIDSMPF_EXPECTS(
                predicate->view().type().id() == cudf::type_id::BOOL8,
                "Predicate filter should return a boolean"
            );
            return cudf::apply_boolean_mask(
                read_table->view(), predicate->view(), stream, mr
            );
        } else {
            return std::move(read_table);
        }
    }
}