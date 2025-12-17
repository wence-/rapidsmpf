#include <algorithm>
#include <list>

#include <thrust/iterator/counting_iterator.h>

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <rapidsmpf/error.hpp>

/**
 * @brief Converts named columns to index reference columns
 *
 * This class has been copied as-is from
 * https://github.com/rapidsai/cudf/blob/5bb723a38eccc1385dc9f2ece822dd612c9d7bfd/cpp/src/io/parquet/reader_impl_helpers.hpp#L574
 * until PR https://github.com/rapidsai/cudf/pull/20604 is merged
 */
class named_to_reference_converter : public cudf::ast::detail::expression_transformer {
  public:
    named_to_reference_converter(
        std::optional<std::reference_wrapper<cudf::ast::expression const>> expr,
        cudf::io::table_metadata const& metadata
    ) {
        if (!expr.has_value())
            return;
        // create map for column name.
        std::ranges::transform(
            metadata.schema_info,
            std::ranges::iota_view(std::size_t{0}),
            std::inserter(column_name_to_index, column_name_to_index.end()),
            [](auto const& sch, auto index) { return std::make_pair(sch.name, index); }
        );

        expr.value().get().accept(*this);
    }

    /**
     * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
     */
    std::reference_wrapper<cudf::ast::expression const> visit(
        cudf::ast::literal const& expr
    ) override {
        _converted_expr = std::reference_wrapper<cudf::ast::expression const>(expr);
        return expr;
    }

    /**
     * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
     */
    std::reference_wrapper<cudf::ast::expression const> visit(
        cudf::ast::column_reference const& expr
    ) override {
        _converted_expr = std::reference_wrapper<cudf::ast::expression const>(expr);
        return expr;
    }

    /**
     * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference
     * const& )
     */
    std::reference_wrapper<cudf::ast::expression const> visit(
        cudf::ast::column_name_reference const& expr
    ) override {
        {
            // check if column name is in metadata
            auto col_index_it = column_name_to_index.find(expr.get_column_name());
            if (col_index_it == column_name_to_index.end()) {
                CUDF_FAIL("Column name not found in metadata");
            }
            auto col_index = col_index_it->second;
            _col_ref.emplace_back(col_index);
            _converted_expr =
                std::reference_wrapper<cudf::ast::expression const>(_col_ref.back());
            return std::reference_wrapper<cudf::ast::expression const>(_col_ref.back());
        }
    }

    /**
     * @copydoc cudf::ast::detail::expression_transformer::visit(ast::operation const& )
     */
    std::reference_wrapper<cudf::ast::expression const> visit(
        cudf::ast::operation const& expr
    ) override {
        auto const operands = expr.get_operands();
        auto op = expr.get_operator();
        auto new_operands = visit_operands(operands);
        auto const operator_arity = cudf::ast::detail::ast_operator_arity(op);
        if (operator_arity == 2) {
            _operators.emplace_back(op, new_operands.front(), new_operands.back());
        } else if (operator_arity == 1) {
            _operators.emplace_back(op, new_operands.front());
        }
        _converted_expr =
            std::reference_wrapper<cudf::ast::expression const>(_operators.back());
        return std::reference_wrapper<cudf::ast::expression const>(_operators.back());
    }

    /**
     * @brief Returns the converted AST expression
     *
     * @return AST operation expression
     */
    [[nodiscard]] std::optional<std::reference_wrapper<cudf::ast::expression const>>
    get_converted_expr() const {
        return _converted_expr;
    }

  private:
    std::vector<std::reference_wrapper<cudf::ast::expression const>> visit_operands(
        cudf::host_span<std::reference_wrapper<cudf::ast::expression const> const>
            operands
    ) {
        std::vector<std::reference_wrapper<cudf::ast::expression const>>
            transformed_operands;
        for (auto const& operand : operands) {
            auto const new_operand = operand.get().accept(*this);
            transformed_operands.push_back(new_operand);
        }
        return transformed_operands;
    }

    std::unordered_map<std::string, cudf::size_type> column_name_to_index;
    std::optional<std::reference_wrapper<cudf::ast::expression const>> _converted_expr;
    // Using std::list or std::deque to avoid reference invalidation
    std::list<cudf::ast::column_reference> _col_ref;
    std::list<cudf::ast::operation> _operators;
};
