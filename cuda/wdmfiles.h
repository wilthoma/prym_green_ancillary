#ifndef WDMFILES_H
#define WDMFILES_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include <limits>
#include <utility>

#include "matrices.h"
#include "zstd_compat.h"

template <typename U>
void read_wdm_value_or_throw(std::istream& file, U& value, const std::string& field_name) {
    if (!(file >> value)) {
        throw std::runtime_error("Failed to read " + field_name + " from WDM file");
    }
}

template <typename T>
std::tuple<uint32_t, size_t, size_t, size_t> load_wdm_file_sym(
    const std::string& wdm_filename,
    std::vector<T>& row_precond,
    std::vector<T>& col_precond,
    std::vector<std::vector<T>>& v_list,
    std::vector<std::vector<T>>& curv_list,
    std::vector<std::vector<T>>& seq_list
) {
    bcw_zstd::InputStream input(wdm_filename);
    std::istream& file = input.stream();

    // Read the first line: m n p Nlen num_v
    size_t m, n, nlen, num_v;
    uint32_t p;
    file >> m >> n >> p >> nlen >> num_v;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read row_precond
    row_precond.resize(m);
    for (size_t i = 0; i < m; ++i) {
        file >> row_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read col_precond
    col_precond.resize(n);
    for (size_t i = 0; i < n; ++i) {
        file >> col_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read v_list
    v_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> v(n);
        for (size_t j = 0; j < n; ++j) {
            file >> v[j];
        }
        v_list.push_back(std::move(v));
    }

    // Read curv_list
    curv_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> curv(n);
        for (size_t j = 0; j < n; ++j) {
            file >> curv[j];
        }
        curv_list.push_back(std::move(curv));
    }

    // Read seq_list
    seq_list.clear();
    size_t seq_count = num_v * (num_v + 1) / 2;
    for (size_t i = 0; i < seq_count; ++i) {
        std::vector<T> seq(nlen);
        for (size_t j = 0; j < nlen; ++j) {
            file >> seq[j];
        }
        seq_list.push_back(std::move(seq));
    }

    // Ensure all vectors are of the correct size
    if (row_precond.size() != m) {
        throw std::runtime_error("Row preconditioner length does not match matrix rows");
    }
    if (col_precond.size() != n) {
        throw std::runtime_error("Column preconditioner length does not match matrix columns");
    }

    return std::make_tuple(p, m, n, num_v);
}

template <typename T>
std::tuple<uint32_t, size_t, size_t, size_t, size_t> load_wdm_initial_state(
    const std::string& wdm_filename,
    std::vector<T>& row_precond,
    std::vector<T>& col_precond,
    std::vector<std::vector<T>>& v_list
) {
    bcw_zstd::InputStream input(wdm_filename);
    std::istream& file = input.stream();

    size_t m = 0;
    size_t n = 0;
    size_t nlen = 0;
    size_t num_v = 0;
    uint32_t p = 0;
    read_wdm_value_or_throw(file, m, "matrix row count");
    read_wdm_value_or_throw(file, n, "matrix column count");
    read_wdm_value_or_throw(file, p, "prime");
    read_wdm_value_or_throw(file, nlen, "sequence length");
    read_wdm_value_or_throw(file, num_v, "vector count");

    row_precond.resize(m);
    for (size_t i = 0; i < m; ++i) {
        read_wdm_value_or_throw(file, row_precond[i], "row preconditioner entry");
    }

    col_precond.resize(n);
    for (size_t i = 0; i < n; ++i) {
        read_wdm_value_or_throw(file, col_precond[i], "column preconditioner entry");
    }

    v_list.clear();
    v_list.reserve(num_v);
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> v(n);
        for (size_t j = 0; j < n; ++j) {
            read_wdm_value_or_throw(file, v[j], "initial vector entry");
        }
        v_list.push_back(std::move(v));
    }

    if (row_precond.size() != m) {
        throw std::runtime_error("Row preconditioner length does not match matrix rows");
    }
    if (col_precond.size() != n) {
        throw std::runtime_error("Column preconditioner length does not match matrix columns");
    }
    if (v_list.size() != num_v) {
        throw std::runtime_error("Initial vector count does not match WDM header");
    }

    return std::make_tuple(p, m, n, num_v, nlen);
}



template <typename T>
void write_wdm_line(bcw_zstd::OutputStream& file, const std::vector<T>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            file.write_space();
        }
        file.write_number(values[i]);
    }
    file.write_newline();
}

template <typename T>
void save_wdm_file_sym(
    const std::string& wdm_filename,
    size_t n_rows,
    size_t n_cols,
    T theprime,
    const std::vector<T>& row_precond,
    const std::vector<T>& col_precond,
    const std::vector<std::vector<T>>& v_list,
    const std::vector<std::vector<T>>& curv_list,
    const std::vector<std::vector<T>>& seq_list
) {
    if (seq_list.empty()) {
        throw std::runtime_error("Cannot save WDM file with empty sequence list");
    }

    bcw_zstd::OutputStream file(
        wdm_filename,
        bcw_zstd::path_requests_compression(wdm_filename)
    );

    // Write the first line: m n p Nlen num_u num_v
    file.write_number(n_rows);
    file.write_space();
    file.write_number(n_cols);
    file.write_space();
    file.write_number(theprime);
    file.write_space();
    file.write_number(seq_list[0].size());
    file.write_space();
    file.write_number(v_list.size());
    file.write_newline();

    // Write the second line: row_precond
    write_wdm_line(file, row_precond);

    // Write the third line: col_precond
    write_wdm_line(file, col_precond);

    // Write the v_list
    for (const auto& vv : v_list) {
        auto v = prettify_vect(vv, theprime);
        write_wdm_line(file, v);
    }

    // Write the curv_list
    for (const auto& curvv : curv_list) {
        auto curv = prettify_vect(curvv, theprime);
        write_wdm_line(file, curv);
    }

    // Write the seq_list
    for (const auto& seq : seq_list) {
        auto seq_pretty = prettify_vect(seq, theprime);
        write_wdm_line(file, seq_pretty);
    }

    file.finish();
}



#endif // WDMFILES_H
