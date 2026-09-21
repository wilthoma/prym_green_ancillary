#ifndef CYCLIC_CUDA_HELPERS_H
#define CYCLIC_CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "matrices.h"
#include "prym_cuda_helpers.h"

struct CyclicRowDescriptor {
    uint32_t subset_rank;
    uint16_t beta;
    uint16_t reserved;
};

struct CyclicColumnDescriptor {
    uint32_t subset_rank;
    uint16_t a;
    uint16_t reserved;
};

struct CyclicRowIncidence {
    uint32_t i;
    uint32_t column_subset_rank;
    uint8_t negative;
};

struct CyclicColumnIncidence {
    uint32_t i;
    uint32_t row_subset_rank;
    uint8_t negative;
};

struct CyclicRowMixPair {
    uint32_t first;
    uint32_t second;
    uint16_t a;
    uint16_t b;
};

template<typename T>
struct CyclicPhiHostData {
    size_t g;
    size_t n;
    size_t m;
    size_t r;
    size_t numRows;
    size_t numCols;
    size_t rowSubsetCount;
    size_t columnSubsetCount;
    size_t rowWidth;
    size_t columnWidth;
    std::vector<T> mu;
    std::vector<T> rowPrecond;
    std::vector<T> colPrecond;
    std::vector<uint16_t> weightV;
    std::vector<uint16_t> weightA0;
    std::vector<uint16_t> weightA1;
    std::vector<uint32_t> a0CharOffsets;
    std::vector<uint16_t> a0ByChar;
    std::vector<uint32_t> a1CharOffsets;
    std::vector<uint16_t> a1ByChar;
    std::vector<uint32_t> rowOffsets;
    std::vector<CyclicRowDescriptor> rowDescriptors;
    std::vector<uint32_t> columnOffsets;
    std::vector<CyclicColumnDescriptor> columnDescriptors;
    std::vector<CyclicRowIncidence> rowIncidence;
    std::vector<CyclicColumnIncidence> columnIncidence;
};

template<typename T>
size_t cyclic_phi_host_data_memory_size(const CyclicPhiHostData<T>& host) {
    return host.mu.size() * sizeof(T)
        + host.rowPrecond.size() * sizeof(T)
        + host.colPrecond.size() * sizeof(T)
        + host.weightV.size() * sizeof(uint16_t)
        + host.weightA0.size() * sizeof(uint16_t)
        + host.weightA1.size() * sizeof(uint16_t)
        + host.a0CharOffsets.size() * sizeof(uint32_t)
        + host.a0ByChar.size() * sizeof(uint16_t)
        + host.a1CharOffsets.size() * sizeof(uint32_t)
        + host.a1ByChar.size() * sizeof(uint16_t)
        + host.rowOffsets.size() * sizeof(uint32_t)
        + host.rowDescriptors.size() * sizeof(CyclicRowDescriptor)
        + host.columnOffsets.size() * sizeof(uint32_t)
        + host.columnDescriptors.size() * sizeof(CyclicColumnDescriptor)
        + host.rowIncidence.size() * sizeof(CyclicRowIncidence)
        + host.columnIncidence.size() * sizeof(CyclicColumnIncidence);
}

__host__ __device__ inline uint32_t cyclic_add_mod(uint32_t a, uint32_t b, uint32_t r) {
    uint32_t out = a + b;
    return out >= r ? out - r : out;
}

__host__ __device__ inline uint32_t cyclic_sub_mod(uint32_t a, uint32_t b, uint32_t r) {
    return a >= b ? a - b : a + r - b;
}

__host__ __device__ inline uint32_t cyclic_reduce_mod_u64(uint64_t x, const ModulusParams& mod) {
    return static_cast<uint32_t>(x % static_cast<uint64_t>(mod.prime));
}

template<typename T, int DENSE_COLS, bool TRANSPOSE>
__global__ void cyclic_rowmix_pair_kernel(
    size_t pair_count,
    const CyclicRowMixPair* __restrict__ pairs,
    T* __restrict__ block,
    ModulusParams mod
) {
    size_t pair_index = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (pair_index >= pair_count || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicRowMixPair pair = pairs[pair_index];
    T* first_cell = block + static_cast<size_t>(pair.first) * DENSE_COLS + dense_col;
    T* second_cell = block + static_cast<size_t>(pair.second) * DENSE_COLS + dense_col;
    uint64_t x = static_cast<uint32_t>(*first_cell);
    uint64_t y = static_cast<uint32_t>(*second_cell);
    uint64_t a = pair.a;
    uint64_t b = pair.b;
    uint64_t one_plus_ab = 1 + a * b;

    uint64_t first_value;
    uint64_t second_value;
    if (TRANSPOSE) {
        first_value = x + b * y;
        second_value = a * x + one_plus_ab * y;
    } else {
        first_value = x + a * y;
        second_value = b * x + one_plus_ab * y;
    }

    *first_cell = static_cast<T>(cyclic_reduce_mod_u64(first_value, mod));
    *second_cell = static_cast<T>(cyclic_reduce_mod_u64(second_value, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_phi_forward_scaled_kernel(
    size_t rows,
    size_t n,
    size_t g,
    size_t r,
    size_t row_width,
    const T* __restrict__ mu,
    const T* __restrict__ row_precond,
    const uint16_t* __restrict__ weight_v,
    const uint16_t* __restrict__ weight_a1,
    const uint32_t* __restrict__ a0_offsets,
    const uint16_t* __restrict__ a0_by_char,
    const uint32_t* __restrict__ column_offsets,
    const CyclicRowDescriptor* __restrict__ row_descriptors,
    const CyclicRowIncidence* __restrict__ row_incidence,
    const T* __restrict__ B,
    T* __restrict__ C,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= rows || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicRowDescriptor desc = row_descriptors[row];
    uint32_t beta = desc.beta;
    uint32_t beta_weight = weight_a1[beta];
    const CyclicRowIncidence* incidences = row_incidence + static_cast<size_t>(desc.subset_rank) * row_width;

    uint32_t sum = 0;
    for (size_t pos = 0; pos < row_width; ++pos) {
        CyclicRowIncidence incidence = incidences[pos];
        uint32_t q = cyclic_sub_mod(beta_weight, weight_v[incidence.i], static_cast<uint32_t>(r));
        uint32_t start = a0_offsets[q];
        uint32_t end = a0_offsets[q + 1];
        uint32_t column_base = column_offsets[incidence.column_subset_rank];
        for (uint32_t idx = start; idx < end; ++idx) {
            uint32_t local_a = idx - start;
            uint32_t a = a0_by_char[idx];
            uint32_t raw = static_cast<uint32_t>(
                mu[((static_cast<size_t>(incidence.i) * n + beta) * g) + a]
            );
            if (raw == 0) {
                continue;
            }
            uint32_t x = static_cast<uint32_t>(
                B[(static_cast<size_t>(column_base) + local_a) * DENSE_COLS + dense_col]
            );
            uint32_t coeff = incidence.negative ? mod.prime - raw : raw;
            sum += coeff * x;
        }
    }

    uint32_t reduced = reduce_mod_u32(sum, mod);
    C[row * DENSE_COLS + dense_col] = static_cast<T>(
        reduce_mod_u32(reduced * static_cast<uint32_t>(row_precond[row]), mod)
    );
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_phi_transpose_kernel(
    size_t cols,
    size_t n,
    size_t g,
    size_t r,
    size_t column_width,
    const T* __restrict__ mu,
    const T* __restrict__ col_precond,
    const uint16_t* __restrict__ weight_v,
    const uint16_t* __restrict__ weight_a0,
    const uint32_t* __restrict__ a1_offsets,
    const uint16_t* __restrict__ a1_by_char,
    const uint32_t* __restrict__ row_offsets,
    const CyclicColumnDescriptor* __restrict__ column_descriptors,
    const CyclicColumnIncidence* __restrict__ column_incidence,
    const T* __restrict__ B,
    T* __restrict__ C,
    ModulusParams mod
) {
    size_t col = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (col >= cols || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicColumnDescriptor desc = column_descriptors[col];
    uint32_t a = desc.a;
    uint32_t a_weight = weight_a0[a];
    const CyclicColumnIncidence* incidences =
        column_incidence + static_cast<size_t>(desc.subset_rank) * column_width;

    uint32_t sum = 0;
    for (size_t pos = 0; pos < column_width; ++pos) {
        CyclicColumnIncidence incidence = incidences[pos];
        uint32_t q = cyclic_add_mod(a_weight, weight_v[incidence.i], static_cast<uint32_t>(r));
        uint32_t start = a1_offsets[q];
        uint32_t end = a1_offsets[q + 1];
        uint32_t row_base = row_offsets[incidence.row_subset_rank];
        for (uint32_t idx = start; idx < end; ++idx) {
            uint32_t local_beta = idx - start;
            uint32_t beta = a1_by_char[idx];
            uint32_t raw = static_cast<uint32_t>(
                mu[((static_cast<size_t>(incidence.i) * n + beta) * g) + a]
            );
            if (raw == 0) {
                continue;
            }
            uint32_t y = static_cast<uint32_t>(
                B[(static_cast<size_t>(row_base) + local_beta) * DENSE_COLS + dense_col]
            );
            uint32_t coeff = incidence.negative ? mod.prime - raw : raw;
            sum += coeff * y;
        }
    }

    uint32_t reduced = reduce_mod_u32(sum, mod);
    C[col * DENSE_COLS + dense_col] = static_cast<T>(
        reduce_mod_u32(reduced * static_cast<uint32_t>(col_precond[col]), mod)
    );
}

template<typename T>
struct CudaCyclicPhiOperator {
    size_t g;
    size_t n;
    size_t m;
    size_t r;
    size_t numRows;
    size_t numCols;
    size_t rowSubsetCount;
    size_t columnSubsetCount;
    size_t rowWidth;
    size_t columnWidth;
    T* d_mu;
    T* d_rowPrecond;
    T* d_colPrecond;
    uint16_t* d_weightV;
    uint16_t* d_weightA0;
    uint16_t* d_weightA1;
    uint32_t* d_a0CharOffsets;
    uint16_t* d_a0ByChar;
    uint32_t* d_a1CharOffsets;
    uint16_t* d_a1ByChar;
    uint32_t* d_rowOffsets;
    CyclicRowDescriptor* d_rowDescriptors;
    uint32_t* d_columnOffsets;
    CyclicColumnDescriptor* d_columnDescriptors;
    CyclicRowIncidence* d_rowIncidence;
    CyclicColumnIncidence* d_columnIncidence;

    static CudaCyclicPhiOperator<T> from_host(const CyclicPhiHostData<T>& host) {
        CudaCyclicPhiOperator<T> op;
        op.g = host.g;
        op.n = host.n;
        op.m = host.m;
        op.r = host.r;
        op.numRows = host.numRows;
        op.numCols = host.numCols;
        op.rowSubsetCount = host.rowSubsetCount;
        op.columnSubsetCount = host.columnSubsetCount;
        op.rowWidth = host.rowWidth;
        op.columnWidth = host.columnWidth;
        op.d_mu = nullptr;
        op.d_rowPrecond = nullptr;
        op.d_colPrecond = nullptr;
        op.d_weightV = nullptr;
        op.d_weightA0 = nullptr;
        op.d_weightA1 = nullptr;
        op.d_a0CharOffsets = nullptr;
        op.d_a0ByChar = nullptr;
        op.d_a1CharOffsets = nullptr;
        op.d_a1ByChar = nullptr;
        op.d_rowOffsets = nullptr;
        op.d_rowDescriptors = nullptr;
        op.d_columnOffsets = nullptr;
        op.d_columnDescriptors = nullptr;
        op.d_rowIncidence = nullptr;
        op.d_columnIncidence = nullptr;

        CHECK_CUDA(cudaMalloc((void**)&op.d_mu, host.mu.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_rowPrecond, host.rowPrecond.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_colPrecond, host.colPrecond.size() * sizeof(T)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_weightV, host.weightV.size() * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_weightA0, host.weightA0.size() * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_weightA1, host.weightA1.size() * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_a0CharOffsets, host.a0CharOffsets.size() * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_a0ByChar, host.a0ByChar.size() * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_a1CharOffsets, host.a1CharOffsets.size() * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_a1ByChar, host.a1ByChar.size() * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_rowOffsets, host.rowOffsets.size() * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_rowDescriptors, host.rowDescriptors.size() * sizeof(CyclicRowDescriptor)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_columnOffsets, host.columnOffsets.size() * sizeof(uint32_t)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_columnDescriptors, host.columnDescriptors.size() * sizeof(CyclicColumnDescriptor)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_rowIncidence, host.rowIncidence.size() * sizeof(CyclicRowIncidence)));
        CHECK_CUDA(cudaMalloc((void**)&op.d_columnIncidence, host.columnIncidence.size() * sizeof(CyclicColumnIncidence)));

        CHECK_CUDA(cudaMemcpy(op.d_mu, host.mu.data(), host.mu.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_rowPrecond, host.rowPrecond.data(), host.rowPrecond.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_colPrecond, host.colPrecond.data(), host.colPrecond.size() * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_weightV, host.weightV.data(), host.weightV.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_weightA0, host.weightA0.data(), host.weightA0.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_weightA1, host.weightA1.data(), host.weightA1.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_a0CharOffsets, host.a0CharOffsets.data(), host.a0CharOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_a0ByChar, host.a0ByChar.data(), host.a0ByChar.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_a1CharOffsets, host.a1CharOffsets.data(), host.a1CharOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_a1ByChar, host.a1ByChar.data(), host.a1ByChar.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_rowOffsets, host.rowOffsets.data(), host.rowOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_rowDescriptors, host.rowDescriptors.data(), host.rowDescriptors.size() * sizeof(CyclicRowDescriptor), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_columnOffsets, host.columnOffsets.data(), host.columnOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_columnDescriptors, host.columnDescriptors.data(), host.columnDescriptors.size() * sizeof(CyclicColumnDescriptor), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_rowIncidence, host.rowIncidence.data(), host.rowIncidence.size() * sizeof(CyclicRowIncidence), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(op.d_columnIncidence, host.columnIncidence.data(), host.columnIncidence.size() * sizeof(CyclicColumnIncidence), cudaMemcpyHostToDevice));

        return op;
    }

    void release() {
        if (d_mu != nullptr) CHECK_CUDA(cudaFree(d_mu));
        if (d_rowPrecond != nullptr) CHECK_CUDA(cudaFree(d_rowPrecond));
        if (d_colPrecond != nullptr) CHECK_CUDA(cudaFree(d_colPrecond));
        if (d_weightV != nullptr) CHECK_CUDA(cudaFree(d_weightV));
        if (d_weightA0 != nullptr) CHECK_CUDA(cudaFree(d_weightA0));
        if (d_weightA1 != nullptr) CHECK_CUDA(cudaFree(d_weightA1));
        if (d_a0CharOffsets != nullptr) CHECK_CUDA(cudaFree(d_a0CharOffsets));
        if (d_a0ByChar != nullptr) CHECK_CUDA(cudaFree(d_a0ByChar));
        if (d_a1CharOffsets != nullptr) CHECK_CUDA(cudaFree(d_a1CharOffsets));
        if (d_a1ByChar != nullptr) CHECK_CUDA(cudaFree(d_a1ByChar));
        if (d_rowOffsets != nullptr) CHECK_CUDA(cudaFree(d_rowOffsets));
        if (d_rowDescriptors != nullptr) CHECK_CUDA(cudaFree(d_rowDescriptors));
        if (d_columnOffsets != nullptr) CHECK_CUDA(cudaFree(d_columnOffsets));
        if (d_columnDescriptors != nullptr) CHECK_CUDA(cudaFree(d_columnDescriptors));
        if (d_rowIncidence != nullptr) CHECK_CUDA(cudaFree(d_rowIncidence));
        if (d_columnIncidence != nullptr) CHECK_CUDA(cudaFree(d_columnIncidence));
        d_mu = nullptr;
        d_rowPrecond = nullptr;
        d_colPrecond = nullptr;
        d_weightV = nullptr;
        d_weightA0 = nullptr;
        d_weightA1 = nullptr;
        d_a0CharOffsets = nullptr;
        d_a0ByChar = nullptr;
        d_a1CharOffsets = nullptr;
        d_a1ByChar = nullptr;
        d_rowOffsets = nullptr;
        d_rowDescriptors = nullptr;
        d_columnOffsets = nullptr;
        d_columnDescriptors = nullptr;
        d_rowIncidence = nullptr;
        d_columnIncidence = nullptr;
    }

    size_t get_memory_size() const {
        return n * n * g * sizeof(T)
            + numRows * sizeof(T)
            + numCols * sizeof(T)
            + n * sizeof(uint16_t)
            + g * sizeof(uint16_t)
            + n * sizeof(uint16_t)
            + (r + 1) * 2 * sizeof(uint32_t)
            + (g + n) * sizeof(uint16_t)
            + (rowSubsetCount + 1) * sizeof(uint32_t)
            + numRows * sizeof(CyclicRowDescriptor)
            + (columnSubsetCount + 1) * sizeof(uint32_t)
            + numCols * sizeof(CyclicColumnDescriptor)
            + rowSubsetCount * rowWidth * sizeof(CyclicRowIncidence)
            + columnSubsetCount * columnWidth * sizeof(CyclicColumnIncidence);
    }

    inline void scale_column_block(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& scaled,
        ModulusParams mod
    ) const {
        if (B.numRows != numCols || scaled.numRows != numCols || scaled.numCols != B.numCols) {
            throw std::runtime_error("Cyclic Phi column scaling dimensions do not match.");
        }
        launch_scale_column_block(B, scaled, mod);
    }

    inline void forward_scaled_columns(
        const CudaDenseMatrix<T>& scaled_B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        if (scaled_B.numRows != numCols) {
            throw std::runtime_error("Cyclic Phi forward dimensions do not match input block.");
        }
        if (C.numRows != numRows || C.numCols != scaled_B.numCols) {
            throw std::runtime_error("Cyclic Phi forward dimensions do not match output block.");
        }
        launch_forward_scaled(scaled_B, C, mod);
    }

    inline void transpose(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        if (B.numRows != numRows) {
            throw std::runtime_error("Cyclic Phi transpose dimensions do not match input block.");
        }
        if (C.numRows != numCols || C.numCols != B.numCols) {
            throw std::runtime_error("Cyclic Phi transpose dimensions do not match output block.");
        }
        launch_transpose(B, C, mod);
    }

private:
    template<int DENSE_COLS>
    inline void launch_forward_scaled_static(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
        const unsigned int block_rows = std::max(1u, 256u / block_cols);
        dim3 blockDim(block_cols, block_rows);
        dim3 gridDim((numRows + block_rows - 1) / block_rows);
        cyclic_phi_forward_scaled_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
            numRows,
            n,
            g,
            r,
            rowWidth,
            d_mu,
            d_rowPrecond,
            d_weightV,
            d_weightA1,
            d_a0CharOffsets,
            d_a0ByChar,
            d_columnOffsets,
            d_rowDescriptors,
            d_rowIncidence,
            B.d_data,
            C.d_data,
            mod
        );
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Cyclic Phi forward kernel launch failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA cyclic Phi forward kernel launch failed");
        }
    }

    template<int DENSE_COLS>
    inline void launch_transpose_static(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
        const unsigned int block_rows = std::max(1u, 256u / block_cols);
        dim3 blockDim(block_cols, block_rows);
        dim3 gridDim((numCols + block_rows - 1) / block_rows);
        cyclic_phi_transpose_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
            numCols,
            n,
            g,
            r,
            columnWidth,
            d_mu,
            d_colPrecond,
            d_weightV,
            d_weightA0,
            d_a1CharOffsets,
            d_a1ByChar,
            d_rowOffsets,
            d_columnDescriptors,
            d_columnIncidence,
            B.d_data,
            C.d_data,
            mod
        );
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Cyclic Phi transpose kernel launch failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA cyclic Phi transpose kernel launch failed");
        }
    }

    template<int DENSE_COLS>
    inline void launch_scale_column_block_static(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& scaled,
        ModulusParams mod
    ) const {
        const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
        const unsigned int block_rows = std::max(1u, 256u / block_cols);
        dim3 blockDim(block_cols, block_rows);
        dim3 gridDim((numCols + block_rows - 1) / block_rows);
        scale_dense_rows_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
            numCols,
            B.d_data,
            d_colPrecond,
            scaled.d_data,
            mod
        );
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Cyclic Phi column scaling kernel launch failed: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA cyclic Phi column scaling kernel launch failed");
        }
    }

    inline void launch_forward_scaled(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        switch (B.numCols) {
            case 1: launch_forward_scaled_static<1>(B, C, mod); break;
            case 2: launch_forward_scaled_static<2>(B, C, mod); break;
            case 3: launch_forward_scaled_static<3>(B, C, mod); break;
            case 4: launch_forward_scaled_static<4>(B, C, mod); break;
            case 5: launch_forward_scaled_static<5>(B, C, mod); break;
            case 6: launch_forward_scaled_static<6>(B, C, mod); break;
            case 7: launch_forward_scaled_static<7>(B, C, mod); break;
            case 8: launch_forward_scaled_static<8>(B, C, mod); break;
            case 9: launch_forward_scaled_static<9>(B, C, mod); break;
            case 10: launch_forward_scaled_static<10>(B, C, mod); break;
            case 11: launch_forward_scaled_static<11>(B, C, mod); break;
            case 12: launch_forward_scaled_static<12>(B, C, mod); break;
            case 13: launch_forward_scaled_static<13>(B, C, mod); break;
            case 14: launch_forward_scaled_static<14>(B, C, mod); break;
            case 15: launch_forward_scaled_static<15>(B, C, mod); break;
            case 16: launch_forward_scaled_static<16>(B, C, mod); break;
            default:
                throw std::runtime_error("Cyclic Phi CUDA kernels currently support -v values from 1 to 16");
        }
    }

    inline void launch_transpose(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C,
        ModulusParams mod
    ) const {
        switch (B.numCols) {
            case 1: launch_transpose_static<1>(B, C, mod); break;
            case 2: launch_transpose_static<2>(B, C, mod); break;
            case 3: launch_transpose_static<3>(B, C, mod); break;
            case 4: launch_transpose_static<4>(B, C, mod); break;
            case 5: launch_transpose_static<5>(B, C, mod); break;
            case 6: launch_transpose_static<6>(B, C, mod); break;
            case 7: launch_transpose_static<7>(B, C, mod); break;
            case 8: launch_transpose_static<8>(B, C, mod); break;
            case 9: launch_transpose_static<9>(B, C, mod); break;
            case 10: launch_transpose_static<10>(B, C, mod); break;
            case 11: launch_transpose_static<11>(B, C, mod); break;
            case 12: launch_transpose_static<12>(B, C, mod); break;
            case 13: launch_transpose_static<13>(B, C, mod); break;
            case 14: launch_transpose_static<14>(B, C, mod); break;
            case 15: launch_transpose_static<15>(B, C, mod); break;
            case 16: launch_transpose_static<16>(B, C, mod); break;
            default:
                throw std::runtime_error("Cyclic Phi CUDA kernels currently support -v values from 1 to 16");
        }
    }

    inline void launch_scale_column_block(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& scaled,
        ModulusParams mod
    ) const {
        switch (B.numCols) {
            case 1: launch_scale_column_block_static<1>(B, scaled, mod); break;
            case 2: launch_scale_column_block_static<2>(B, scaled, mod); break;
            case 3: launch_scale_column_block_static<3>(B, scaled, mod); break;
            case 4: launch_scale_column_block_static<4>(B, scaled, mod); break;
            case 5: launch_scale_column_block_static<5>(B, scaled, mod); break;
            case 6: launch_scale_column_block_static<6>(B, scaled, mod); break;
            case 7: launch_scale_column_block_static<7>(B, scaled, mod); break;
            case 8: launch_scale_column_block_static<8>(B, scaled, mod); break;
            case 9: launch_scale_column_block_static<9>(B, scaled, mod); break;
            case 10: launch_scale_column_block_static<10>(B, scaled, mod); break;
            case 11: launch_scale_column_block_static<11>(B, scaled, mod); break;
            case 12: launch_scale_column_block_static<12>(B, scaled, mod); break;
            case 13: launch_scale_column_block_static<13>(B, scaled, mod); break;
            case 14: launch_scale_column_block_static<14>(B, scaled, mod); break;
            case 15: launch_scale_column_block_static<15>(B, scaled, mod); break;
            case 16: launch_scale_column_block_static<16>(B, scaled, mod); break;
            default:
                throw std::runtime_error("Cyclic Phi CUDA kernels currently support -v values from 1 to 16");
        }
    }
};

template<typename T>
inline CudaDenseMatrix<T> cyclic_dense_view(T* data, size_t rows, size_t cols) {
    CudaDenseMatrix<T> view;
    view.numRows = rows;
    view.numCols = cols;
    view.d_data = data;
    return view;
}

template<typename T, int DENSE_COLS, bool TRANSPOSE>
inline void cyclic_launch_rowmix_static(
    CudaDenseMatrix<T>& block,
    const CyclicRowMixPair* pairs,
    size_t pair_count,
    ModulusParams mod
) {
    if (pair_count == 0) {
        return;
    }
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((pair_count + block_rows - 1) / block_rows);
    cyclic_rowmix_pair_kernel<T, DENSE_COLS, TRANSPOSE><<<gridDim, blockDim>>>(
        pair_count,
        pairs,
        block.d_data,
        mod
    );
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cyclic rowmix kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA cyclic rowmix kernel launch failed");
    }
}

template<typename T, bool TRANSPOSE>
inline void cyclic_launch_rowmix(
    CudaDenseMatrix<T>& block,
    const CyclicRowMixPair* pairs,
    size_t pair_count,
    ModulusParams mod
) {
    switch (block.numCols) {
        case 1: cyclic_launch_rowmix_static<T, 1, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 2: cyclic_launch_rowmix_static<T, 2, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 3: cyclic_launch_rowmix_static<T, 3, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 4: cyclic_launch_rowmix_static<T, 4, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 5: cyclic_launch_rowmix_static<T, 5, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 6: cyclic_launch_rowmix_static<T, 6, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 7: cyclic_launch_rowmix_static<T, 7, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 8: cyclic_launch_rowmix_static<T, 8, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 9: cyclic_launch_rowmix_static<T, 9, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 10: cyclic_launch_rowmix_static<T, 10, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 11: cyclic_launch_rowmix_static<T, 11, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 12: cyclic_launch_rowmix_static<T, 12, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 13: cyclic_launch_rowmix_static<T, 13, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 14: cyclic_launch_rowmix_static<T, 14, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 15: cyclic_launch_rowmix_static<T, 15, TRANSPOSE>(block, pairs, pair_count, mod); break;
        case 16: cyclic_launch_rowmix_static<T, 16, TRANSPOSE>(block, pairs, pair_count, mod); break;
        default:
            throw std::runtime_error("Cyclic rowmix kernels currently support -v values from 1 to 16");
    }
}

template<typename T>
inline void cyclic_launch_rowmix_forward(
    CudaDenseMatrix<T>& block,
    const CyclicRowMixPair* pairs,
    size_t pair_count,
    ModulusParams mod
) {
    cyclic_launch_rowmix<T, false>(block, pairs, pair_count, mod);
}

template<typename T>
inline void cyclic_launch_rowmix_transpose(
    CudaDenseMatrix<T>& block,
    const CyclicRowMixPair* pairs,
    size_t pair_count,
    ModulusParams mod
) {
    cyclic_launch_rowmix<T, true>(block, pairs, pair_count, mod);
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_elim_z_to_y_kernel(
    size_t y_rows,
    size_t kernel_dim,
    const CyclicColumnDescriptor* __restrict__ y_descriptors,
    const uint32_t* __restrict__ z_offsets,
    const CyclicColumnDescriptor* __restrict__ z_descriptors,
    const T* __restrict__ kernel_basis_a0_by_kernel,
    const T* __restrict__ z,
    T* __restrict__ y,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= y_rows || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicColumnDescriptor y_desc = y_descriptors[row];
    uint32_t sum = 0;
    uint32_t start = z_offsets[y_desc.subset_rank];
    uint32_t end = z_offsets[y_desc.subset_rank + 1];
    for (uint32_t idx = start; idx < end; ++idx) {
        uint32_t kernel_idx = z_descriptors[idx].a;
        uint32_t coeff = static_cast<uint32_t>(
            kernel_basis_a0_by_kernel[static_cast<size_t>(y_desc.a) * kernel_dim + kernel_idx]
        );
        if (coeff == 0) {
            continue;
        }
        uint32_t value = static_cast<uint32_t>(z[static_cast<size_t>(idx) * DENSE_COLS + dense_col]);
        sum += reduce_mod_u32(coeff * value, mod);
    }
    y[row * DENSE_COLS + dense_col] = static_cast<T>(reduce_mod_u32(sum, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_elim_subtract_Rt_kernel(
    size_t y_rows,
    size_t a1_dim,
    const CyclicColumnDescriptor* __restrict__ y_descriptors,
    const uint32_t* __restrict__ t_offsets,
    const CyclicRowDescriptor* __restrict__ t_descriptors,
    const T* __restrict__ right_inverse_a0_by_a1,
    const T* __restrict__ t,
    T* __restrict__ y,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= y_rows || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicColumnDescriptor y_desc = y_descriptors[row];
    uint32_t sum = static_cast<uint32_t>(y[row * DENSE_COLS + dense_col]);
    uint32_t start = t_offsets[y_desc.subset_rank];
    uint32_t end = t_offsets[y_desc.subset_rank + 1];
    for (uint32_t idx = start; idx < end; ++idx) {
        uint32_t beta = t_descriptors[idx].beta;
        uint32_t coeff = static_cast<uint32_t>(
            right_inverse_a0_by_a1[static_cast<size_t>(y_desc.a) * a1_dim + beta]
        );
        if (coeff == 0) {
            continue;
        }
        uint32_t value = static_cast<uint32_t>(t[static_cast<size_t>(idx) * DENSE_COLS + dense_col]);
        uint32_t term = reduce_mod_u32(coeff * value, mod);
        if (term != 0) {
            sum += mod.prime - term;
        }
    }
    y[row * DENSE_COLS + dense_col] = static_cast<T>(reduce_mod_u32(sum, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_elim_RT_h_kernel(
    size_t t_rows,
    size_t a1_dim,
    const CyclicRowDescriptor* __restrict__ t_descriptors,
    const uint32_t* __restrict__ y_offsets,
    const CyclicColumnDescriptor* __restrict__ y_descriptors,
    const T* __restrict__ right_inverse_a0_by_a1,
    const T* __restrict__ h,
    T* __restrict__ t_adj,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= t_rows || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicRowDescriptor t_desc = t_descriptors[row];
    uint32_t sum = 0;
    uint32_t start = y_offsets[t_desc.subset_rank];
    uint32_t end = y_offsets[t_desc.subset_rank + 1];
    for (uint32_t idx = start; idx < end; ++idx) {
        uint32_t a = y_descriptors[idx].a;
        uint32_t coeff = static_cast<uint32_t>(
            right_inverse_a0_by_a1[static_cast<size_t>(a) * a1_dim + t_desc.beta]
        );
        if (coeff == 0) {
            continue;
        }
        uint32_t value = static_cast<uint32_t>(h[static_cast<size_t>(idx) * DENSE_COLS + dense_col]);
        sum += reduce_mod_u32(coeff * value, mod);
    }
    t_adj[row * DENSE_COLS + dense_col] = static_cast<T>(reduce_mod_u32(sum, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_elim_BT_h_kernel(
    size_t z_rows,
    size_t kernel_dim,
    const CyclicColumnDescriptor* __restrict__ z_descriptors,
    const uint32_t* __restrict__ y_offsets,
    const CyclicColumnDescriptor* __restrict__ y_descriptors,
    const T* __restrict__ kernel_basis_a0_by_kernel,
    const T* __restrict__ h,
    T* __restrict__ z_adj,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= z_rows || dense_col >= DENSE_COLS) {
        return;
    }

    CyclicColumnDescriptor z_desc = z_descriptors[row];
    uint32_t sum = 0;
    uint32_t start = y_offsets[z_desc.subset_rank];
    uint32_t end = y_offsets[z_desc.subset_rank + 1];
    for (uint32_t idx = start; idx < end; ++idx) {
        uint32_t a = y_descriptors[idx].a;
        uint32_t coeff = static_cast<uint32_t>(
            kernel_basis_a0_by_kernel[static_cast<size_t>(a) * kernel_dim + z_desc.a]
        );
        if (coeff == 0) {
            continue;
        }
        uint32_t value = static_cast<uint32_t>(h[static_cast<size_t>(idx) * DENSE_COLS + dense_col]);
        sum += reduce_mod_u32(coeff * value, mod);
    }
    z_adj[row * DENSE_COLS + dense_col] = static_cast<T>(reduce_mod_u32(sum, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_elim_assemble_adjoint_kernel(
    size_t x_rows,
    size_t z_rows,
    const T* __restrict__ x_adj,
    const T* __restrict__ z_adj,
    const T* __restrict__ source_precond,
    T* __restrict__ output,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    size_t total_rows = x_rows + z_rows;
    if (row >= total_rows || dense_col >= DENSE_COLS) {
        return;
    }

    uint32_t value;
    if (row < x_rows) {
        value = static_cast<uint32_t>(x_adj[row * DENSE_COLS + dense_col]);
        if (value != 0) {
            value = mod.prime - value;
        }
    } else {
        size_t z_row = row - x_rows;
        value = static_cast<uint32_t>(z_adj[z_row * DENSE_COLS + dense_col]);
    }
    uint32_t scale = static_cast<uint32_t>(source_precond[row]);
    output[row * DENSE_COLS + dense_col] =
        static_cast<T>(reduce_mod_u32(value * scale, mod));
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_scale_rows_static(
    const CudaDenseMatrix<T>& input,
    const T* scale,
    CudaDenseMatrix<T>& output,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((input.numRows + block_rows - 1) / block_rows);
    scale_dense_rows_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        input.numRows, input.d_data, scale, output.d_data, mod);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Cyclic scale rows: Error in kernel launch: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA cyclic scale rows kernel launch failed");
    }
}

template<typename T>
inline void cyclic_launch_scale_rows(
    const CudaDenseMatrix<T>& input,
    const T* scale,
    CudaDenseMatrix<T>& output,
    ModulusParams mod
) {
    switch (input.numCols) {
        case 1: cyclic_launch_scale_rows_static<T, 1>(input, scale, output, mod); break;
        case 2: cyclic_launch_scale_rows_static<T, 2>(input, scale, output, mod); break;
        case 3: cyclic_launch_scale_rows_static<T, 3>(input, scale, output, mod); break;
        case 4: cyclic_launch_scale_rows_static<T, 4>(input, scale, output, mod); break;
        case 5: cyclic_launch_scale_rows_static<T, 5>(input, scale, output, mod); break;
        case 6: cyclic_launch_scale_rows_static<T, 6>(input, scale, output, mod); break;
        case 7: cyclic_launch_scale_rows_static<T, 7>(input, scale, output, mod); break;
        case 8: cyclic_launch_scale_rows_static<T, 8>(input, scale, output, mod); break;
        case 9: cyclic_launch_scale_rows_static<T, 9>(input, scale, output, mod); break;
        case 10: cyclic_launch_scale_rows_static<T, 10>(input, scale, output, mod); break;
        case 11: cyclic_launch_scale_rows_static<T, 11>(input, scale, output, mod); break;
        case 12: cyclic_launch_scale_rows_static<T, 12>(input, scale, output, mod); break;
        case 13: cyclic_launch_scale_rows_static<T, 13>(input, scale, output, mod); break;
        case 14: cyclic_launch_scale_rows_static<T, 14>(input, scale, output, mod); break;
        case 15: cyclic_launch_scale_rows_static<T, 15>(input, scale, output, mod); break;
        case 16: cyclic_launch_scale_rows_static<T, 16>(input, scale, output, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_elim_z_to_y_static(
    const CudaDenseMatrix<T>& z,
    const CudaDenseMatrix<T>& y,
    size_t kernel_dim,
    const CyclicColumnDescriptor* y_descriptors,
    const uint32_t* z_offsets,
    const CyclicColumnDescriptor* z_descriptors,
    const T* kernel_basis,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((y.numRows + block_rows - 1) / block_rows);
    cyclic_elim_z_to_y_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        y.numRows, kernel_dim, y_descriptors, z_offsets, z_descriptors,
        kernel_basis, z.d_data, y.d_data, mod);
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_elim_z_to_y(
    const CudaDenseMatrix<T>& z,
    const CudaDenseMatrix<T>& y,
    size_t kernel_dim,
    const CyclicColumnDescriptor* y_descriptors,
    const uint32_t* z_offsets,
    const CyclicColumnDescriptor* z_descriptors,
    const T* kernel_basis,
    ModulusParams mod
) {
    switch (y.numCols) {
        case 1: cyclic_launch_elim_z_to_y_static<T, 1>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 2: cyclic_launch_elim_z_to_y_static<T, 2>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 3: cyclic_launch_elim_z_to_y_static<T, 3>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 4: cyclic_launch_elim_z_to_y_static<T, 4>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 5: cyclic_launch_elim_z_to_y_static<T, 5>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 6: cyclic_launch_elim_z_to_y_static<T, 6>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 7: cyclic_launch_elim_z_to_y_static<T, 7>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 8: cyclic_launch_elim_z_to_y_static<T, 8>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 9: cyclic_launch_elim_z_to_y_static<T, 9>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 10: cyclic_launch_elim_z_to_y_static<T, 10>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 11: cyclic_launch_elim_z_to_y_static<T, 11>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 12: cyclic_launch_elim_z_to_y_static<T, 12>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 13: cyclic_launch_elim_z_to_y_static<T, 13>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 14: cyclic_launch_elim_z_to_y_static<T, 14>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 15: cyclic_launch_elim_z_to_y_static<T, 15>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        case 16: cyclic_launch_elim_z_to_y_static<T, 16>(z, y, kernel_dim, y_descriptors, z_offsets, z_descriptors, kernel_basis, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_elim_subtract_Rt_static(
    const CudaDenseMatrix<T>& t,
    const CudaDenseMatrix<T>& y,
    size_t a1_dim,
    const CyclicColumnDescriptor* y_descriptors,
    const uint32_t* t_offsets,
    const CyclicRowDescriptor* t_descriptors,
    const T* right_inverse,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((y.numRows + block_rows - 1) / block_rows);
    cyclic_elim_subtract_Rt_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        y.numRows, a1_dim, y_descriptors, t_offsets, t_descriptors,
        right_inverse, t.d_data, y.d_data, mod);
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_elim_subtract_Rt(
    const CudaDenseMatrix<T>& t,
    const CudaDenseMatrix<T>& y,
    size_t a1_dim,
    const CyclicColumnDescriptor* y_descriptors,
    const uint32_t* t_offsets,
    const CyclicRowDescriptor* t_descriptors,
    const T* right_inverse,
    ModulusParams mod
) {
    switch (y.numCols) {
        case 1: cyclic_launch_elim_subtract_Rt_static<T, 1>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 2: cyclic_launch_elim_subtract_Rt_static<T, 2>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 3: cyclic_launch_elim_subtract_Rt_static<T, 3>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 4: cyclic_launch_elim_subtract_Rt_static<T, 4>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 5: cyclic_launch_elim_subtract_Rt_static<T, 5>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 6: cyclic_launch_elim_subtract_Rt_static<T, 6>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 7: cyclic_launch_elim_subtract_Rt_static<T, 7>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 8: cyclic_launch_elim_subtract_Rt_static<T, 8>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 9: cyclic_launch_elim_subtract_Rt_static<T, 9>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 10: cyclic_launch_elim_subtract_Rt_static<T, 10>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 11: cyclic_launch_elim_subtract_Rt_static<T, 11>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 12: cyclic_launch_elim_subtract_Rt_static<T, 12>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 13: cyclic_launch_elim_subtract_Rt_static<T, 13>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 14: cyclic_launch_elim_subtract_Rt_static<T, 14>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 15: cyclic_launch_elim_subtract_Rt_static<T, 15>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        case 16: cyclic_launch_elim_subtract_Rt_static<T, 16>(t, y, a1_dim, y_descriptors, t_offsets, t_descriptors, right_inverse, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_elim_RT_h_static(
    const CudaDenseMatrix<T>& h,
    const CudaDenseMatrix<T>& t_adj,
    size_t a1_dim,
    const CyclicRowDescriptor* t_descriptors,
    const uint32_t* y_offsets,
    const CyclicColumnDescriptor* y_descriptors,
    const T* right_inverse,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((t_adj.numRows + block_rows - 1) / block_rows);
    cyclic_elim_RT_h_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        t_adj.numRows, a1_dim, t_descriptors, y_offsets, y_descriptors,
        right_inverse, h.d_data, t_adj.d_data, mod);
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_elim_RT_h(
    const CudaDenseMatrix<T>& h,
    const CudaDenseMatrix<T>& t_adj,
    size_t a1_dim,
    const CyclicRowDescriptor* t_descriptors,
    const uint32_t* y_offsets,
    const CyclicColumnDescriptor* y_descriptors,
    const T* right_inverse,
    ModulusParams mod
) {
    switch (h.numCols) {
        case 1: cyclic_launch_elim_RT_h_static<T, 1>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 2: cyclic_launch_elim_RT_h_static<T, 2>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 3: cyclic_launch_elim_RT_h_static<T, 3>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 4: cyclic_launch_elim_RT_h_static<T, 4>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 5: cyclic_launch_elim_RT_h_static<T, 5>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 6: cyclic_launch_elim_RT_h_static<T, 6>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 7: cyclic_launch_elim_RT_h_static<T, 7>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 8: cyclic_launch_elim_RT_h_static<T, 8>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 9: cyclic_launch_elim_RT_h_static<T, 9>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 10: cyclic_launch_elim_RT_h_static<T, 10>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 11: cyclic_launch_elim_RT_h_static<T, 11>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 12: cyclic_launch_elim_RT_h_static<T, 12>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 13: cyclic_launch_elim_RT_h_static<T, 13>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 14: cyclic_launch_elim_RT_h_static<T, 14>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 15: cyclic_launch_elim_RT_h_static<T, 15>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        case 16: cyclic_launch_elim_RT_h_static<T, 16>(h, t_adj, a1_dim, t_descriptors, y_offsets, y_descriptors, right_inverse, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_elim_BT_h_static(
    const CudaDenseMatrix<T>& h,
    const CudaDenseMatrix<T>& z_adj,
    size_t kernel_dim,
    const CyclicColumnDescriptor* z_descriptors,
    const uint32_t* y_offsets,
    const CyclicColumnDescriptor* y_descriptors,
    const T* kernel_basis,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((z_adj.numRows + block_rows - 1) / block_rows);
    cyclic_elim_BT_h_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        z_adj.numRows, kernel_dim, z_descriptors, y_offsets, y_descriptors,
        kernel_basis, h.d_data, z_adj.d_data, mod);
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_elim_BT_h(
    const CudaDenseMatrix<T>& h,
    const CudaDenseMatrix<T>& z_adj,
    size_t kernel_dim,
    const CyclicColumnDescriptor* z_descriptors,
    const uint32_t* y_offsets,
    const CyclicColumnDescriptor* y_descriptors,
    const T* kernel_basis,
    ModulusParams mod
) {
    switch (h.numCols) {
        case 1: cyclic_launch_elim_BT_h_static<T, 1>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 2: cyclic_launch_elim_BT_h_static<T, 2>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 3: cyclic_launch_elim_BT_h_static<T, 3>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 4: cyclic_launch_elim_BT_h_static<T, 4>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 5: cyclic_launch_elim_BT_h_static<T, 5>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 6: cyclic_launch_elim_BT_h_static<T, 6>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 7: cyclic_launch_elim_BT_h_static<T, 7>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 8: cyclic_launch_elim_BT_h_static<T, 8>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 9: cyclic_launch_elim_BT_h_static<T, 9>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 10: cyclic_launch_elim_BT_h_static<T, 10>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 11: cyclic_launch_elim_BT_h_static<T, 11>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 12: cyclic_launch_elim_BT_h_static<T, 12>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 13: cyclic_launch_elim_BT_h_static<T, 13>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 14: cyclic_launch_elim_BT_h_static<T, 14>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 15: cyclic_launch_elim_BT_h_static<T, 15>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        case 16: cyclic_launch_elim_BT_h_static<T, 16>(h, z_adj, kernel_dim, z_descriptors, y_offsets, y_descriptors, kernel_basis, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_elim_assemble_adjoint_static(
    const CudaDenseMatrix<T>& x_adj,
    const CudaDenseMatrix<T>& z_adj,
    const T* source_precond,
    CudaDenseMatrix<T>& output,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((output.numRows + block_rows - 1) / block_rows);
    cyclic_elim_assemble_adjoint_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        x_adj.numRows, z_adj.numRows, x_adj.d_data, z_adj.d_data,
        source_precond, output.d_data, mod);
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_elim_assemble_adjoint(
    const CudaDenseMatrix<T>& x_adj,
    const CudaDenseMatrix<T>& z_adj,
    const T* source_precond,
    CudaDenseMatrix<T>& output,
    ModulusParams mod
) {
    switch (output.numCols) {
        case 1: cyclic_launch_elim_assemble_adjoint_static<T, 1>(x_adj, z_adj, source_precond, output, mod); break;
        case 2: cyclic_launch_elim_assemble_adjoint_static<T, 2>(x_adj, z_adj, source_precond, output, mod); break;
        case 3: cyclic_launch_elim_assemble_adjoint_static<T, 3>(x_adj, z_adj, source_precond, output, mod); break;
        case 4: cyclic_launch_elim_assemble_adjoint_static<T, 4>(x_adj, z_adj, source_precond, output, mod); break;
        case 5: cyclic_launch_elim_assemble_adjoint_static<T, 5>(x_adj, z_adj, source_precond, output, mod); break;
        case 6: cyclic_launch_elim_assemble_adjoint_static<T, 6>(x_adj, z_adj, source_precond, output, mod); break;
        case 7: cyclic_launch_elim_assemble_adjoint_static<T, 7>(x_adj, z_adj, source_precond, output, mod); break;
        case 8: cyclic_launch_elim_assemble_adjoint_static<T, 8>(x_adj, z_adj, source_precond, output, mod); break;
        case 9: cyclic_launch_elim_assemble_adjoint_static<T, 9>(x_adj, z_adj, source_precond, output, mod); break;
        case 10: cyclic_launch_elim_assemble_adjoint_static<T, 10>(x_adj, z_adj, source_precond, output, mod); break;
        case 11: cyclic_launch_elim_assemble_adjoint_static<T, 11>(x_adj, z_adj, source_precond, output, mod); break;
        case 12: cyclic_launch_elim_assemble_adjoint_static<T, 12>(x_adj, z_adj, source_precond, output, mod); break;
        case 13: cyclic_launch_elim_assemble_adjoint_static<T, 13>(x_adj, z_adj, source_precond, output, mod); break;
        case 14: cyclic_launch_elim_assemble_adjoint_static<T, 14>(x_adj, z_adj, source_precond, output, mod); break;
        case 15: cyclic_launch_elim_assemble_adjoint_static<T, 15>(x_adj, z_adj, source_precond, output, mod); break;
        case 16: cyclic_launch_elim_assemble_adjoint_static<T, 16>(x_adj, z_adj, source_precond, output, mod); break;
        default:
            throw std::runtime_error("Cyclic bridge kernels currently support -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS, bool SUBTRACT>
__global__ void cyclic_dense_columns_axpy_kernel(
    size_t rows,
    size_t dense_columns,
    const T* __restrict__ dense_block,
    const T* __restrict__ coefficients,
    T* __restrict__ target,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= rows || dense_col >= DENSE_COLS) {
        return;
    }

    uint32_t sum = static_cast<uint32_t>(target[row * DENSE_COLS + dense_col]);
    for (size_t j = 0; j < dense_columns; ++j) {
        uint32_t a = static_cast<uint32_t>(dense_block[row * dense_columns + j]);
        uint32_t b = static_cast<uint32_t>(coefficients[j * DENSE_COLS + dense_col]);
        uint32_t term = reduce_mod_u32(a * b, mod);
        if (SUBTRACT) {
            if (term != 0) {
                sum += mod.prime - term;
            }
        } else {
            sum += term;
        }
    }
    target[row * DENSE_COLS + dense_col] =
        static_cast<T>(reduce_mod_u32(sum, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_dense_columns_transpose_accum_kernel(
    size_t rows,
    size_t dense_columns,
    size_t chunk_size,
    const T* __restrict__ dense_block,
    const T* __restrict__ target,
    uint32_t* __restrict__ scratch,
    ModulusParams mod
) {
    size_t chunk = blockIdx.x;
    size_t dense_col = blockIdx.y * blockDim.y + threadIdx.y;
    size_t column = blockIdx.z * blockDim.z + threadIdx.z;
    if (dense_col >= DENSE_COLS || column >= dense_columns) {
        return;
    }

    size_t begin = chunk * chunk_size;
    size_t end = begin + chunk_size;
    if (end > rows) {
        end = rows;
    }
    uint32_t acc = 0;
    for (size_t row = begin; row < end; ++row) {
        uint32_t a = static_cast<uint32_t>(dense_block[row * dense_columns + column]);
        uint32_t b = static_cast<uint32_t>(target[row * DENSE_COLS + dense_col]);
        acc += reduce_mod_u32(a * b, mod);
    }
    atomicAdd(&scratch[column * DENSE_COLS + dense_col], reduce_mod_u32(acc, mod));
}

template<typename T, int DENSE_COLS, bool SUBTRACT>
__global__ void cyclic_dense_columns_transpose_finish_kernel(
    size_t dense_columns,
    size_t output_offset,
    const uint32_t* __restrict__ scratch,
    const T* __restrict__ source_precond,
    T* __restrict__ output,
    ModulusParams mod
) {
    size_t column = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (column >= dense_columns || dense_col >= DENSE_COLS) {
        return;
    }

    uint32_t value = reduce_mod_u32(scratch[column * DENSE_COLS + dense_col], mod);
    if (SUBTRACT && value != 0) {
        value = mod.prime - value;
    }
    uint32_t scale = static_cast<uint32_t>(source_precond[output_offset + column]);
    output[(output_offset + column) * DENSE_COLS + dense_col] =
        static_cast<T>(reduce_mod_u32(value * scale, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_insert_complement_scaled_kernel(
    size_t full_rows,
    uint32_t drop0,
    uint32_t drop1,
    const T* __restrict__ input,
    const T* __restrict__ source_precond,
    T* __restrict__ full_output,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= full_rows || dense_col >= DENSE_COLS) {
        return;
    }
    if (row == drop0 || row == drop1) {
        full_output[row * DENSE_COLS + dense_col] = 0;
        return;
    }
    size_t source_row = row;
    if (row > drop0) {
        --source_row;
    }
    if (row > drop1) {
        --source_row;
    }
    uint32_t x = static_cast<uint32_t>(input[source_row * DENSE_COLS + dense_col]);
    uint32_t s = static_cast<uint32_t>(source_precond[source_row]);
    full_output[row * DENSE_COLS + dense_col] =
        static_cast<T>(reduce_mod_u32(x * s, mod));
}

template<typename T, int DENSE_COLS>
__global__ void cyclic_gather_complement_kernel(
    size_t full_rows,
    uint32_t drop0,
    uint32_t drop1,
    const T* __restrict__ full_input,
    T* __restrict__ output
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= full_rows || dense_col >= DENSE_COLS || row == drop0 || row == drop1) {
        return;
    }
    size_t output_row = row;
    if (row > drop0) {
        --output_row;
    }
    if (row > drop1) {
        --output_row;
    }
    output[output_row * DENSE_COLS + dense_col] =
        full_input[row * DENSE_COLS + dense_col];
}

template<typename T, int DENSE_COLS, bool SUBTRACT>
inline void cyclic_launch_dense_columns_axpy_static(
    const CudaDenseMatrix<T>& dense_block,
    const CudaDenseMatrix<T>& coefficients,
    CudaDenseMatrix<T>& target,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((target.numRows + block_rows - 1) / block_rows);
    cyclic_dense_columns_axpy_kernel<T, DENSE_COLS, SUBTRACT><<<gridDim, blockDim>>>(
        target.numRows,
        dense_block.numCols,
        dense_block.d_data,
        coefficients.d_data,
        target.d_data,
        mod
    );
    CHECK_CUDA(cudaGetLastError());
}

template<typename T, bool SUBTRACT>
inline void cyclic_launch_dense_columns_axpy(
    const CudaDenseMatrix<T>& dense_block,
    const CudaDenseMatrix<T>& coefficients,
    CudaDenseMatrix<T>& target,
    ModulusParams mod
) {
    if (dense_block.numRows != target.numRows || dense_block.numCols != coefficients.numRows
            || coefficients.numCols != target.numCols) {
        throw std::runtime_error("dense-column AXPY dimensions do not match");
    }
    switch (target.numCols) {
        case 1: cyclic_launch_dense_columns_axpy_static<T, 1, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 2: cyclic_launch_dense_columns_axpy_static<T, 2, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 3: cyclic_launch_dense_columns_axpy_static<T, 3, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 4: cyclic_launch_dense_columns_axpy_static<T, 4, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 5: cyclic_launch_dense_columns_axpy_static<T, 5, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 6: cyclic_launch_dense_columns_axpy_static<T, 6, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 7: cyclic_launch_dense_columns_axpy_static<T, 7, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 8: cyclic_launch_dense_columns_axpy_static<T, 8, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 9: cyclic_launch_dense_columns_axpy_static<T, 9, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 10: cyclic_launch_dense_columns_axpy_static<T, 10, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 11: cyclic_launch_dense_columns_axpy_static<T, 11, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 12: cyclic_launch_dense_columns_axpy_static<T, 12, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 13: cyclic_launch_dense_columns_axpy_static<T, 13, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 14: cyclic_launch_dense_columns_axpy_static<T, 14, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 15: cyclic_launch_dense_columns_axpy_static<T, 15, SUBTRACT>(dense_block, coefficients, target, mod); break;
        case 16: cyclic_launch_dense_columns_axpy_static<T, 16, SUBTRACT>(dense_block, coefficients, target, mod); break;
        default:
            throw std::runtime_error("dense-column AXPY supports -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS, bool SUBTRACT>
inline void cyclic_launch_dense_columns_transpose_static(
    const CudaDenseMatrix<T>& dense_block,
    const CudaDenseMatrix<T>& target,
    const T* source_precond,
    size_t output_offset,
    CudaDenseMatrix<T>& output,
    uint32_t* scratch,
    size_t chunk_size,
    ModulusParams mod
) {
    if (chunk_size == 0) {
        throw std::runtime_error("dense-column transpose chunk size must be positive");
    }
    const size_t scratch_entries = dense_block.numCols * DENSE_COLS;
    CHECK_CUDA(cudaMemset(scratch, 0, scratch_entries * sizeof(uint32_t)));
    size_t num_chunks = (dense_block.numRows + chunk_size - 1) / chunk_size;
    dim3 blockDim(1, 16, 16);
    dim3 gridDim(num_chunks, (DENSE_COLS + blockDim.y - 1) / blockDim.y,
                 (dense_block.numCols + blockDim.z - 1) / blockDim.z);
    cyclic_dense_columns_transpose_accum_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        dense_block.numRows,
        dense_block.numCols,
        chunk_size,
        dense_block.d_data,
        target.d_data,
        scratch,
        mod
    );
    CHECK_CUDA(cudaGetLastError());

    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 finishBlock(block_cols, block_rows);
    dim3 finishGrid((dense_block.numCols + block_rows - 1) / block_rows);
    cyclic_dense_columns_transpose_finish_kernel<T, DENSE_COLS, SUBTRACT><<<finishGrid, finishBlock>>>(
        dense_block.numCols,
        output_offset,
        scratch,
        source_precond,
        output.d_data,
        mod
    );
    CHECK_CUDA(cudaGetLastError());
}

template<typename T, bool SUBTRACT>
inline void cyclic_launch_dense_columns_transpose(
    const CudaDenseMatrix<T>& dense_block,
    const CudaDenseMatrix<T>& target,
    const T* source_precond,
    size_t output_offset,
    CudaDenseMatrix<T>& output,
    uint32_t* scratch,
    size_t chunk_size,
    ModulusParams mod
) {
    if (dense_block.numRows != target.numRows || target.numCols != output.numCols
            || output_offset + dense_block.numCols > output.numRows) {
        throw std::runtime_error("dense-column transpose dimensions do not match");
    }
    switch (target.numCols) {
        case 1: cyclic_launch_dense_columns_transpose_static<T, 1, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 2: cyclic_launch_dense_columns_transpose_static<T, 2, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 3: cyclic_launch_dense_columns_transpose_static<T, 3, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 4: cyclic_launch_dense_columns_transpose_static<T, 4, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 5: cyclic_launch_dense_columns_transpose_static<T, 5, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 6: cyclic_launch_dense_columns_transpose_static<T, 6, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 7: cyclic_launch_dense_columns_transpose_static<T, 7, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 8: cyclic_launch_dense_columns_transpose_static<T, 8, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 9: cyclic_launch_dense_columns_transpose_static<T, 9, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 10: cyclic_launch_dense_columns_transpose_static<T, 10, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 11: cyclic_launch_dense_columns_transpose_static<T, 11, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 12: cyclic_launch_dense_columns_transpose_static<T, 12, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 13: cyclic_launch_dense_columns_transpose_static<T, 13, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 14: cyclic_launch_dense_columns_transpose_static<T, 14, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 15: cyclic_launch_dense_columns_transpose_static<T, 15, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        case 16: cyclic_launch_dense_columns_transpose_static<T, 16, SUBTRACT>(dense_block, target, source_precond, output_offset, output, scratch, chunk_size, mod); break;
        default:
            throw std::runtime_error("dense-column transpose supports -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_insert_complement_scaled_static(
    const CudaDenseMatrix<T>& input,
    uint32_t drop0,
    uint32_t drop1,
    const T* source_precond,
    CudaDenseMatrix<T>& full_output,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((full_output.numRows + block_rows - 1) / block_rows);
    cyclic_insert_complement_scaled_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        full_output.numRows,
        drop0,
        drop1,
        input.d_data,
        source_precond,
        full_output.d_data,
        mod
    );
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_insert_complement_scaled(
    const CudaDenseMatrix<T>& input,
    uint32_t drop0,
    uint32_t drop1,
    const T* source_precond,
    CudaDenseMatrix<T>& full_output,
    ModulusParams mod
) {
    if (drop0 >= drop1 || drop1 >= full_output.numRows || input.numRows + 2 != full_output.numRows
            || input.numCols != full_output.numCols) {
        throw std::runtime_error("complement insertion dimensions do not match");
    }
    switch (input.numCols) {
        case 1: cyclic_launch_insert_complement_scaled_static<T, 1>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 2: cyclic_launch_insert_complement_scaled_static<T, 2>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 3: cyclic_launch_insert_complement_scaled_static<T, 3>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 4: cyclic_launch_insert_complement_scaled_static<T, 4>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 5: cyclic_launch_insert_complement_scaled_static<T, 5>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 6: cyclic_launch_insert_complement_scaled_static<T, 6>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 7: cyclic_launch_insert_complement_scaled_static<T, 7>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 8: cyclic_launch_insert_complement_scaled_static<T, 8>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 9: cyclic_launch_insert_complement_scaled_static<T, 9>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 10: cyclic_launch_insert_complement_scaled_static<T, 10>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 11: cyclic_launch_insert_complement_scaled_static<T, 11>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 12: cyclic_launch_insert_complement_scaled_static<T, 12>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 13: cyclic_launch_insert_complement_scaled_static<T, 13>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 14: cyclic_launch_insert_complement_scaled_static<T, 14>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 15: cyclic_launch_insert_complement_scaled_static<T, 15>(input, drop0, drop1, source_precond, full_output, mod); break;
        case 16: cyclic_launch_insert_complement_scaled_static<T, 16>(input, drop0, drop1, source_precond, full_output, mod); break;
        default:
            throw std::runtime_error("complement insertion supports -v values from 1 to 16");
    }
}

template<typename T, int DENSE_COLS>
inline void cyclic_launch_gather_complement_static(
    const CudaDenseMatrix<T>& full_input,
    uint32_t drop0,
    uint32_t drop1,
    CudaDenseMatrix<T>& output
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((full_input.numRows + block_rows - 1) / block_rows);
    cyclic_gather_complement_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        full_input.numRows,
        drop0,
        drop1,
        full_input.d_data,
        output.d_data
    );
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
inline void cyclic_launch_gather_complement(
    const CudaDenseMatrix<T>& full_input,
    uint32_t drop0,
    uint32_t drop1,
    CudaDenseMatrix<T>& output
) {
    if (drop0 >= drop1 || drop1 >= full_input.numRows || output.numRows + 2 != full_input.numRows
            || output.numCols != full_input.numCols) {
        throw std::runtime_error("complement gather dimensions do not match");
    }
    switch (output.numCols) {
        case 1: cyclic_launch_gather_complement_static<T, 1>(full_input, drop0, drop1, output); break;
        case 2: cyclic_launch_gather_complement_static<T, 2>(full_input, drop0, drop1, output); break;
        case 3: cyclic_launch_gather_complement_static<T, 3>(full_input, drop0, drop1, output); break;
        case 4: cyclic_launch_gather_complement_static<T, 4>(full_input, drop0, drop1, output); break;
        case 5: cyclic_launch_gather_complement_static<T, 5>(full_input, drop0, drop1, output); break;
        case 6: cyclic_launch_gather_complement_static<T, 6>(full_input, drop0, drop1, output); break;
        case 7: cyclic_launch_gather_complement_static<T, 7>(full_input, drop0, drop1, output); break;
        case 8: cyclic_launch_gather_complement_static<T, 8>(full_input, drop0, drop1, output); break;
        case 9: cyclic_launch_gather_complement_static<T, 9>(full_input, drop0, drop1, output); break;
        case 10: cyclic_launch_gather_complement_static<T, 10>(full_input, drop0, drop1, output); break;
        case 11: cyclic_launch_gather_complement_static<T, 11>(full_input, drop0, drop1, output); break;
        case 12: cyclic_launch_gather_complement_static<T, 12>(full_input, drop0, drop1, output); break;
        case 13: cyclic_launch_gather_complement_static<T, 13>(full_input, drop0, drop1, output); break;
        case 14: cyclic_launch_gather_complement_static<T, 14>(full_input, drop0, drop1, output); break;
        case 15: cyclic_launch_gather_complement_static<T, 15>(full_input, drop0, drop1, output); break;
        case 16: cyclic_launch_gather_complement_static<T, 16>(full_input, drop0, drop1, output); break;
        default:
            throw std::runtime_error("complement gather supports -v values from 1 to 16");
    }
}

#endif // CYCLIC_CUDA_HELPERS_H
