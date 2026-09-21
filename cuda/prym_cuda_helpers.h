
#ifndef PRYM_CUDA_HELPERS_H
#define PRYM_CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

const size_t DEFAULT_DOT_CHUNK_SIZE = 64;

struct ModulusParams {
    uint32_t prime;
    uint32_t reciprocal;
};

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        throw std::runtime_error("CUDA error");                                \
    }                                                                          \
}


__host__ __device__ inline uint32_t reduce_mod_u32(uint32_t x, const ModulusParams& mod) {
    #if defined(__CUDA_ARCH__)
    uint32_t q = __umulhi(x, mod.reciprocal);
    #else
    uint32_t q = static_cast<uint32_t>(
        (static_cast<uint64_t>(x) * static_cast<uint64_t>(mod.reciprocal)) >> 32
    );
    #endif
    uint32_t r = x - q * mod.prime;
    if (r >= mod.prime) {
        r -= mod.prime;
    }
    return r;
}

__host__ __device__ inline size_t upper_triangular_size(size_t n) {
    return n * (n + 1) / 2;
}

__host__ __device__ inline size_t upper_triangular_index(size_t row, size_t col, size_t n) {
    return row * (2 * n - row + 1) / 2 + (col - row);
}


template<typename T>
__global__ void modp_kernel(T *device_matrix, size_t matrix_size, size_t offset, ModulusParams mod) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix_size) {
        index += offset;
        device_matrix[index] = static_cast<T>(reduce_mod_u32(static_cast<uint32_t>(device_matrix[index]), mod));
    }
}

template<typename T, int DENSE_COLS>
__global__ void scale_dense_rows_kernel(
    size_t rows,
    const T* __restrict__ input,
    const T* __restrict__ scale,
    T* __restrict__ output,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= rows || dense_col >= DENSE_COLS) {
        return;
    }

    uint32_t x = static_cast<uint32_t>(input[row * DENSE_COLS + dense_col]);
    uint32_t s = static_cast<uint32_t>(scale[row]);
    output[row * DENSE_COLS + dense_col] =
        static_cast<T>(reduce_mod_u32(x * s, mod));
}

template<typename T>
__global__ void modp_pack_upper_kernel(
    const uint32_t* __restrict__ full_matrix,
    T* __restrict__ packed_sequence,
    size_t seq_position,
    size_t n,
    ModulusParams mod
) {
    size_t row = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n || row > col) {
        return;
    }

    size_t packed_size = upper_triangular_size(n);
    size_t packed_index = upper_triangular_index(row, col, n);
    packed_sequence[seq_position * packed_size + packed_index] =
        static_cast<T>(reduce_mod_u32(full_matrix[row + col * n], mod));
}

template<typename T>
__global__ void dense_gemm_TN_chunked3D_offset_u32(size_t n, size_t k,
    const T* __restrict__ A,
    const T* __restrict__ B,
    uint32_t* C,
    size_t offset,
    size_t chunk_size,
    ModulusParams mod)
{
    size_t chunk = blockIdx.x;
    size_t row = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t) blockIdx.z * blockDim.z + threadIdx.z;

    if (row > col) {
        return;
    }

    size_t chunk_start = chunk * chunk_size;
    size_t chunk_end = min(chunk_start + chunk_size, k);

    if (row < n && col < n) {
        uint32_t acc = 0;
        for (size_t i = chunk_start; i < chunk_end; ++i) {
            size_t index = (size_t)i * n;
            acc += static_cast<uint32_t>(A[index + row]) *
                   static_cast<uint32_t>(B[index + col]);
        }

        atomicAdd(&C[(size_t)row + (size_t)col * n + offset], reduce_mod_u32(acc, mod));
    }
}

template<typename T>
__global__ void dense_gemm_TN_chunked3D_offset(size_t n, size_t k,  // n = n_dense_vectors, k = n_veclen
    const T* __restrict__ A,  // Transposed: A^T [n x k]
    const T* __restrict__ B,  // B [k x n]
    T* C,                    // Output: [n x n] slice of C at position offset
    size_t offset, // offset in C, in units of myfloat
    size_t chunk_size, ModulusParams mod)
{
    size_t chunk = blockIdx.x;
    size_t row = (size_t) blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t) blockIdx.z * blockDim.z + threadIdx.z;

    // Compute the upper triangle, including the diagonal (row <= col).
    // The symmetric moment matrix is packed into WDM order after accumulation.
    if (row > col) {
        return;
    }

    size_t chunk_start = chunk * chunk_size;
    size_t chunk_end = min(chunk_start + chunk_size, k);

    if (row < n && col < n) {
        uint32_t acc = 0;
        for (size_t i = chunk_start; i < chunk_end; ++i) {
            // Explicitly cast to size_t to prevent 32-bit integer overflow on the index calculation
            size_t index = (size_t)i * n;
            acc += static_cast<uint32_t>(A[index + row]) *
                   static_cast<uint32_t>(B[index + col]);
        }

        // Accumulate the result into global memory (C[row + col * n])
        atomicAdd(&C[(size_t)row + (size_t)col * n + offset], static_cast<T>(reduce_mod_u32(acc, mod)));
    }
}

template<typename T>
__global__ void dense_gemm_with_row(size_t M, // number of rows in A and C
    size_t NA,  // number of columns in A
    size_t NC,  // number of columns in C
    const T* __restrict__ A,  // A [M x NA]
    const T* __restrict__ B,  // B [NA*NC x ?]
    T* C,                    // Output: C [M x NC] -- the result is added to C, it does not replace C
    size_t offset // the column of B that contains the matrix to multiply with A from the right to build C
    )
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    // we are computing the (row, col) entry of C

    if (row >= M || col >= NC) return;

    size_t NB = NA * NC; // number of rows in B
    size_t total_offset = NB * offset; // offset in B

    uint32_t sum = 0;

    for (size_t i = 0; i < NA; ++i) {
        uint32_t a = static_cast<uint32_t>(A[row * NA + i]);
        uint32_t b = static_cast<uint32_t>(B[total_offset + col * NA + i]); // multiply with the transpose of the matrix stored in B
        // T b = B[total_offset + i * NC + col]; // column-major access of B
        sum += a * b;
    }

    C[row * NC + col] += static_cast<T>(sum);

}

template<typename T>
struct CudaDenseMatrix;

inline unsigned int dense_column_block_width(size_t dense_cols) {
    unsigned int width = 1;
    while (width < dense_cols && width < 32) {
        width <<= 1;
    }
    return width;
}

inline ModulusParams make_modulus_params(uint32_t prime) {
    uint32_t reciprocal = prime == 0
        ? 0
        : static_cast<uint32_t>((uint64_t{1} << 32) / prime);
    return ModulusParams{prime, reciprocal};
}

template<typename T>
struct CudaDenseMatrix {
    size_t numRows;
    size_t numCols;
    T* d_data;

    static CudaDenseMatrix<T> allocate_uninitialized(size_t rows, size_t cols) {
        CudaDenseMatrix<T> cuda_matrix;
        cuda_matrix.numRows = rows;
        cuda_matrix.numCols = cols;

        size_t size_data = static_cast<size_t>(rows) * cols * sizeof(T);
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_data, size_data));

        return cuda_matrix;
    }

    static CudaDenseMatrix<T> allocate(size_t rows, size_t cols, T default_value = 0) {
        CudaDenseMatrix<T> cuda_matrix = allocate_uninitialized(rows, cols);
        size_t size_data = static_cast<size_t>(rows) * cols * sizeof(T);
        CHECK_CUDA(cudaMemset(cuda_matrix.d_data, default_value, size_data));

        return cuda_matrix;
    }

    static CudaDenseMatrix<T> from_host(const std::vector<T>& host_matrix, size_t rows, size_t cols) {
        CudaDenseMatrix<T> cuda_matrix;
        cuda_matrix.numRows = rows;
        cuda_matrix.numCols = cols;

        size_t size_data = static_cast<size_t>(rows) * cols * sizeof(T);
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_data, size_data));
        CHECK_CUDA(cudaMemcpy(cuda_matrix.d_data, host_matrix.data(), size_data, cudaMemcpyHostToDevice));

        return cuda_matrix;
    }

    void release() {
        CHECK_CUDA(cudaFree(d_data));
    }

    inline void modp(ModulusParams mod) {
        size_t size = numRows * numCols;
        //CHECK_CUDA(
            modp_kernel<<<((size + 255) / 256), 256>>>(d_data, size, 0, mod);
            CHECK_CUDA(cudaGetLastError());
        //);
    }

    inline void modp(T prime) {
        modp(make_modulus_params(static_cast<uint32_t>(prime)));
    }

    // computes the upper tringular part of this^T* B and stores the desult in dC, at position position
    inline void mTm_tri(
        const CudaDenseMatrix<T>& B,
        T* dC,
        size_t seq_position,
        ModulusParams mod,
        size_t dot_chunk_size = DEFAULT_DOT_CHUNK_SIZE
    ) {
        // Check if the dimensions are compatible
        if (numRows != B.numRows || numCols != B.numCols) {
            throw std::runtime_error("Matrix dimensions do not match for M^T * B.\n");
        }
        if (dot_chunk_size == 0) {
            throw std::runtime_error("Gram dot chunk size must be positive.");
        }
        size_t n_veclen = numRows;
        size_t n_dense_vectors = numCols;
        size_t Sp_size = n_dense_vectors * n_dense_vectors;

        size_t num_chunks = (n_veclen + dot_chunk_size - 1) / dot_chunk_size;
        dim3 blockDim(1, 16, 16);  // Define a 3D block (x, y, z)
        dim3 gridDim(num_chunks,
                     (n_dense_vectors + blockDim.y - 1) / blockDim.y,
                     (n_dense_vectors + blockDim.z - 1) / blockDim.z);

        size_t offset = seq_position * Sp_size; // offset in units of myfloat
        CHECK_CUDA(cudaMemset(dC + offset, 0, Sp_size * sizeof(T)));

        dense_gemm_TN_chunked3D_offset<<<gridDim, blockDim>>>(
        n_dense_vectors, n_veclen, d_data, B.d_data, dC, offset, dot_chunk_size, mod);
        CHECK_CUDA(cudaGetLastError());


        modp_kernel<T><<<((Sp_size + 255) / 256), 256>>>(dC, Sp_size, offset, mod);

        CHECK_CUDA(cudaGetLastError());

    }

    // computes this^T * B into a full scratch matrix, then packs the upper triangle in WDM order
    inline void mTm_tri_pack_from_scratch(
        const CudaDenseMatrix<T>& B,
        T* dPacked,
        uint32_t* dFullScratch,
        size_t seq_position,
        ModulusParams mod,
        size_t dot_chunk_size = DEFAULT_DOT_CHUNK_SIZE
    ) {
        if (numRows != B.numRows || numCols != B.numCols) {
            throw std::runtime_error("Matrix dimensions do not match for packed M^T * B.\n");
        }
        if (dot_chunk_size == 0) {
            throw std::runtime_error("Gram dot chunk size must be positive.");
        }
        size_t n_veclen = numRows;
        size_t n_dense_vectors = numCols;
        size_t Sp_size = n_dense_vectors * n_dense_vectors;

        size_t num_chunks = (n_veclen + dot_chunk_size - 1) / dot_chunk_size;
        dim3 blockDim(1, 16, 16);
        dim3 gridDim(num_chunks,
                     (n_dense_vectors + blockDim.y - 1) / blockDim.y,
                     (n_dense_vectors + blockDim.z - 1) / blockDim.z);

        CHECK_CUDA(cudaMemset(dFullScratch, 0, Sp_size * sizeof(uint32_t)));

        dense_gemm_TN_chunked3D_offset_u32<<<gridDim, blockDim>>>(
            n_dense_vectors, n_veclen, d_data, B.d_data, dFullScratch, 0, dot_chunk_size, mod);
        CHECK_CUDA(cudaGetLastError());

        dim3 packBlock(16, 16);
        dim3 packGrid((n_dense_vectors + packBlock.x - 1) / packBlock.x,
                      (n_dense_vectors + packBlock.y - 1) / packBlock.y);
        modp_pack_upper_kernel<T><<<packGrid, packBlock>>>(
            dFullScratch, dPacked, seq_position, n_dense_vectors, mod);
        CHECK_CUDA(cudaGetLastError());
    }

    inline void gemm_slice(const CudaDenseMatrix<T>& G, CudaDenseMatrix<T>& D, size_t slice_index, T prime) {
        // does a matrix multiplication of this (A) with the slice_index's row of G, interpreted as a matrix.
        // adds the result to D
        if ( numRows != D.numRows) {
            throw std::runtime_error("Matrix dimensions do not match for GEMM slice.\n");
        }

        dim3 blockDim(16, 16);
        dim3 gridDim((numRows + blockDim.x - 1) / blockDim.x,
                     (D.numCols + blockDim.y - 1) / blockDim.y);

        dense_gemm_with_row<<<gridDim, blockDim>>>(
            numRows, numCols, D.numCols,
            d_data, G.d_data, D.d_data, slice_index);
        CHECK_CUDA(cudaGetLastError());

        // reduce result mod p
        D.modp(prime);
    }

    inline std::vector<T> copy_to_host() const {
        size_t size_data = static_cast<size_t>(numRows) * numCols * sizeof(T);
        std::vector<T> host_matrix(numRows * numCols);
        // wait for GPU to finish
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(host_matrix.data(), d_data, size_data, cudaMemcpyDeviceToHost));
        return host_matrix;
    }

};


#endif // PRYM_CUDA_HELPERS_H
