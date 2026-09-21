// Dense-vector layout and seeded generation used by the matrix-free operators.
#ifndef MATRICES_H
#define MATRICES_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>


template<typename T>
std::vector<std::vector<T>> reshape_to_vector_of_vectors(const std::vector<T>& input, size_t N) {
    // the output is a vector of N-vectors x_1, x_2, ..., x_k, with k = input.size()/N
    // the input vector is x_{11}, x_{21}, ....
    if (input.size() % N != 0) {
        throw std::invalid_argument("Input size is not divisible by N");
    }

    size_t num_vectors = input.size() / N;
    std::vector<std::vector<T>> result(num_vectors, std::vector<T>(N));

    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t j = 0; j < N; ++j) {
            result[i][j] = input[j * num_vectors + i];
        }
    }

    return result;
}


template<typename T, typename URBG>
std::vector<T> generate_random_vector(size_t size, T prime, URBG& gen, bool only_nonzero=false) {
    std::vector<T> random_vector(size);
    uint64_t lower = only_nonzero ? 1 : 0;
    uint64_t upper = static_cast<uint64_t>(prime) - 1;
    std::uniform_int_distribution<uint64_t> dis(lower, upper);

    for (size_t i = 0; i < size; ++i) {
        random_vector[i] = static_cast<T>(dis(gen));
    }

    return random_vector;
}



// WDM stores canonical residues, including for signed host-side values.
template <typename T>
std::vector<T> prettify_vect(const std::vector<T>& vec, T theprime) {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = (vec[i] % theprime + theprime) % theprime;
    }
    return result;
}

#endif // MATRICES_H
