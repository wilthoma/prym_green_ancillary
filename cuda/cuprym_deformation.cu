// Journal production extraction: cyclic elimination and exact symmetric Wiedemann.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cctype>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "cyclic_cuda_helpers.h"
#include "include/CLI11.hpp"
#include "matrices.h"
#include "wdmfiles.h"

typedef uint16_t my_int;
typedef uint32_t accum_int;

constexpr size_t AUTO_GRAM_CHUNK_CAP = 128;
constexpr const char* ROWMIX_ALGORITHM_VERSION = "paired-sl2-target-v1";
constexpr uint64_t FNV1A64_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV1A64_PRIME = 1099511628211ULL;

using namespace std;

struct CyclicInstance {
    size_t genus;
    size_t n;
    size_t m;
    size_t r;
    uint32_t modulus;
    vector<my_int> mu;
    vector<uint16_t> weight_v;
    vector<uint16_t> weight_a0;
    vector<uint16_t> weight_a1;
    vector<uint64_t> expected_sector_columns;
    vector<uint64_t> expected_sector_rows;
    vector<uint64_t> expected_sector_nnz;
    bool has_elimination;
    size_t elimination_w_index;
    uint16_t elimination_w_weight;
    vector<my_int> elimination_right_inverse_a0_by_a1;
    vector<my_int> elimination_kernel_basis_a0_by_kernel;
    vector<uint16_t> elimination_kernel_weights;
    vector<uint64_t> expected_eliminated_sector_columns;
    vector<uint64_t> expected_eliminated_sector_rows;
};

struct CyclicGeometry {
    size_t genus;
    size_t n;
    size_t m;
    size_t r;
    size_t row_subset_count;
    size_t column_subset_count;
};

struct CharacterMap {
    vector<uint32_t> offsets;
    vector<uint16_t> indices;
};

struct CyclicCommonIncidence {
    vector<CyclicRowIncidence> row_incidence;
    vector<CyclicColumnIncidence> column_incidence;
    vector<uint16_t> row_subset_weights;
    vector<uint16_t> column_subset_weights;
    size_t row_width;
    size_t column_width;
};

struct CyclicSectorLayout {
    size_t sector;
    vector<uint32_t> row_offsets;
    vector<CyclicRowDescriptor> row_descriptors;
    vector<uint32_t> column_offsets;
    vector<CyclicColumnDescriptor> column_descriptors;
    uint64_t nnz;
};

struct CyclicTensorLayout {
    size_t sector;
    vector<uint32_t> offsets;
    vector<CyclicColumnDescriptor> descriptors;
};

struct RowMixPlan {
    size_t target_rows;
    size_t rounds;
    uint64_t seed;
    uint64_t realized_seed;
    vector<vector<CyclicRowMixPair>> round_pairs;

    bool enabled() const {
        return rounds > 0;
    }
};

struct RowMixOptions {
    size_t rounds;
    uint64_t seed;
};

struct ValidationOptions {
    bool enabled = false;
    size_t columns = 4;
};

struct DenseBlockFile {
    size_t rows = 0;
    size_t columns = 0;
    uint32_t prime = 0;
    vector<my_int> values;
};

bool is_prime(uint32_t n) {
    if (n <= 1) {
        return false;
    }
    for (uint32_t i = 2; i <= n / i; ++i) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

uint64_t checked_binom(size_t n, size_t k) {
    if (k > n) {
        return 0;
    }
    k = std::min(k, n - k);
    __uint128_t out = 1;
    for (size_t i = 0; i < k; ++i) {
        out = (out * (n - i)) / (i + 1);
        if (out > std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("binomial coefficient does not fit in u64");
        }
    }
    return static_cast<uint64_t>(out);
}

size_t checked_size(uint64_t value, const string& label) {
    if (value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(label + " does not fit in size_t");
    }
    return static_cast<size_t>(value);
}

size_t checked_mul(size_t a, size_t b, const string& label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(label + " overflows size_t");
    }
    return a * b;
}

uint32_t checked_u32_index(size_t value, const string& label) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(label + " does not fit in uint32_t");
    }
    return static_cast<uint32_t>(value);
}

CyclicGeometry compute_cyclic_geometry(size_t genus, size_t r) {
    if (genus < 6 || genus % 2 != 0) {
        throw std::runtime_error("cyclic Prym CUDA path expects an even genus >= 6");
    }
    size_t n = genus - 3;
    size_t m = genus / 2;
    if (r == 0 || m > n) {
        throw std::runtime_error("invalid cyclic Prym dimensions");
    }
    size_t row_subset_count = checked_size(checked_binom(n, m - 1), "row subset count");
    size_t column_subset_count = checked_size(checked_binom(n, m), "column subset count");
    return CyclicGeometry{genus, n, m, r, row_subset_count, column_subset_count};
}

vector<size_t> unrank_subset(uint64_t rank, size_t n, size_t k) {
    uint64_t total = checked_binom(n, k);
    if (rank >= total) {
        throw std::runtime_error("subset rank outside lexicographic range");
    }

    vector<size_t> subset;
    subset.reserve(k);
    size_t start = 0;
    for (size_t pos = 0; pos < k; ++pos) {
        size_t remaining = k - pos - 1;
        bool chosen = false;
        for (size_t value = start; value < n; ++value) {
            uint64_t count = checked_binom(n - value - 1, remaining);
            if (rank < count) {
                subset.push_back(value);
                start = value + 1;
                chosen = true;
                break;
            }
            rank -= count;
        }
        if (!chosen) {
            throw std::runtime_error("failed to unrank subset");
        }
    }
    return subset;
}

uint64_t subset_mask(const vector<size_t>& subset) {
    uint64_t mask = 0;
    for (size_t i : subset) {
        if (i >= 64) {
            throw std::runtime_error("subset mask supports only n < 64");
        }
        mask |= (1ULL << i);
    }
    return mask;
}

uint16_t subset_weight(const vector<size_t>& subset, const vector<uint16_t>& weights, size_t r) {
    size_t sum = 0;
    for (size_t i : subset) {
        sum += weights[i];
    }
    return static_cast<uint16_t>(sum % r);
}

uint32_t sub_mod(size_t a, size_t b, size_t r) {
    return static_cast<uint32_t>((a + r - (b % r)) % r);
}

string read_text_file(const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + filename);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

size_t skip_ws(const string& text, size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

size_t find_json_key_colon(const string& text, const string& key, size_t start = 0) {
    string needle = "\"" + key + "\"";
    size_t pos = text.find(needle, start);
    if (pos == string::npos) {
        throw std::runtime_error("missing JSON key '" + key + "'");
    }
    size_t colon = text.find(':', pos + needle.size());
    if (colon == string::npos) {
        throw std::runtime_error("missing ':' after JSON key '" + key + "'");
    }
    return colon;
}

size_t find_matching_json_delimiter(const string& text, size_t open_pos, char open, char close) {
    if (open_pos >= text.size() || text[open_pos] != open) {
        throw std::runtime_error("JSON delimiter search started at the wrong character");
    }

    size_t depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t pos = open_pos; pos < text.size(); ++pos) {
        char ch = text[pos];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == open) {
            ++depth;
        } else if (ch == close) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
    }
    throw std::runtime_error("failed to find matching JSON delimiter");
}

string extract_object_text(const string& text, const string& key) {
    size_t colon = find_json_key_colon(text, key);
    size_t start = text.find('{', colon);
    if (start == string::npos) {
        throw std::runtime_error("JSON key '" + key + "' is not an object");
    }
    size_t end = find_matching_json_delimiter(text, start, '{', '}');
    return text.substr(start, end - start + 1);
}

string extract_array_text(const string& text, const string& key) {
    size_t colon = find_json_key_colon(text, key);
    size_t start = text.find('[', colon);
    if (start == string::npos) {
        throw std::runtime_error("JSON key '" + key + "' is not an array");
    }
    size_t end = find_matching_json_delimiter(text, start, '[', ']');
    return text.substr(start, end - start + 1);
}

uint64_t parse_u64_at(const string& text, size_t pos, const string& label) {
    pos = skip_ws(text, pos);
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        throw std::runtime_error("expected unsigned integer for " + label);
    }
    uint64_t value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        uint32_t digit = static_cast<uint32_t>(text[pos] - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            throw std::runtime_error(label + " overflows u64");
        }
        value = value * 10 + digit;
        ++pos;
    }
    return value;
}

uint64_t extract_u64(const string& text, const string& key) {
    size_t colon = find_json_key_colon(text, key);
    return parse_u64_at(text, colon + 1, key);
}

vector<uint64_t> parse_u64_array_values(const string& array_text, const string& label) {
    vector<uint64_t> values;
    size_t pos = 0;
    while (pos < array_text.size()) {
        if (std::isdigit(static_cast<unsigned char>(array_text[pos]))) {
            uint64_t value = 0;
            while (pos < array_text.size() && std::isdigit(static_cast<unsigned char>(array_text[pos]))) {
                uint32_t digit = static_cast<uint32_t>(array_text[pos] - '0');
                if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                    throw std::runtime_error(label + " entry overflows u64");
                }
                value = value * 10 + digit;
                ++pos;
            }
            values.push_back(value);
        } else {
            ++pos;
        }
    }
    return values;
}

vector<uint16_t> parse_weight_array(const string& object_text, const string& key, size_t r) {
    vector<uint64_t> raw = parse_u64_array_values(extract_array_text(object_text, key), key);
    vector<uint16_t> out;
    out.reserve(raw.size());
    for (uint64_t value : raw) {
        if (value >= r || value > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("weight " + std::to_string(value) + " in " + key + " is outside 0..r-1");
        }
        out.push_back(static_cast<uint16_t>(value));
    }
    return out;
}

vector<my_int> parse_mu_values(const string& mu_text, uint32_t modulus) {
    vector<uint64_t> raw = parse_u64_array_values(extract_array_text(mu_text, "values"), "mu.values");
    vector<my_int> out;
    out.reserve(raw.size());
    for (uint64_t value : raw) {
        if (value >= modulus) {
            throw std::runtime_error("mu.values entry is not reduced modulo the instance modulus");
        }
        if (value > std::numeric_limits<my_int>::max()) {
            throw std::runtime_error("mu.values entry does not fit uint16_t CUDA storage");
        }
        out.push_back(static_cast<my_int>(value));
    }
    return out;
}

vector<my_int> parse_mod_array_values(
    const string& object_text,
    const string& key,
    size_t expected_len,
    uint32_t modulus
) {
    vector<uint64_t> raw = parse_u64_array_values(extract_array_text(object_text, key), key);
    if (raw.size() != expected_len) {
        throw std::runtime_error(key + " has " + std::to_string(raw.size())
            + " entries; expected " + std::to_string(expected_len));
    }
    vector<my_int> out;
    out.reserve(raw.size());
    for (uint64_t value : raw) {
        if (value >= modulus) {
            throw std::runtime_error(key + " entry is not reduced modulo the instance modulus");
        }
        if (value > std::numeric_limits<my_int>::max()) {
            throw std::runtime_error(key + " entry does not fit uint16_t CUDA storage");
        }
        out.push_back(static_cast<my_int>(value));
    }
    return out;
}

DenseBlockFile load_dense_block_file(const string& filename, uint32_t expected_prime) {
    string text = read_text_file(filename);
    uint64_t rows64 = extract_u64(text, "rows");
    uint64_t cols64 = extract_u64(text, "columns");
    uint64_t prime64 = extract_u64(text, "prime");
    if (rows64 > std::numeric_limits<size_t>::max()
            || cols64 > std::numeric_limits<size_t>::max()
            || prime64 > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("dense block scalar metadata is too large");
    }
    DenseBlockFile block;
    block.rows = static_cast<size_t>(rows64);
    block.columns = static_cast<size_t>(cols64);
    block.prime = static_cast<uint32_t>(prime64);
    if (block.prime != expected_prime) {
        throw std::runtime_error("dense block prime does not match instance modulus");
    }
    block.values = parse_mod_array_values(
        text,
        "values",
        checked_mul(block.rows, block.columns, "dense block value count"),
        expected_prime
    );
    return block;
}

vector<uint32_t> load_drop_columns_file(const string& filename, size_t expected_count, size_t base_columns) {
    string text = read_text_file(filename);
    vector<uint64_t> raw = parse_u64_array_values(extract_array_text(text, "indices"), "drop-column indices");
    if (raw.size() != expected_count) {
        throw std::runtime_error("drop-column file has the wrong number of indices");
    }
    vector<uint32_t> out;
    out.reserve(raw.size());
    for (uint64_t value : raw) {
        if (value >= base_columns || value > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("drop-column index is outside the base column range");
        }
        out.push_back(static_cast<uint32_t>(value));
    }
    std::sort(out.begin(), out.end());
    for (size_t i = 1; i < out.size(); ++i) {
        if (out[i] == out[i - 1]) {
            throw std::runtime_error("drop-column indices must be distinct");
        }
    }
    return out;
}

my_int mu_at(const CyclicInstance& instance, size_t i, size_t beta, size_t a) {
    return instance.mu[((i * instance.n + beta) * instance.genus) + a];
}

void validate_elimination_data(const CyclicInstance& instance) {
    if (!instance.has_elimination) {
        return;
    }
    if (instance.elimination_w_index >= instance.n) {
        throw std::runtime_error("elimination w_V_index is outside V");
    }
    if (instance.elimination_w_weight != instance.weight_v[instance.elimination_w_index]) {
        throw std::runtime_error("elimination w_weight does not match V weight");
    }
    if (instance.elimination_right_inverse_a0_by_a1.size() != instance.genus * instance.n) {
        throw std::runtime_error("elimination right inverse has wrong shape");
    }
    const size_t kernel_dim = instance.elimination_kernel_weights.size();
    if (kernel_dim == 0) {
        throw std::runtime_error("elimination kernel basis is empty");
    }
    if (instance.elimination_kernel_basis_a0_by_kernel.size() != instance.genus * kernel_dim) {
        throw std::runtime_error("elimination kernel basis has wrong shape");
    }
    for (uint16_t weight : instance.elimination_kernel_weights) {
        if (weight >= instance.r) {
            throw std::runtime_error("elimination kernel weight is outside 0..r-1");
        }
    }

    for (size_t beta = 0; beta < instance.n; ++beta) {
        for (size_t gamma = 0; gamma < instance.n; ++gamma) {
            uint32_t acc = 0;
            for (size_t a = 0; a < instance.genus; ++a) {
                uint32_t mu = static_cast<uint32_t>(mu_at(instance, instance.elimination_w_index, beta, a));
                uint32_t rcoeff = static_cast<uint32_t>(
                    instance.elimination_right_inverse_a0_by_a1[a * instance.n + gamma]
                );
                acc = (acc + mu * rcoeff) % instance.modulus;
            }
            uint32_t expected = beta == gamma ? 1u : 0u;
            if (acc != expected) {
                throw std::runtime_error("elimination right inverse check mu_w*R=I failed");
            }
        }
    }

    for (size_t beta = 0; beta < instance.n; ++beta) {
        for (size_t k = 0; k < kernel_dim; ++k) {
            uint32_t acc = 0;
            for (size_t a = 0; a < instance.genus; ++a) {
                uint32_t mu = static_cast<uint32_t>(mu_at(instance, instance.elimination_w_index, beta, a));
                uint32_t bcoeff = static_cast<uint32_t>(
                    instance.elimination_kernel_basis_a0_by_kernel[a * kernel_dim + k]
                );
                acc = (acc + mu * bcoeff) % instance.modulus;
            }
            if (acc != 0) {
                throw std::runtime_error("elimination kernel check mu_w*B=0 failed");
            }
        }
    }
}

CyclicInstance load_cyclic_instance(const string& filename) {
    string text = read_text_file(filename);

    uint64_t genus64 = extract_u64(text, "genus");
    uint64_t modulus64 = extract_u64(text, "modulus");
    string cyclic_text = extract_object_text(text, "cyclic");
    uint64_t r64 = extract_u64(cyclic_text, "order");
    if (genus64 > std::numeric_limits<size_t>::max()
            || r64 > std::numeric_limits<size_t>::max()
            || modulus64 > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("cyclic instance scalar metadata is too large");
    }

    size_t genus = static_cast<size_t>(genus64);
    size_t r = static_cast<size_t>(r64);
    uint32_t modulus = static_cast<uint32_t>(modulus64);
    if (modulus == 0 || modulus > std::numeric_limits<my_int>::max()) {
        throw std::runtime_error("instance modulus must fit uint16_t CUDA storage");
    }

    CyclicGeometry geom = compute_cyclic_geometry(genus, r);
    string weights_text = extract_object_text(text, "weights");
    vector<uint16_t> weight_v = parse_weight_array(weights_text, "V", r);
    vector<uint16_t> weight_a0 = parse_weight_array(weights_text, "A0", r);
    vector<uint16_t> weight_a1 = parse_weight_array(weights_text, "A1", r);
    if (weight_v.size() != geom.n || weight_a0.size() != geom.genus || weight_a1.size() != geom.n) {
        throw std::runtime_error("cyclic weight vector lengths do not match genus metadata");
    }

    string artinian_text = extract_object_text(text, "artinian");
    string mu_text = extract_object_text(artinian_text, "mu");
    uint64_t mu_n64 = extract_u64(mu_text, "n");
    uint64_t mu_g64 = extract_u64(mu_text, "g");
    vector<my_int> mu = parse_mu_values(mu_text, modulus);
    size_t expected_mu_len = checked_mul(checked_mul(geom.n, geom.n, "mu n*n"), geom.genus, "mu length");
    if (mu_n64 != geom.n || mu_g64 != geom.genus || mu.size() != expected_mu_len) {
        throw std::runtime_error("artinian.mu dimensions do not match cyclic genus metadata");
    }

    vector<uint64_t> expected_cols;
    vector<uint64_t> expected_rows;
    vector<uint64_t> expected_nnz;
    try {
        string koszul_text = extract_object_text(text, "koszul");
        expected_cols = parse_u64_array_values(extract_array_text(koszul_text, "sector_columns"), "koszul.sector_columns");
        expected_rows = parse_u64_array_values(extract_array_text(koszul_text, "sector_rows"), "koszul.sector_rows");
        expected_nnz = parse_u64_array_values(extract_array_text(koszul_text, "sector_nnz"), "koszul.sector_nnz");
    } catch (const std::exception&) {
        expected_cols.clear();
        expected_rows.clear();
        expected_nnz.clear();
    }

    bool has_elimination = false;
    size_t elimination_w_index = 0;
    uint16_t elimination_w_weight = 0;
    vector<my_int> elimination_right_inverse;
    vector<my_int> elimination_kernel_basis;
    vector<uint16_t> elimination_kernel_weights;
    vector<uint64_t> expected_eliminated_cols;
    vector<uint64_t> expected_eliminated_rows;
    string elimination_text;
    try {
        elimination_text = extract_object_text(text, "third_section_elimination");
        has_elimination = true;
    } catch (const std::exception&) {
        has_elimination = false;
    }
    if (has_elimination) {
        uint64_t w_index64 = extract_u64(elimination_text, "w_V_index");
        uint64_t w_weight64 = extract_u64(elimination_text, "w_weight");
        if (w_index64 > std::numeric_limits<size_t>::max()
                || w_weight64 >= r
                || w_weight64 > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("third_section_elimination scalar metadata is invalid");
        }
        elimination_w_index = static_cast<size_t>(w_index64);
        elimination_w_weight = static_cast<uint16_t>(w_weight64);
        elimination_kernel_weights = parse_weight_array(elimination_text, "kernel_weights", r);
        size_t kernel_dim = elimination_kernel_weights.size();
        elimination_right_inverse = parse_mod_array_values(
            elimination_text,
            "right_inverse_A0_by_A1",
            checked_mul(geom.genus, geom.n, "elimination right inverse length"),
            modulus
        );
        elimination_kernel_basis = parse_mod_array_values(
            elimination_text,
            "kernel_basis_A0_by_3",
            checked_mul(geom.genus, kernel_dim, "elimination kernel basis length"),
            modulus
        );
        expected_eliminated_cols = parse_u64_array_values(
            extract_array_text(elimination_text, "sector_columns"),
            "third_section_elimination.sector_columns"
        );
        expected_eliminated_rows = parse_u64_array_values(
            extract_array_text(elimination_text, "sector_rows"),
            "third_section_elimination.sector_rows"
        );
    }

    CyclicInstance instance{
        genus,
        geom.n,
        geom.m,
        r,
        modulus,
        std::move(mu),
        std::move(weight_v),
        std::move(weight_a0),
        std::move(weight_a1),
        std::move(expected_cols),
        std::move(expected_rows),
        std::move(expected_nnz),
        has_elimination,
        elimination_w_index,
        elimination_w_weight,
        std::move(elimination_right_inverse),
        std::move(elimination_kernel_basis),
        std::move(elimination_kernel_weights),
        std::move(expected_eliminated_cols),
        std::move(expected_eliminated_rows)
    };
    validate_elimination_data(instance);
    return instance;
}

CharacterMap build_character_map(const vector<uint16_t>& weights, size_t r, const string& label) {
    CharacterMap map;
    map.offsets.resize(r + 1);
    for (size_t ch = 0; ch < r; ++ch) {
        map.offsets[ch] = checked_u32_index(map.indices.size(), label + " character offset");
        for (size_t idx = 0; idx < weights.size(); ++idx) {
            if (weights[idx] == ch) {
                if (idx > std::numeric_limits<uint16_t>::max()) {
                    throw std::runtime_error(label + " basis index does not fit uint16_t");
                }
                map.indices.push_back(static_cast<uint16_t>(idx));
            }
        }
    }
    map.offsets[r] = checked_u32_index(map.indices.size(), label + " final character offset");
    return map;
}



CyclicCommonIncidence build_common_incidences_generic(
    size_t local_n,
    size_t source_degree,
    size_t r,
    const vector<size_t>& original_indices,
    const vector<uint16_t>& local_weights,
    const string& label
) {
    if (local_n >= 64) {
        throw std::runtime_error(label + " implicit operator supports local_n < 64");
    }
    if (source_degree == 0 || source_degree > local_n) {
        throw std::runtime_error(label + " source exterior degree is invalid");
    }
    if (original_indices.size() != local_n || local_weights.size() != local_n) {
        throw std::runtime_error(label + " local index metadata has the wrong length");
    }

    size_t target_degree = source_degree - 1;
    size_t row_subset_count = checked_size(checked_binom(local_n, target_degree), label + " row subset count");
    size_t column_subset_count = checked_size(checked_binom(local_n, source_degree), label + " column subset count");

    unordered_map<uint64_t, size_t> row_rank_by_mask;
    row_rank_by_mask.reserve(row_subset_count);
    vector<uint16_t> row_subset_weights(row_subset_count);
    for (size_t rank = 0; rank < row_subset_count; ++rank) {
        vector<size_t> subset = unrank_subset(rank, local_n, target_degree);
        row_rank_by_mask.emplace(subset_mask(subset), rank);
        row_subset_weights[rank] = subset_weight(subset, local_weights, r);
    }

    unordered_map<uint64_t, size_t> column_rank_by_mask;
    column_rank_by_mask.reserve(column_subset_count);
    vector<uint16_t> column_subset_weights(column_subset_count);
    for (size_t rank = 0; rank < column_subset_count; ++rank) {
        vector<size_t> subset = unrank_subset(rank, local_n, source_degree);
        column_rank_by_mask.emplace(subset_mask(subset), rank);
        column_subset_weights[rank] = subset_weight(subset, local_weights, r);
    }

    size_t row_width = local_n - target_degree;
    vector<CyclicRowIncidence> row_incidence;
    row_incidence.reserve(checked_mul(row_subset_count, row_width, label + " row incidence size"));
    for (size_t row_rank = 0; row_rank < row_subset_count; ++row_rank) {
        vector<size_t> subset = unrank_subset(row_rank, local_n, target_degree);
        uint64_t mask = subset_mask(subset);
        for (size_t local_i = 0; local_i < local_n; ++local_i) {
            if ((mask & (1ULL << local_i)) != 0) {
                continue;
            }
            size_t position = 0;
            while (position < subset.size() && subset[position] < local_i) {
                ++position;
            }
            uint64_t column_mask = mask | (1ULL << local_i);
            auto column_it = column_rank_by_mask.find(column_mask);
            if (column_it == column_rank_by_mask.end()) {
                throw std::runtime_error(label + " column subset rank lookup failed");
            }
            row_incidence.push_back(CyclicRowIncidence{
                checked_u32_index(original_indices[local_i], label + " row incidence i"),
                checked_u32_index(column_it->second, label + " row incidence column subset rank"),
                static_cast<uint8_t>(position % 2 == 1)
            });
        }
    }

    size_t column_width = source_degree;
    vector<CyclicColumnIncidence> column_incidence;
    column_incidence.reserve(checked_mul(column_subset_count, column_width, label + " column incidence size"));
    for (size_t column_rank = 0; column_rank < column_subset_count; ++column_rank) {
        vector<size_t> subset = unrank_subset(column_rank, local_n, source_degree);
        uint64_t mask = subset_mask(subset);
        for (size_t position = 0; position < subset.size(); ++position) {
            size_t local_i = subset[position];
            uint64_t row_mask = mask & ~(1ULL << local_i);
            auto row_it = row_rank_by_mask.find(row_mask);
            if (row_it == row_rank_by_mask.end()) {
                throw std::runtime_error(label + " row subset rank lookup failed");
            }
            column_incidence.push_back(CyclicColumnIncidence{
                checked_u32_index(original_indices[local_i], label + " column incidence i"),
                checked_u32_index(row_it->second, label + " column incidence row subset rank"),
                static_cast<uint8_t>(position % 2 == 1)
            });
        }
    }

    return CyclicCommonIncidence{
        std::move(row_incidence),
        std::move(column_incidence),
        std::move(row_subset_weights),
        std::move(column_subset_weights),
        row_width,
        column_width
    };
}





CyclicSectorLayout build_sector_layout_generic(
    const CyclicInstance& instance,
    const CyclicCommonIncidence& common,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    size_t sector,
    const string& label
) {
    CyclicSectorLayout layout;
    layout.sector = sector;
    layout.row_offsets.resize(common.row_subset_weights.size() + 1);
    layout.column_offsets.resize(common.column_subset_weights.size() + 1);

    for (size_t row_rank = 0; row_rank < common.row_subset_weights.size(); ++row_rank) {
        layout.row_offsets[row_rank] = checked_u32_index(layout.row_descriptors.size(), label + " row offset");
        uint32_t ch = sub_mod(sector, common.row_subset_weights[row_rank], instance.r);
        for (uint32_t idx = a1_map.offsets[ch]; idx < a1_map.offsets[ch + 1]; ++idx) {
            layout.row_descriptors.push_back(CyclicRowDescriptor{
                checked_u32_index(row_rank, label + " row subset rank"),
                a1_map.indices[idx],
                0
            });
        }
    }
    layout.row_offsets[common.row_subset_weights.size()] =
        checked_u32_index(layout.row_descriptors.size(), label + " final row offset");

    for (size_t column_rank = 0; column_rank < common.column_subset_weights.size(); ++column_rank) {
        layout.column_offsets[column_rank] = checked_u32_index(layout.column_descriptors.size(), label + " column offset");
        uint32_t ch = sub_mod(sector, common.column_subset_weights[column_rank], instance.r);
        for (uint32_t idx = a0_map.offsets[ch]; idx < a0_map.offsets[ch + 1]; ++idx) {
            layout.column_descriptors.push_back(CyclicColumnDescriptor{
                checked_u32_index(column_rank, label + " column subset rank"),
                a0_map.indices[idx],
                0
            });
        }
    }
    layout.column_offsets[common.column_subset_weights.size()] =
        checked_u32_index(layout.column_descriptors.size(), label + " final column offset");

    uint64_t nnz = 0;
    for (const CyclicRowDescriptor& row_desc : layout.row_descriptors) {
        const CyclicRowIncidence* incidences =
            common.row_incidence.data() + static_cast<size_t>(row_desc.subset_rank) * common.row_width;
        uint32_t beta = row_desc.beta;
        uint32_t beta_weight = instance.weight_a1[beta];
        for (size_t pos = 0; pos < common.row_width; ++pos) {
            const CyclicRowIncidence& incidence = incidences[pos];
            uint32_t q = sub_mod(beta_weight, instance.weight_v[incidence.i], instance.r);
            for (uint32_t idx = a0_map.offsets[q]; idx < a0_map.offsets[q + 1]; ++idx) {
                uint32_t a = a0_map.indices[idx];
                my_int raw = mu_at(instance, incidence.i, beta, a);
                if (raw != 0) {
                    ++nnz;
                }
            }
        }
    }
    layout.nnz = nnz;
    return layout;
}

CyclicTensorLayout build_tensor_layout(
    const vector<uint16_t>& subset_weights,
    const CharacterMap& coeff_map,
    size_t r,
    size_t sector,
    const string& label
) {
    CyclicTensorLayout layout;
    layout.sector = sector;
    layout.offsets.resize(subset_weights.size() + 1);
    for (size_t subset_rank = 0; subset_rank < subset_weights.size(); ++subset_rank) {
        layout.offsets[subset_rank] = checked_u32_index(layout.descriptors.size(), label + " offset");
        uint32_t ch = sub_mod(sector, subset_weights[subset_rank], r);
        for (uint32_t idx = coeff_map.offsets[ch]; idx < coeff_map.offsets[ch + 1]; ++idx) {
            layout.descriptors.push_back(CyclicColumnDescriptor{
                checked_u32_index(subset_rank, label + " subset rank"),
                coeff_map.indices[idx],
                0
            });
        }
    }
    layout.offsets[subset_weights.size()] = checked_u32_index(layout.descriptors.size(), label + " final offset");
    return layout;
}

vector<my_int> ones_vector(size_t len) {
    return vector<my_int>(len, 1);
}

CyclicPhiHostData<my_int> make_operator_host_data(
    const CyclicInstance& instance,
    const CyclicCommonIncidence& common,
    CyclicSectorLayout&& layout,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    vector<my_int> row_precond,
    vector<my_int> col_precond
) {
    CyclicPhiHostData<my_int> host;
    host.g = instance.genus;
    host.n = instance.n;
    host.m = instance.m;
    host.r = instance.r;
    host.numRows = layout.row_descriptors.size();
    host.numCols = layout.column_descriptors.size();
    host.rowSubsetCount = common.row_subset_weights.size();
    host.columnSubsetCount = common.column_subset_weights.size();
    host.rowWidth = common.row_width;
    host.columnWidth = common.column_width;
    host.mu = instance.mu;
    host.rowPrecond = std::move(row_precond);
    host.colPrecond = std::move(col_precond);
    host.weightV = instance.weight_v;
    host.weightA0 = instance.weight_a0;
    host.weightA1 = instance.weight_a1;
    host.a0CharOffsets = a0_map.offsets;
    host.a0ByChar = a0_map.indices;
    host.a1CharOffsets = a1_map.offsets;
    host.a1ByChar = a1_map.indices;
    host.rowOffsets = std::move(layout.row_offsets);
    host.rowDescriptors = std::move(layout.row_descriptors);
    host.columnOffsets = std::move(layout.column_offsets);
    host.columnDescriptors = std::move(layout.column_descriptors);
    host.rowIncidence = common.row_incidence;
    host.columnIncidence = common.column_incidence;
    return host;
}

template <typename T>
void flatten_vector_list(
    const vector<vector<T>>& list,
    size_t expected_vectors,
    size_t expected_len,
    vector<T>& out,
    const string& label
) {
    if (list.size() != expected_vectors) {
        throw std::runtime_error(label + " vector count does not match WDM header");
    }
    out.resize(checked_mul(expected_len, expected_vectors, label + " flattened length"));
    for (size_t j = 0; j < expected_vectors; ++j) {
        if (list[j].size() != expected_len) {
            throw std::runtime_error(label + " vector length does not match WDM header");
        }
        for (size_t i = 0; i < expected_len; ++i) {
            out[i * expected_vectors + j] = list[j][i];
        }
    }
}

inline void compute_and_push_bigsp2(
    CudaDenseMatrix<my_int> &B,
    CudaDenseMatrix<my_int> &C,
    my_int* dBigSp,
    accum_int* dGramScratch,
    size_t &seq_position,
    ModulusParams mod,
    size_t gram_chunk_size
) {
    B.mTm_tri_pack_from_scratch(C, dBigSp, dGramScratch, seq_position, mod, gram_chunk_size);
    seq_position++;
}

template <typename T>
T* copy_vector_to_device(const vector<T>& host, const string& label) {
    if (host.empty()) {
        throw std::runtime_error(label + " is empty");
    }
    T* device = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

struct DeviceRowMixRound {
    CyclicRowMixPair* d_pairs = nullptr;
    size_t pair_count = 0;
};

struct DeviceRowMixPlan {
    vector<DeviceRowMixRound> rounds;

    void release() {
        for (DeviceRowMixRound& round : rounds) {
            if (round.d_pairs != nullptr) {
                CHECK_CUDA(cudaFree(round.d_pairs));
                round.d_pairs = nullptr;
            }
            round.pair_count = 0;
        }
        rounds.clear();
    }
};

DeviceRowMixPlan copy_rowmix_plan_to_device(const RowMixPlan& plan) {
    DeviceRowMixPlan device_plan;
    device_plan.rounds.resize(plan.round_pairs.size());
    for (size_t round_index = 0; round_index < plan.round_pairs.size(); ++round_index) {
        const vector<CyclicRowMixPair>& pairs = plan.round_pairs[round_index];
        device_plan.rounds[round_index].pair_count = pairs.size();
        if (!pairs.empty()) {
            device_plan.rounds[round_index].d_pairs =
                copy_vector_to_device(pairs, "rowmix pairs round " + std::to_string(round_index));
        }
    }
    return device_plan;
}

void apply_rowmix_forward_device(
    CudaDenseMatrix<my_int>& block,
    const DeviceRowMixPlan& plan,
    ModulusParams mod
) {
    for (const DeviceRowMixRound& round : plan.rounds) {
        cyclic_launch_rowmix_forward(block, round.d_pairs, round.pair_count, mod);
    }
}

void apply_rowmix_transpose_device(
    CudaDenseMatrix<my_int>& block,
    const DeviceRowMixPlan& plan,
    ModulusParams mod
) {
    for (size_t round_index = plan.rounds.size(); round_index > 0; --round_index) {
        const DeviceRowMixRound& round = plan.rounds[round_index - 1];
        cyclic_launch_rowmix_transpose(block, round.d_pairs, round.pair_count, mod);
    }
}

void save_all_data(
    const std::string& filename,
    size_t n_rows,
    size_t n_cols,
    size_t n_dense_cols,
    my_int theprime,
    const std::vector<my_int>& row_precond,
    const std::vector<my_int>& col_precond,
    const std::vector<my_int>& initial_B,
    const my_int* dB,
    const my_int* dBigSp,
    size_t seq_position
) {
    std::cout << "Saving WDM state to " << filename << "..." << std::endl;

    std::vector<my_int> hB(n_cols * n_dense_cols);
    CHECK_CUDA(cudaMemcpy(hB.data(), dB, n_cols * n_dense_cols * sizeof(my_int), cudaMemcpyDeviceToHost));

    size_t seq_entry_size = upper_triangular_size(n_dense_cols);
    size_t effective_size = seq_position * seq_entry_size;
    std::vector<my_int> hBigSp(effective_size);
    CHECK_CUDA(cudaMemcpy(hBigSp.data(), dBigSp, effective_size * sizeof(my_int), cudaMemcpyDeviceToHost));

    std::vector<std::vector<my_int>> cur_B = reshape_to_vector_of_vectors(hB, n_cols);
    std::vector<std::vector<my_int>> ini_B = reshape_to_vector_of_vectors(initial_B, n_cols);
    std::vector<std::vector<my_int>> sp_list_upper = reshape_to_vector_of_vectors(hBigSp, seq_position);

    string filename_tmp = filename + ".tmp";
    save_wdm_file_sym(
        filename_tmp,
        n_rows,
        n_cols,
        theprime,
        row_precond,
        col_precond,
        ini_B,
        cur_B,
        sp_list_upper
    );

    std::filesystem::rename(filename_tmp, filename);
    std::cout << "Saved " << seq_position << " sequence elements to " << filename << std::endl;
}

void report_progress(
    const long long elapsed,
    long long& last_report,
    size_t& last_nlen,
    const size_t nlen,
    const size_t max_nlen,
    const size_t num_v,
    const std::string& suffix = ""
) {
    double seconds = static_cast<double>(elapsed - last_report) / 1000.0;
    double speed = seconds > 0.0 ? static_cast<double>(nlen - last_nlen) / seconds : 0.0;
    double remaining = speed > 0.0 ? static_cast<double>(max_nlen - nlen) / speed : 0.0;

    std::cout << "\rProgress: " << nlen << "/" << max_nlen
              << " | Elapsed: " << (elapsed / 1000) << "s"
              << " | Throughput: " << std::fixed << std::setprecision(2) << speed
              << "/s (total " << speed * num_v << "/s)"
              << " | Remaining: " << std::setprecision(1) << remaining << "s"
              << " | " << suffix << "           " << std::flush;

    last_nlen = nlen;
    last_report = elapsed;
}

double to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

size_t max_safe_gram_chunk_size(uint32_t prime) {
    if (prime == 0) {
        return 0;
    }
    uint64_t p64 = prime;
    uint64_t product_bound = p64 * p64;
    uint64_t accum_max = std::numeric_limits<accum_int>::max() - 1ULL;
    uint64_t chunk = accum_max / product_bound;
    while (chunk > 0 && chunk * product_bound >= accum_max) {
        --chunk;
    }
    return static_cast<size_t>(chunk);
}

size_t max_character_block_size(const CharacterMap& map) {
    size_t max_size = 0;
    for (size_t ch = 0; ch + 1 < map.offsets.size(); ++ch) {
        max_size = std::max<size_t>(max_size, map.offsets[ch + 1] - map.offsets[ch]);
    }
    return max_size;
}

uint64_t checked_mul_u64(uint64_t a, uint64_t b, const string& label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(label + " overflows uint64_t");
    }
    return a * b;
}

void validate_accumulation_terms(uint32_t prime, uint64_t terms, const string& label) {
    uint64_t p64 = prime;
    uint64_t accum_max = std::numeric_limits<accum_int>::max() - 1ULL;
    __uint128_t worst = static_cast<__uint128_t>(terms) * p64 * p64;
    if (terms == 0 || worst >= accum_max) {
        throw std::runtime_error(
            label + " can accumulate " + std::to_string(terms)
            + " products, which is not safe for uint32_t accumulators at p="
            + std::to_string(prime)
        );
    }
    std::cout << label << " max product terms per output: " << terms << std::endl;
}



uint64_t eliminated_accumulation_terms(
    const CyclicCommonIncidence& common_dm,
    const CyclicCommonIncidence& common_d2,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    const CharacterMap& kernel_map
) {
    uint64_t max_a0 = max_character_block_size(a0_map);
    uint64_t max_a1 = max_character_block_size(a1_map);
    uint64_t max_kernel = max_character_block_size(kernel_map);
    uint64_t dm_forward = checked_mul_u64(
        common_dm.row_width,
        max_a0,
        "eliminated D_m forward accumulation term count"
    );
    uint64_t dm_transpose = checked_mul_u64(
        common_dm.column_width,
        max_a1,
        "eliminated D_m transpose accumulation term count"
    );
    uint64_t d2_forward = checked_mul_u64(
        common_d2.row_width,
        max_a0,
        "eliminated D_(m-1) forward accumulation term count"
    );
    uint64_t d2_transpose = checked_mul_u64(
        common_d2.column_width,
        max_a1,
        "eliminated D_(m-1) transpose accumulation term count"
    );
    return std::max({
        dm_forward,
        dm_transpose,
        d2_forward,
        d2_transpose,
        max_a0,
        max_a1,
        max_kernel
    });
}



void validate_eliminated_sector_against_fixture(
    const CyclicInstance& instance,
    size_t sector,
    size_t columns,
    size_t rows
) {
    if (!instance.expected_eliminated_sector_columns.empty()
            && sector < instance.expected_eliminated_sector_columns.size()
            && instance.expected_eliminated_sector_columns[sector] != columns) {
        throw std::runtime_error("eliminated sector column count does not match fixture metadata");
    }
    if (!instance.expected_eliminated_sector_rows.empty()
            && sector < instance.expected_eliminated_sector_rows.size()
            && instance.expected_eliminated_sector_rows[sector] != rows) {
        throw std::runtime_error("eliminated sector row count does not match fixture metadata");
    }
}

uint64_t seeded_sector_value(uint64_t seed, size_t sector) {
    uint64_t x = seed ^ (0x9E3779B97F4A7C15ULL + (static_cast<uint64_t>(sector) << 6) + (static_cast<uint64_t>(sector) >> 2));
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

uint64_t fnv1a64_append_u8(uint64_t hash, uint8_t value) {
    hash ^= value;
    hash *= FNV1A64_PRIME;
    return hash;
}

uint64_t fnv1a64_append_u64(uint64_t hash, uint64_t value) {
    for (size_t byte = 0; byte < 8; ++byte) {
        hash = fnv1a64_append_u8(hash, static_cast<uint8_t>((value >> (8 * byte)) & 0xffu));
    }
    return hash;
}

uint64_t fnv1a64_append_string(uint64_t hash, const string& value) {
    for (unsigned char ch : value) {
        hash = fnv1a64_append_u8(hash, ch);
    }
    return hash;
}

template<typename T>
uint64_t fnv1a64_vector_hash(const vector<T>& values) {
    uint64_t hash = FNV1A64_OFFSET;
    hash = fnv1a64_append_u64(hash, values.size());
    for (T value : values) {
        hash = fnv1a64_append_u64(hash, static_cast<uint64_t>(value));
    }
    return hash;
}

string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

uint64_t rowmix_realized_seed(uint64_t seed, const string& op, size_t sector, size_t target_rows, uint32_t prime) {
    uint64_t hash = FNV1A64_OFFSET;
    hash = fnv1a64_append_string(hash, ROWMIX_ALGORITHM_VERSION);
    hash = fnv1a64_append_string(hash, op);
    hash = fnv1a64_append_u64(hash, seed);
    hash = fnv1a64_append_u64(hash, sector);
    hash = fnv1a64_append_u64(hash, target_rows);
    hash = fnv1a64_append_u64(hash, prime);
    return seeded_sector_value(hash, sector);
}

uint64_t rowmix_bounded(uint64_t& rng_state, uint64_t bound) {
    if (bound == 0) {
        throw std::runtime_error("rowmix bounded RNG called with zero bound");
    }
    uint64_t threshold = (0ULL - bound) % bound;
    while (true) {
        uint64_t value = splitmix64_next(rng_state);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

my_int rowmix_nonzero(uint64_t& rng_state, uint32_t prime) {
    if (prime <= 1 || prime > std::numeric_limits<my_int>::max()) {
        throw std::runtime_error("rowmix prime must be in 2..65535");
    }
    return static_cast<my_int>(1 + rowmix_bounded(rng_state, prime - 1));
}

RowMixPlan generate_rowmix_plan(
    size_t target_rows,
    const string& op,
    size_t sector,
    uint32_t prime,
    RowMixOptions options
) {
    RowMixPlan plan;
    plan.target_rows = target_rows;
    plan.rounds = options.rounds;
    plan.seed = options.seed;
    plan.realized_seed = rowmix_realized_seed(options.seed, op, sector, target_rows, prime);
    plan.round_pairs.resize(options.rounds);
    if (options.rounds == 0) {
        return plan;
    }

    vector<uint32_t> permutation(target_rows);
    for (size_t i = 0; i < target_rows; ++i) {
        permutation[i] = checked_u32_index(i, "rowmix target coordinate");
    }

    uint64_t rng_state = plan.realized_seed;
    size_t pair_count = target_rows / 2;
    for (size_t round = 0; round < options.rounds; ++round) {
        for (size_t i = target_rows; i > 1; --i) {
            size_t j = static_cast<size_t>(rowmix_bounded(rng_state, i));
            std::swap(permutation[i - 1], permutation[j]);
        }
        vector<CyclicRowMixPair>& pairs = plan.round_pairs[round];
        pairs.reserve(pair_count);
        for (size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
            pairs.push_back(CyclicRowMixPair{
                permutation[2 * pair_index],
                permutation[2 * pair_index + 1],
                static_cast<uint16_t>(rowmix_nonzero(rng_state, prime)),
                static_cast<uint16_t>(rowmix_nonzero(rng_state, prime))
            });
        }
    }
    return plan;
}

uint64_t rowmix_plan_hash(const RowMixPlan& plan) {
    uint64_t hash = FNV1A64_OFFSET;
    hash = fnv1a64_append_string(hash, ROWMIX_ALGORITHM_VERSION);
    hash = fnv1a64_append_u64(hash, plan.target_rows);
    hash = fnv1a64_append_u64(hash, plan.rounds);
    hash = fnv1a64_append_u64(hash, plan.seed);
    hash = fnv1a64_append_u64(hash, plan.realized_seed);
    for (const vector<CyclicRowMixPair>& round : plan.round_pairs) {
        hash = fnv1a64_append_u64(hash, round.size());
        for (const CyclicRowMixPair& pair : round) {
            hash = fnv1a64_append_u64(hash, pair.first);
            hash = fnv1a64_append_u64(hash, pair.second);
            hash = fnv1a64_append_u64(hash, pair.a);
            hash = fnv1a64_append_u64(hash, pair.b);
        }
    }
    return hash;
}

string rowmix_label(const RowMixOptions& options) {
    std::ostringstream out;
    out << "rowmix-r" << options.rounds << "-s" << options.seed;
    return out.str();
}

string sector_filename(
    const string& out_dir,
    const string& op,
    size_t sector,
    size_t num_v,
    const RowMixOptions& rowmix
) {
    std::filesystem::path dir(out_dir.empty() ? "." : out_dir);
    std::filesystem::create_directories(dir);
    std::ostringstream name;
    name << op << "-sector" << std::setw(2) << std::setfill('0') << sector
         << "-v" << num_v;
    if (rowmix.rounds > 0) {
        name << "-rowmix-r" << rowmix.rounds << "-s" << rowmix.seed;
    }
    name << ".wdm.zst";
    return (dir / name.str()).string();
}

string rowmix_sidecar_filename(const string& wdm_filename) {
    string suffix = ".wdm.zst";
    if (wdm_filename.size() >= suffix.size()
            && wdm_filename.compare(wdm_filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return wdm_filename.substr(0, wdm_filename.size() - suffix.size()) + ".rowmix.json";
    }
    return wdm_filename + ".rowmix.json";
}

void ensure_output_absent(const string& wdm_filename, bool no_save) {
    if (no_save) {
        return;
    }
    if (std::filesystem::exists(wdm_filename)) {
        throw std::runtime_error("refusing to overwrite existing WDM file " + wdm_filename);
    }
    string sidecar = rowmix_sidecar_filename(wdm_filename);
    if (std::filesystem::exists(sidecar)) {
        throw std::runtime_error("refusing to reuse existing rowmix sidecar " + sidecar);
    }
}

void write_rowmix_sidecar(
    const string& wdm_filename,
    const string& op,
    size_t sector,
    size_t rows,
    size_t cols,
    size_t num_v,
    uint32_t prime,
    uint64_t random_seed,
    const RowMixPlan& plan,
    const vector<my_int>& row_precond,
    const vector<my_int>& col_precond,
    const vector<my_int>& initial_B
) {
    string sidecar = rowmix_sidecar_filename(wdm_filename);
    string tmp = sidecar + ".tmp";
    std::ofstream out(tmp, ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open rowmix sidecar " + tmp);
    }
    vector<size_t> pair_counts;
    pair_counts.reserve(plan.round_pairs.size());
    for (const auto& round : plan.round_pairs) {
        pair_counts.push_back(round.size());
    }

    out << "{\n";
    out << "  \"type\": \"prym_cyclic_rowmix_sidecar\",\n";
    out << "  \"version\": 1,\n";
    out << "  \"algorithm_version\": \"" << ROWMIX_ALGORITHM_VERSION << "\",\n";
    out << "  \"wdm_file\": \"" << wdm_filename << "\",\n";
    out << "  \"operator\": \"" << op << "\",\n";
    out << "  \"sector\": " << sector << ",\n";
    out << "  \"rows\": " << rows << ",\n";
    out << "  \"columns\": " << cols << ",\n";
    out << "  \"vector_count\": " << num_v << ",\n";
    out << "  \"prime\": " << prime << ",\n";
    out << "  \"numerical_seed\": " << random_seed << ",\n";
    out << "  \"rowmix\": {\n";
    out << "    \"rounds\": " << plan.rounds << ",\n";
    out << "    \"seed\": " << plan.seed << ",\n";
    out << "    \"realized_seed\": " << plan.realized_seed << ",\n";
    out << "    \"rng\": \"splitmix64-v1\",\n";
    out << "    \"target_rows\": " << plan.target_rows << ",\n";
    out << "    \"pair_count_per_round\": [";
    for (size_t i = 0; i < pair_counts.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << pair_counts[i];
    }
    out << "],\n";
    out << "    \"recipe\": \"Starting from realized_seed, for each round reset permutation to the previous round's final order, run descending Fisher-Yates using splitmix64 rejection-bounded draws, pair consecutive coordinates, then draw nonzero a,b in F_p by rejection-bounded draws in 1..p-1.\",\n";
    out << "    \"hash_fnv1a64\": \"" << hex_u64(rowmix_plan_hash(plan)) << "\"\n";
    out << "  },\n";
    out << "  \"hashes\": {\n";
    out << "    \"row_preconditioner_fnv1a64\": \"" << hex_u64(fnv1a64_vector_hash(row_precond)) << "\",\n";
    out << "    \"column_preconditioner_fnv1a64\": \"" << hex_u64(fnv1a64_vector_hash(col_precond)) << "\",\n";
    out << "    \"initial_block_fnv1a64\": \"" << hex_u64(fnv1a64_vector_hash(initial_B)) << "\"\n";
    out << "  }\n";
    out << "}\n";
    out.close();
    std::filesystem::rename(tmp, sidecar);
    std::cout << "Wrote rowmix sidecar to " << sidecar << std::endl;
}

uint32_t cpu_mod_u64(uint64_t value, uint32_t prime) {
    return static_cast<uint32_t>(value % static_cast<uint64_t>(prime));
}

uint32_t cpu_mul_mod(uint32_t a, uint32_t b, uint32_t prime) {
    return cpu_mod_u64(static_cast<uint64_t>(a) * static_cast<uint64_t>(b), prime);
}

vector<my_int> deterministic_dense_block(size_t rows, size_t cols, uint32_t prime, uint64_t salt) {
    vector<my_int> out(checked_mul(rows, cols, "deterministic dense block"));
    uint64_t state = salt ^ (0xD6E8FEB86659FD93ULL + rows * 1315423911ULL + cols * 2654435761ULL);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            uint64_t value = splitmix64_next(state);
            out[row * cols + col] = static_cast<my_int>(value % prime);
        }
    }
    return out;
}

vector<my_int> copy_cuda_dense_to_host(const CudaDenseMatrix<my_int>& matrix) {
    vector<my_int> host(checked_mul(matrix.numRows, matrix.numCols, "CUDA dense host copy"));
    CHECK_CUDA(cudaMemcpy(
        host.data(),
        matrix.d_data,
        host.size() * sizeof(my_int),
        cudaMemcpyDeviceToHost
    ));
    return host;
}

void compare_dense_exact(
    const vector<my_int>& expected,
    const vector<my_int>& actual,
    size_t rows,
    size_t cols,
    const string& label
) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(label + " dense comparison size mismatch");
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (expected[index] == actual[index]) {
            continue;
        }
        size_t row = index / cols;
        size_t col = index % cols;
        std::ostringstream error;
        error << label << " mismatch at (" << row << "," << col
              << "): CPU=" << static_cast<uint32_t>(expected[index])
              << ", CUDA=" << static_cast<uint32_t>(actual[index]);
        throw std::runtime_error(error.str());
    }
    std::cout << "Validation matched: " << label << " (" << rows << " x " << cols << ")" << std::endl;
}

vector<my_int> slice_row_block(
    const vector<my_int>& input,
    size_t start_row,
    size_t rows,
    size_t cols
) {
    vector<my_int> out(checked_mul(rows, cols, "dense row slice"));
    for (size_t row = 0; row < rows; ++row) {
        std::copy_n(
            input.data() + (start_row + row) * cols,
            cols,
            out.data() + row * cols
        );
    }
    return out;
}

uint32_t dense_dot_mod(
    const vector<my_int>& left,
    const vector<my_int>& right,
    uint32_t prime
) {
    if (left.size() != right.size()) {
        throw std::runtime_error("dense dot input size mismatch");
    }
    uint64_t sum = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        sum += static_cast<uint64_t>(left[i]) * static_cast<uint64_t>(right[i]);
        if (sum > (1ULL << 62)) {
            sum %= prime;
        }
    }
    return cpu_mod_u64(sum, prime);
}

void cpu_scale_rows_dense(
    const vector<my_int>& input,
    const vector<my_int>& scale,
    size_t rows,
    size_t cols,
    uint32_t prime,
    vector<my_int>& output
) {
    if (input.size() != rows * cols || scale.size() != rows) {
        throw std::runtime_error("CPU dense row scaling dimensions do not match");
    }
    output.resize(input.size());
    for (size_t row = 0; row < rows; ++row) {
        uint32_t factor = scale[row];
        for (size_t col = 0; col < cols; ++col) {
            output[row * cols + col] = static_cast<my_int>(
                cpu_mul_mod(input[row * cols + col], factor, prime)
            );
        }
    }
}

void cpu_apply_rowmix_forward(
    const RowMixPlan& plan,
    vector<my_int>& block,
    size_t dense_cols,
    uint32_t prime
) {
    if (block.size() != plan.target_rows * dense_cols) {
        throw std::runtime_error("CPU rowmix forward dimensions do not match");
    }
    for (const vector<CyclicRowMixPair>& round : plan.round_pairs) {
        for (const CyclicRowMixPair& pair : round) {
            for (size_t col = 0; col < dense_cols; ++col) {
                size_t first_index = static_cast<size_t>(pair.first) * dense_cols + col;
                size_t second_index = static_cast<size_t>(pair.second) * dense_cols + col;
                uint64_t x = block[first_index];
                uint64_t y = block[second_index];
                uint64_t a = pair.a;
                uint64_t b = pair.b;
                uint64_t one_plus_ab = 1 + a * b;
                block[first_index] = static_cast<my_int>(cpu_mod_u64(x + a * y, prime));
                block[second_index] = static_cast<my_int>(cpu_mod_u64(b * x + one_plus_ab * y, prime));
            }
        }
    }
}

void cpu_apply_rowmix_transpose(
    const RowMixPlan& plan,
    vector<my_int>& block,
    size_t dense_cols,
    uint32_t prime
) {
    if (block.size() != plan.target_rows * dense_cols) {
        throw std::runtime_error("CPU rowmix transpose dimensions do not match");
    }
    for (size_t round_index = plan.round_pairs.size(); round_index > 0; --round_index) {
        const vector<CyclicRowMixPair>& round = plan.round_pairs[round_index - 1];
        for (const CyclicRowMixPair& pair : round) {
            for (size_t col = 0; col < dense_cols; ++col) {
                size_t first_index = static_cast<size_t>(pair.first) * dense_cols + col;
                size_t second_index = static_cast<size_t>(pair.second) * dense_cols + col;
                uint64_t x = block[first_index];
                uint64_t y = block[second_index];
                uint64_t a = pair.a;
                uint64_t b = pair.b;
                uint64_t one_plus_ab = 1 + a * b;
                block[first_index] = static_cast<my_int>(cpu_mod_u64(x + b * y, prime));
                block[second_index] = static_cast<my_int>(cpu_mod_u64(a * x + one_plus_ab * y, prime));
            }
        }
    }
}

void validate_rowmix_transpose_identity(
    const RowMixPlan& plan,
    size_t dense_cols,
    uint32_t prime,
    const string& label
) {
    vector<my_int> x = deterministic_dense_block(plan.target_rows, dense_cols, prime, 0xA11CE001ULL);
    vector<my_int> y = deterministic_dense_block(plan.target_rows, dense_cols, prime, 0xB0B5EED2ULL);
    vector<my_int> tx = x;
    vector<my_int> tty = y;
    cpu_apply_rowmix_forward(plan, tx, dense_cols, prime);
    cpu_apply_rowmix_transpose(plan, tty, dense_cols, prime);
    uint32_t left = dense_dot_mod(tx, y, prime);
    uint32_t right = dense_dot_mod(x, tty, prime);
    if (left != right) {
        std::ostringstream error;
        error << label << " rowmix transpose identity failed: <Tx,y>="
              << left << ", <x,T^t y>=" << right;
        throw std::runtime_error(error.str());
    }
    std::cout << "Validation matched: " << label << " rowmix transpose identity" << std::endl;
}

void cpu_phi_forward_scaled_columns(
    const CyclicPhiHostData<my_int>& host,
    const vector<my_int>& scaled_B,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& C
) {
    if (scaled_B.size() != host.numCols * dense_cols) {
        throw std::runtime_error("CPU cyclic Phi forward dimensions do not match input block");
    }
    C.assign(checked_mul(host.numRows, dense_cols, "CPU cyclic Phi forward output"), 0);
    for (size_t row = 0; row < host.numRows; ++row) {
        const CyclicRowDescriptor& desc = host.rowDescriptors[row];
        uint32_t beta = desc.beta;
        uint32_t beta_weight = host.weightA1[beta];
        const CyclicRowIncidence* incidences =
            host.rowIncidence.data() + static_cast<size_t>(desc.subset_rank) * host.rowWidth;
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = 0;
            for (size_t pos = 0; pos < host.rowWidth; ++pos) {
                const CyclicRowIncidence& incidence = incidences[pos];
                uint32_t q = sub_mod(beta_weight, host.weightV[incidence.i], host.r);
                uint32_t start = host.a0CharOffsets[q];
                uint32_t end = host.a0CharOffsets[q + 1];
                uint32_t column_base = host.columnOffsets[incidence.column_subset_rank];
                for (uint32_t idx = start; idx < end; ++idx) {
                    uint32_t local_a = idx - start;
                    uint32_t a = host.a0ByChar[idx];
                    uint32_t raw = static_cast<uint32_t>(
                        host.mu[((static_cast<size_t>(incidence.i) * host.n + beta) * host.g) + a]
                    );
                    if (raw == 0) {
                        continue;
                    }
                    uint32_t value = scaled_B[(static_cast<size_t>(column_base) + local_a) * dense_cols + dense_col];
                    uint32_t coeff = incidence.negative ? prime - raw : raw;
                    sum += static_cast<uint64_t>(coeff) * value;
                }
            }
            uint32_t reduced = cpu_mod_u64(sum, prime);
            C[row * dense_cols + dense_col] = static_cast<my_int>(
                cpu_mul_mod(reduced, host.rowPrecond[row], prime)
            );
        }
    }
}

void cpu_phi_transpose(
    const CyclicPhiHostData<my_int>& host,
    const vector<my_int>& B,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& C
) {
    if (B.size() != host.numRows * dense_cols) {
        throw std::runtime_error("CPU cyclic Phi transpose dimensions do not match input block");
    }
    C.assign(checked_mul(host.numCols, dense_cols, "CPU cyclic Phi transpose output"), 0);
    for (size_t col = 0; col < host.numCols; ++col) {
        const CyclicColumnDescriptor& desc = host.columnDescriptors[col];
        uint32_t a = desc.a;
        uint32_t a_weight = host.weightA0[a];
        const CyclicColumnIncidence* incidences =
            host.columnIncidence.data() + static_cast<size_t>(desc.subset_rank) * host.columnWidth;
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = 0;
            for (size_t pos = 0; pos < host.columnWidth; ++pos) {
                const CyclicColumnIncidence& incidence = incidences[pos];
                uint32_t q = cyclic_add_mod(a_weight, host.weightV[incidence.i], static_cast<uint32_t>(host.r));
                uint32_t start = host.a1CharOffsets[q];
                uint32_t end = host.a1CharOffsets[q + 1];
                uint32_t row_base = host.rowOffsets[incidence.row_subset_rank];
                for (uint32_t idx = start; idx < end; ++idx) {
                    uint32_t local_beta = idx - start;
                    uint32_t beta = host.a1ByChar[idx];
                    uint32_t raw = static_cast<uint32_t>(
                        host.mu[((static_cast<size_t>(incidence.i) * host.n + beta) * host.g) + a]
                    );
                    if (raw == 0) {
                        continue;
                    }
                    uint32_t value = B[(static_cast<size_t>(row_base) + local_beta) * dense_cols + dense_col];
                    uint32_t coeff = incidence.negative ? prime - raw : raw;
                    sum += static_cast<uint64_t>(coeff) * value;
                }
            }
            uint32_t reduced = cpu_mod_u64(sum, prime);
            C[col * dense_cols + dense_col] = static_cast<my_int>(
                cpu_mul_mod(reduced, host.colPrecond[col], prime)
            );
        }
    }
}

void cpu_apply_target_preconditioner(
    const RowMixPlan& rowmix_plan,
    const vector<my_int>& row_precond,
    size_t rows,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& target_block
) {
    if (target_block.size() != rows * dense_cols || row_precond.size() != rows) {
        throw std::runtime_error("CPU mixed target preconditioner dimensions do not match");
    }
    cpu_apply_rowmix_forward(rowmix_plan, target_block, dense_cols, prime);
    vector<my_int> scaled;
    cpu_scale_rows_dense(target_block, row_precond, rows, dense_cols, prime, scaled);
    cpu_apply_rowmix_transpose(rowmix_plan, scaled, dense_cols, prime);
    target_block.swap(scaled);
}



void cpu_elim_z_to_y(
    const vector<my_int>& z,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& y
) {
    const size_t kernel_dim = instance.elimination_kernel_weights.size();
    y.assign(checked_mul(d2_host.numCols, dense_cols, "CPU eliminated z-to-y output"), 0);
    for (size_t row = 0; row < d2_host.numCols; ++row) {
        const CyclicColumnDescriptor& y_desc = d2_host.columnDescriptors[row];
        uint32_t start = z_layout.offsets[y_desc.subset_rank];
        uint32_t end = z_layout.offsets[y_desc.subset_rank + 1];
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = 0;
            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t kernel_idx = z_layout.descriptors[idx].a;
                uint32_t coeff = instance.elimination_kernel_basis_a0_by_kernel[
                    static_cast<size_t>(y_desc.a) * kernel_dim + kernel_idx
                ];
                if (coeff == 0) {
                    continue;
                }
                uint32_t value = z[static_cast<size_t>(idx) * dense_cols + dense_col];
                sum += static_cast<uint64_t>(coeff) * value;
            }
            y[row * dense_cols + dense_col] = static_cast<my_int>(cpu_mod_u64(sum, prime));
        }
    }
}

void cpu_elim_subtract_Rt(
    const vector<my_int>& t,
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicInstance& instance,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& y
) {
    if (y.size() != d2_host.numCols * dense_cols || t.size() != dm_host.numRows * dense_cols) {
        throw std::runtime_error("CPU eliminated subtract_Rt dimensions do not match");
    }
    for (size_t row = 0; row < d2_host.numCols; ++row) {
        const CyclicColumnDescriptor& y_desc = d2_host.columnDescriptors[row];
        uint32_t start = dm_host.rowOffsets[y_desc.subset_rank];
        uint32_t end = dm_host.rowOffsets[y_desc.subset_rank + 1];
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = y[row * dense_cols + dense_col];
            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t beta = dm_host.rowDescriptors[idx].beta;
                uint32_t coeff = instance.elimination_right_inverse_a0_by_a1[
                    static_cast<size_t>(y_desc.a) * instance.n + beta
                ];
                if (coeff == 0) {
                    continue;
                }
                uint32_t value = t[static_cast<size_t>(idx) * dense_cols + dense_col];
                uint32_t term = cpu_mul_mod(coeff, value, prime);
                if (term != 0) {
                    sum += prime - term;
                }
            }
            y[row * dense_cols + dense_col] = static_cast<my_int>(cpu_mod_u64(sum, prime));
        }
    }
}

void cpu_eliminated_forward(
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    const vector<my_int>& source_precond,
    size_t x_cols,
    size_t z_cols,
    const vector<my_int>& input,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& output
) {
    const size_t total_cols = x_cols + z_cols;
    vector<my_int> scaled;
    cpu_scale_rows_dense(input, source_precond, total_cols, dense_cols, prime, scaled);
    vector<my_int> scaled_x = slice_row_block(scaled, 0, x_cols, dense_cols);
    vector<my_int> scaled_z = slice_row_block(scaled, x_cols, z_cols, dense_cols);
    vector<my_int> t;
    vector<my_int> y;
    cpu_phi_forward_scaled_columns(dm_host, scaled_x, dense_cols, prime, t);
    cpu_elim_z_to_y(scaled_z, d2_host, z_layout, instance, dense_cols, prime, y);
    cpu_elim_subtract_Rt(t, dm_host, d2_host, instance, dense_cols, prime, y);
    cpu_phi_forward_scaled_columns(d2_host, y, dense_cols, prime, output);
}

void cpu_elim_RT_h(
    const vector<my_int>& h,
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicInstance& instance,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& t_adj
) {
    t_adj.assign(checked_mul(dm_host.numRows, dense_cols, "CPU eliminated R^t h output"), 0);
    for (size_t row = 0; row < dm_host.numRows; ++row) {
        const CyclicRowDescriptor& t_desc = dm_host.rowDescriptors[row];
        uint32_t start = d2_host.columnOffsets[t_desc.subset_rank];
        uint32_t end = d2_host.columnOffsets[t_desc.subset_rank + 1];
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = 0;
            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t a = d2_host.columnDescriptors[idx].a;
                uint32_t coeff = instance.elimination_right_inverse_a0_by_a1[
                    static_cast<size_t>(a) * instance.n + t_desc.beta
                ];
                if (coeff == 0) {
                    continue;
                }
                uint32_t value = h[static_cast<size_t>(idx) * dense_cols + dense_col];
                sum += static_cast<uint64_t>(coeff) * value;
            }
            t_adj[row * dense_cols + dense_col] = static_cast<my_int>(cpu_mod_u64(sum, prime));
        }
    }
}

void cpu_elim_BT_h(
    const vector<my_int>& h,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& z_adj
) {
    const size_t kernel_dim = instance.elimination_kernel_weights.size();
    z_adj.assign(checked_mul(z_layout.descriptors.size(), dense_cols, "CPU eliminated B^t h output"), 0);
    for (size_t row = 0; row < z_layout.descriptors.size(); ++row) {
        const CyclicColumnDescriptor& z_desc = z_layout.descriptors[row];
        uint32_t start = d2_host.columnOffsets[z_desc.subset_rank];
        uint32_t end = d2_host.columnOffsets[z_desc.subset_rank + 1];
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint64_t sum = 0;
            for (uint32_t idx = start; idx < end; ++idx) {
                uint32_t a = d2_host.columnDescriptors[idx].a;
                uint32_t coeff = instance.elimination_kernel_basis_a0_by_kernel[
                    static_cast<size_t>(a) * kernel_dim + z_desc.a
                ];
                if (coeff == 0) {
                    continue;
                }
                uint32_t value = h[static_cast<size_t>(idx) * dense_cols + dense_col];
                sum += static_cast<uint64_t>(coeff) * value;
            }
            z_adj[row * dense_cols + dense_col] = static_cast<my_int>(cpu_mod_u64(sum, prime));
        }
    }
}

void cpu_elim_assemble_adjoint(
    const vector<my_int>& x_adj,
    const vector<my_int>& z_adj,
    const vector<my_int>& source_precond,
    size_t x_cols,
    size_t z_cols,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& output
) {
    const size_t total_cols = x_cols + z_cols;
    output.assign(checked_mul(total_cols, dense_cols, "CPU eliminated adjoint output"), 0);
    for (size_t row = 0; row < total_cols; ++row) {
        for (size_t dense_col = 0; dense_col < dense_cols; ++dense_col) {
            uint32_t value;
            if (row < x_cols) {
                value = x_adj[row * dense_cols + dense_col];
                if (value != 0) {
                    value = prime - value;
                }
            } else {
                size_t z_row = row - x_cols;
                value = z_adj[z_row * dense_cols + dense_col];
            }
            output[row * dense_cols + dense_col] = static_cast<my_int>(
                cpu_mul_mod(value, source_precond[row], prime)
            );
        }
    }
}

void cpu_eliminated_backward(
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    const vector<my_int>& source_precond,
    size_t x_cols,
    size_t z_cols,
    const vector<my_int>& input,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& output
) {
    vector<my_int> h;
    vector<my_int> t_adj;
    vector<my_int> x_adj;
    vector<my_int> z_adj;
    cpu_phi_transpose(d2_host, input, dense_cols, prime, h);
    cpu_elim_RT_h(h, dm_host, d2_host, instance, dense_cols, prime, t_adj);
    cpu_phi_transpose(dm_host, t_adj, dense_cols, prime, x_adj);
    cpu_elim_BT_h(h, d2_host, z_layout, instance, dense_cols, prime, z_adj);
    cpu_elim_assemble_adjoint(x_adj, z_adj, source_precond, x_cols, z_cols, dense_cols, prime, output);
}

void cpu_apply_eliminated_square(
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    const vector<my_int>& source_precond,
    const vector<my_int>& row_precond,
    const RowMixPlan& rowmix_plan,
    size_t x_cols,
    size_t z_cols,
    const vector<my_int>& input,
    size_t dense_cols,
    uint32_t prime,
    vector<my_int>& output
) {
    vector<my_int> target;
    cpu_eliminated_forward(
        dm_host,
        d2_host,
        z_layout,
        instance,
        source_precond,
        x_cols,
        z_cols,
        input,
        dense_cols,
        prime,
        target
    );
    if (rowmix_plan.enabled()) {
        cpu_apply_target_preconditioner(rowmix_plan, row_precond, d2_host.numRows, dense_cols, prime, target);
    }
    cpu_eliminated_backward(
        dm_host,
        d2_host,
        z_layout,
        instance,
        source_precond,
        x_cols,
        z_cols,
        target,
        dense_cols,
        prime,
        output
    );
}



void validate_eliminated_cpu_cuda(
    const CudaCyclicPhiOperator<my_int>& cuDm,
    const CudaCyclicPhiOperator<my_int>& cuD2,
    const CyclicPhiHostData<my_int>& dm_host,
    const CyclicPhiHostData<my_int>& d2_host,
    const CyclicTensorLayout& z_layout,
    const CyclicInstance& instance,
    const vector<my_int>& source_precond,
    const vector<my_int>& row_precond,
    const RowMixPlan& rowmix_plan,
    const DeviceRowMixPlan& device_rowmix,
    const uint32_t* dZOffsets,
    const CyclicColumnDescriptor* dZDescriptors,
    const my_int* dRightInverse,
    const my_int* dKernelBasis,
    const my_int* dSourcePrecond,
    const my_int* dTargetRowPrecond,
    size_t x_cols,
    size_t z_cols,
    const ValidationOptions& validation,
    uint32_t prime,
    ModulusParams mod_params
) {
    if (!validation.enabled) {
        return;
    }
    size_t dense_cols = std::max<size_t>(1, std::min<size_t>(validation.columns, 16));
    size_t total_cols = x_cols + z_cols;
    std::cout << "Running eliminated CPU/CUDA validation with " << dense_cols
              << " deterministic RHS columns..." << std::endl;
    validate_rowmix_transpose_identity(rowmix_plan, dense_cols, prime, "eliminated target");

    auto apply_forward_gpu = [&](CudaDenseMatrix<my_int>& input,
                                 CudaDenseMatrix<my_int>& output,
                                 CudaDenseMatrix<my_int>& cuScaled,
                                 CudaDenseMatrix<my_int>& cuT,
                                 CudaDenseMatrix<my_int>& cuY) {
        cyclic_launch_scale_rows(input, dSourcePrecond, cuScaled, mod_params);
        CudaDenseMatrix<my_int> scaled_x = cyclic_dense_view(cuScaled.d_data, x_cols, dense_cols);
        CudaDenseMatrix<my_int> scaled_z = cyclic_dense_view(cuScaled.d_data + x_cols * dense_cols, z_cols, dense_cols);
        cuDm.forward_scaled_columns(scaled_x, cuT, mod_params);
        cyclic_launch_elim_z_to_y(
            scaled_z,
            cuY,
            instance.elimination_kernel_weights.size(),
            cuD2.d_columnDescriptors,
            dZOffsets,
            dZDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_subtract_Rt(
            cuT,
            cuY,
            instance.n,
            cuD2.d_columnDescriptors,
            cuDm.d_rowOffsets,
            cuDm.d_rowDescriptors,
            dRightInverse,
            mod_params
        );
        cuD2.forward_scaled_columns(cuY, output, mod_params);
    };

    auto apply_backward_gpu = [&](CudaDenseMatrix<my_int>& input,
                                  CudaDenseMatrix<my_int>& output,
                                  CudaDenseMatrix<my_int>& cuH,
                                  CudaDenseMatrix<my_int>& cuTAdj,
                                  CudaDenseMatrix<my_int>& cuXAdj,
                                  CudaDenseMatrix<my_int>& cuZAdj) {
        cuD2.transpose(input, cuH, mod_params);
        cyclic_launch_elim_RT_h(
            cuH,
            cuTAdj,
            instance.n,
            cuDm.d_rowDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dRightInverse,
            mod_params
        );
        cuDm.transpose(cuTAdj, cuXAdj, mod_params);
        cyclic_launch_elim_BT_h(
            cuH,
            cuZAdj,
            instance.elimination_kernel_weights.size(),
            dZDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_assemble_adjoint(cuXAdj, cuZAdj, dSourcePrecond, output, mod_params);
    };

    vector<uint64_t> salts = {0x404ULL, 0x505ULL, 0x606ULL};
    for (size_t idx = 0; idx < salts.size(); ++idx) {
        vector<my_int> input = deterministic_dense_block(total_cols, dense_cols, prime, salts[idx]);
        vector<my_int> cpu_output;
        cpu_apply_eliminated_square(
            dm_host,
            d2_host,
            z_layout,
            instance,
            source_precond,
            row_precond,
            rowmix_plan,
            x_cols,
            z_cols,
            input,
            dense_cols,
            prime,
            cpu_output
        );

        CudaDenseMatrix<my_int> cuInput = CudaDenseMatrix<my_int>::from_host(input, total_cols, dense_cols);
        CudaDenseMatrix<my_int> cuOutput = CudaDenseMatrix<my_int>::allocate_uninitialized(total_cols, dense_cols);
        CudaDenseMatrix<my_int> cuScaled = CudaDenseMatrix<my_int>::allocate_uninitialized(total_cols, dense_cols);
        CudaDenseMatrix<my_int> cuTarget = CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numRows, dense_cols);
        CudaDenseMatrix<my_int> cuT = CudaDenseMatrix<my_int>::allocate_uninitialized(dm_host.numRows, dense_cols);
        CudaDenseMatrix<my_int> cuY = CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numCols, dense_cols);
        CudaDenseMatrix<my_int> cuH = CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numCols, dense_cols);
        CudaDenseMatrix<my_int> cuTAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(dm_host.numRows, dense_cols);
        CudaDenseMatrix<my_int> cuXAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(x_cols, dense_cols);
        CudaDenseMatrix<my_int> cuZAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(z_cols, dense_cols);

        apply_forward_gpu(cuInput, cuTarget, cuScaled, cuT, cuY);
        if (rowmix_plan.enabled()) {
            if (dTargetRowPrecond == nullptr) {
                throw std::runtime_error("eliminated rowmix validation missing row preconditioner on device");
            }
            CudaDenseMatrix<my_int> cuMixed =
                CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numRows, dense_cols);
            apply_rowmix_forward_device(cuTarget, device_rowmix, mod_params);
            cyclic_launch_scale_rows(cuTarget, dTargetRowPrecond, cuMixed, mod_params);
            apply_rowmix_transpose_device(cuMixed, device_rowmix, mod_params);
            apply_backward_gpu(cuMixed, cuOutput, cuH, cuTAdj, cuXAdj, cuZAdj);
            cuMixed.release();
        } else {
            apply_backward_gpu(cuTarget, cuOutput, cuH, cuTAdj, cuXAdj, cuZAdj);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        vector<my_int> gpu_output = copy_cuda_dense_to_host(cuOutput);
        compare_dense_exact(
            cpu_output,
            gpu_output,
            total_cols,
            dense_cols,
            "eliminated square block " + std::to_string(idx)
        );

        cuInput.release();
        cuOutput.release();
        cuScaled.release();
        cuTarget.release();
        cuT.release();
        cuY.release();
        cuH.release();
        cuTAdj.release();
        cuXAdj.release();
        cuZAdj.release();
    }
}



void run_eliminated_sector(
    const CyclicInstance& instance,
    const CyclicCommonIncidence& common_dm,
    const CyclicCommonIncidence& common_d2,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    const CharacterMap& kernel_map,
    size_t sector,
    const string& output_dir,
    size_t num_v,
    size_t max_nlen_override,
    size_t save_after,
    bool no_save,
    uint64_t random_seed,
    RowMixOptions rowmix_options,
    ValidationOptions validation_options,
    size_t gram_chunk_size,
    ModulusParams mod_params
) {
    if (!instance.has_elimination) {
        throw std::runtime_error("instance does not contain third_section_elimination data");
    }

    auto setup_start = std::chrono::high_resolution_clock::now();
    size_t y_sector = sub_mod(sector, instance.elimination_w_weight, instance.r);
    CyclicSectorLayout dm_layout = build_sector_layout_generic(
        instance, common_dm, a0_map, a1_map, sector, "eliminated D_m"
    );
    CyclicSectorLayout d2_layout = build_sector_layout_generic(
        instance, common_d2, a0_map, a1_map, y_sector, "eliminated D_(m-1)"
    );
    CyclicTensorLayout z_layout = build_tensor_layout(
        common_dm.row_subset_weights, kernel_map, instance.r, y_sector, "eliminated z"
    );

    size_t x_cols = dm_layout.column_descriptors.size();
    size_t t_rows = dm_layout.row_descriptors.size();
    size_t y_rows = d2_layout.column_descriptors.size();
    size_t z_cols = z_layout.descriptors.size();
    size_t target_rows = d2_layout.row_descriptors.size();
    size_t total_cols = x_cols + z_cols;
    validate_eliminated_sector_against_fixture(instance, sector, total_cols, target_rows);

    auto setup_stop = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_stop - setup_start).count();

    std::cout << "\n=== Eliminated Sector " << sector << " ===" << std::endl;
    std::cout << "Reduced dimensions: " << target_rows
              << " x " << total_cols
              << " (x=" << x_cols
              << ", z=" << z_cols
              << ", y=" << y_rows
              << ", t=" << t_rows
              << "), layout_ms=" << setup_ms
              << ", y_sector=" << y_sector
              << std::endl;
    std::cout << "Implicit D nnz counts: D_m=" << dm_layout.nnz
              << ", D_(m-1)=" << d2_layout.nnz << std::endl;

    my_int storage_prime = static_cast<my_int>(instance.modulus);
    std::mt19937_64 gen(seeded_sector_value(random_seed, sector));
    vector<my_int> row_precond = generate_random_vector(target_rows, storage_prime, gen, true);
    vector<my_int> col_precond = generate_random_vector(total_cols, storage_prime, gen, true);
    vector<my_int> initial_B = generate_random_vector(total_cols * num_v, storage_prime, gen);
    vector<my_int> current_B = initial_B;
    RowMixPlan rowmix_plan = generate_rowmix_plan(
        target_rows,
        "eliminated",
        sector,
        instance.modulus,
        rowmix_options
    );
    string wdm_filename = sector_filename(output_dir, "eliminated", sector, num_v, rowmix_options);
    ensure_output_absent(wdm_filename, no_save);

    std::cout << "Row mixing: rounds=" << rowmix_plan.rounds
              << ", seed=" << rowmix_plan.seed
              << ", realized_seed=" << rowmix_plan.realized_seed
              << ", plan_hash=" << hex_u64(rowmix_plan_hash(rowmix_plan))
              << std::endl;
    if (!no_save) {
        write_rowmix_sidecar(
            wdm_filename,
            "eliminated",
            sector,
            target_rows,
            total_cols,
            num_v,
            instance.modulus,
            random_seed,
            rowmix_plan,
            row_precond,
            col_precond,
            initial_B
        );
    }

    vector<my_int> dm_row_ones = ones_vector(t_rows);
    vector<my_int> dm_col_ones = ones_vector(x_cols);
    vector<my_int> d2_col_ones = ones_vector(y_rows);
    vector<my_int> d2_row_precond_for_operator =
        rowmix_plan.enabled() ? ones_vector(target_rows) : row_precond;

    std::cout << "Packaging eliminated D-operators for GPU..." << std::endl;
    CyclicPhiHostData<my_int> dm_host = make_operator_host_data(
        instance,
        common_dm,
        std::move(dm_layout),
        a0_map,
        a1_map,
        std::move(dm_row_ones),
        std::move(dm_col_ones)
    );
    CyclicPhiHostData<my_int> d2_host = make_operator_host_data(
        instance,
        common_d2,
        std::move(d2_layout),
        a0_map,
        a1_map,
        std::move(d2_row_precond_for_operator),
        std::move(d2_col_ones)
    );

    std::cout << "Loading eliminated data onto GPU: D_m="
              << std::fixed << std::setprecision(2)
              << to_mib(cyclic_phi_host_data_memory_size(dm_host))
              << " MiB, D_(m-1)="
              << to_mib(cyclic_phi_host_data_memory_size(d2_host))
              << " MiB, z layout="
              << to_mib(z_layout.offsets.size() * sizeof(uint32_t)
                    + z_layout.descriptors.size() * sizeof(CyclicColumnDescriptor))
              << " MiB" << std::endl;
    CudaCyclicPhiOperator<my_int> cuDm = CudaCyclicPhiOperator<my_int>::from_host(dm_host);
    CudaCyclicPhiOperator<my_int> cuD2 = CudaCyclicPhiOperator<my_int>::from_host(d2_host);
    uint32_t* dZOffsets = copy_vector_to_device(z_layout.offsets, "eliminated z offsets");
    CyclicColumnDescriptor* dZDescriptors =
        copy_vector_to_device(z_layout.descriptors, "eliminated z descriptors");
    my_int* dRightInverse = copy_vector_to_device(
        instance.elimination_right_inverse_a0_by_a1,
        "elimination right inverse"
    );
    my_int* dKernelBasis = copy_vector_to_device(
        instance.elimination_kernel_basis_a0_by_kernel,
        "elimination kernel basis"
    );
    my_int* dSourcePrecond = copy_vector_to_device(col_precond, "eliminated source preconditioner");
    my_int* dTargetRowPrecond = nullptr;
    DeviceRowMixPlan device_rowmix;
    if (rowmix_plan.enabled()) {
        dTargetRowPrecond = copy_vector_to_device(row_precond, "eliminated target row preconditioner");
        device_rowmix = copy_rowmix_plan_to_device(rowmix_plan);
    }

    size_t mem_total = total_cols * num_v * sizeof(my_int);
    size_t mem_target = target_rows * num_v * sizeof(my_int);
    size_t mem_y = y_rows * num_v * sizeof(my_int);
    size_t mem_t = t_rows * num_v * sizeof(my_int);
    std::cout << "Dense block memory: source/D/scaled=" << to_mib(mem_total)
              << " MiB each, target=" << to_mib(mem_target)
              << " MiB, y=" << to_mib(mem_y)
              << " MiB, t=" << to_mib(mem_t) << " MiB" << std::endl;

    CudaDenseMatrix<my_int> cuB = CudaDenseMatrix<my_int>::from_host(current_B, total_cols, num_v);
    CudaDenseMatrix<my_int> cuD = CudaDenseMatrix<my_int>::allocate_uninitialized(total_cols, num_v);
    CudaDenseMatrix<my_int> cuScaled = CudaDenseMatrix<my_int>::allocate_uninitialized(total_cols, num_v);
    CudaDenseMatrix<my_int> cuTarget = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
    CudaDenseMatrix<my_int> cuT = CudaDenseMatrix<my_int>::allocate_uninitialized(t_rows, num_v);
    CudaDenseMatrix<my_int> cuY = CudaDenseMatrix<my_int>::allocate_uninitialized(y_rows, num_v);
    CudaDenseMatrix<my_int> cuH = CudaDenseMatrix<my_int>::allocate_uninitialized(y_rows, num_v);
    CudaDenseMatrix<my_int> cuTAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(t_rows, num_v);
    CudaDenseMatrix<my_int> cuXAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(x_cols, num_v);
    CudaDenseMatrix<my_int> cuZAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(z_cols, num_v);
    CudaDenseMatrix<my_int> cuTargetMixed;
    if (rowmix_plan.enabled()) {
        cuTargetMixed = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
        std::cout << "Mixed target scratch memory: " << to_mib(mem_target)
                  << " MiB; rowmix device plan rounds=" << device_rowmix.rounds.size()
                  << std::endl;
    }

    size_t default_max_nlen = (2 * total_cols) / num_v + 50;
    size_t max_nlen = max_nlen_override == 0 ? default_max_nlen : max_nlen_override;
    size_t sp_size = upper_triangular_size(num_v);
    size_t bigsp_len = max_nlen + 5;
    std::cout << "Sequence target: " << max_nlen
              << " entries; packed sequence GPU memory "
              << to_mib(bigsp_len * sp_size * sizeof(my_int)) << " MiB" << std::endl;
    CudaDenseMatrix<my_int> cuBigSp = CudaDenseMatrix<my_int>::allocate_uninitialized(bigsp_len, sp_size);
    CudaDenseMatrix<accum_int> cuGramScratch = CudaDenseMatrix<accum_int>::allocate_uninitialized(1, num_v * num_v);

    my_int* dBigSp = cuBigSp.d_data;
    size_t seq_position = 0;
    compute_and_push_bigsp2(cuB, cuB, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);

    auto apply_forward = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        cyclic_launch_scale_rows(input, dSourcePrecond, cuScaled, mod_params);
        CudaDenseMatrix<my_int> scaled_x = cyclic_dense_view(cuScaled.d_data, x_cols, num_v);
        CudaDenseMatrix<my_int> scaled_z = cyclic_dense_view(cuScaled.d_data + x_cols * num_v, z_cols, num_v);
        cuDm.forward_scaled_columns(scaled_x, cuT, mod_params);
        cyclic_launch_elim_z_to_y(
            scaled_z,
            cuY,
            instance.elimination_kernel_weights.size(),
            cuD2.d_columnDescriptors,
            dZOffsets,
            dZDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_subtract_Rt(
            cuT,
            cuY,
            instance.n,
            cuD2.d_columnDescriptors,
            cuDm.d_rowOffsets,
            cuDm.d_rowDescriptors,
            dRightInverse,
            mod_params
        );
        cuD2.forward_scaled_columns(cuY, output, mod_params);
    };

    auto apply_backward = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        cuD2.transpose(input, cuH, mod_params);
        cyclic_launch_elim_RT_h(
            cuH,
            cuTAdj,
            instance.n,
            cuDm.d_rowDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dRightInverse,
            mod_params
        );
        cuDm.transpose(cuTAdj, cuXAdj, mod_params);
        cyclic_launch_elim_BT_h(
            cuH,
            cuZAdj,
            instance.elimination_kernel_weights.size(),
            dZDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_assemble_adjoint(cuXAdj, cuZAdj, dSourcePrecond, output, mod_params);
    };

    validate_eliminated_cpu_cuda(
        cuDm,
        cuD2,
        dm_host,
        d2_host,
        z_layout,
        instance,
        col_precond,
        row_precond,
        rowmix_plan,
        device_rowmix,
        dZOffsets,
        dZDescriptors,
        dRightInverse,
        dKernelBasis,
        dSourcePrecond,
        dTargetRowPrecond,
        x_cols,
        z_cols,
        validation_options,
        instance.modulus,
        mod_params
    );

    auto apply_square = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        apply_forward(input, cuTarget);
        if (rowmix_plan.enabled()) {
            apply_rowmix_forward_device(cuTarget, device_rowmix, mod_params);
            cyclic_launch_scale_rows(cuTarget, dTargetRowPrecond, cuTargetMixed, mod_params);
            apply_rowmix_transpose_device(cuTargetMixed, device_rowmix, mod_params);
            apply_backward(cuTargetMixed, output);
        } else {
            apply_backward(cuTarget, output);
        }
    };

    auto computation_start = std::chrono::high_resolution_clock::now();
    long long last_save = 0;
    long long last_report = 0;
    long long report_interval = 1000;
    size_t last_nlen = seq_position;
    size_t rounds = ((max_nlen - seq_position) + 3) / 4;
    std::cout << "Computing " << rounds << " eliminated BCW rounds for sector "
              << sector << "." << std::endl;

    for (size_t round = 0; round < rounds; ++round) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - computation_start).count();
        if (save_after > 0 && !no_save && elapsed - last_save > static_cast<long long>(save_after) * 1000) {
            std::cout << "\nSaving eliminated sector " << sector << " after " << elapsed / 1000 << " s" << std::endl;
            save_all_data(wdm_filename, target_rows, total_cols, num_v, static_cast<my_int>(instance.modulus),
                          row_precond, col_precond, initial_B, cuB.d_data, dBigSp, seq_position);
            last_save = elapsed;
        }
        if (elapsed - last_report > report_interval) {
            report_progress(elapsed, last_report, last_nlen, seq_position, max_nlen, num_v,
                            "eliminated sector " + std::to_string(sector));
        }

        apply_square(cuB, cuD);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
        compute_and_push_bigsp2(cuD, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);

        apply_square(cuD, cuB);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
        compute_and_push_bigsp2(cuB, cuB, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto computation_stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(computation_stop - computation_start).count();
    std::cout << std::endl;
    std::cout << "Eliminated sector " << sector << " BCW sequence runtime: "
              << milliseconds << " ms" << std::endl;
    if (milliseconds > 0) {
        std::cout << "Eliminated sector " << sector << " total throughput: "
                  << seq_position * num_v * 1e3 / milliseconds << "/s." << std::endl;
    }

    if (no_save) {
        std::cout << "Skipping final WDM save for eliminated sector " << sector << " due to --no-save." << std::endl;
    } else {
        save_all_data(wdm_filename, target_rows, total_cols, num_v, static_cast<my_int>(instance.modulus),
                      row_precond, col_precond, initial_B, cuB.d_data, dBigSp, seq_position);
    }

    cuDm.release();
    cuD2.release();
    CHECK_CUDA(cudaFree(dZOffsets));
    CHECK_CUDA(cudaFree(dZDescriptors));
    CHECK_CUDA(cudaFree(dRightInverse));
    CHECK_CUDA(cudaFree(dKernelBasis));
    CHECK_CUDA(cudaFree(dSourcePrecond));
    if (rowmix_plan.enabled()) {
        CHECK_CUDA(cudaFree(dTargetRowPrecond));
        device_rowmix.release();
    }
    cuB.release();
    cuD.release();
    cuScaled.release();
    cuTarget.release();
    if (rowmix_plan.enabled()) {
        cuTargetMixed.release();
    }
    cuT.release();
    cuY.release();
    cuH.release();
    cuTAdj.release();
    cuXAdj.release();
    cuZAdj.release();
    cuBigSp.release();
    cuGramScratch.release();
}

void run_deformation_eliminated_variant(
    const CyclicInstance& instance,
    const CyclicCommonIncidence& common_dm,
    const CyclicCommonIncidence& common_d2,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    const CharacterMap& kernel_map,
    size_t sector,
    const string& output_dir,
    const string& operator_name,
    const string& dense_filename,
    const string& drop_columns_filename,
    size_t num_v,
    size_t max_nlen_override,
    size_t save_after,
    bool no_save,
    uint64_t random_seed,
    RowMixOptions rowmix_options,
    size_t gram_chunk_size,
    ModulusParams mod_params
) {
    if (!instance.has_elimination) {
        throw std::runtime_error("instance does not contain third_section_elimination data");
    }
    if (operator_name != "augmented" && operator_name != "replacement") {
        throw std::runtime_error("deformation operator must be augmented or replacement");
    }

    auto setup_start = std::chrono::high_resolution_clock::now();
    size_t y_sector = sub_mod(sector, instance.elimination_w_weight, instance.r);
    CyclicSectorLayout dm_layout = build_sector_layout_generic(
        instance, common_dm, a0_map, a1_map, sector, "deformation D_m"
    );
    CyclicSectorLayout d2_layout = build_sector_layout_generic(
        instance, common_d2, a0_map, a1_map, y_sector, "deformation D_(m-1)"
    );
    CyclicTensorLayout z_layout = build_tensor_layout(
        common_dm.row_subset_weights, kernel_map, instance.r, y_sector, "deformation z"
    );

    size_t x_cols = dm_layout.column_descriptors.size();
    size_t t_rows = dm_layout.row_descriptors.size();
    size_t y_rows = d2_layout.column_descriptors.size();
    size_t z_cols = z_layout.descriptors.size();
    size_t target_rows = d2_layout.row_descriptors.size();
    size_t base_cols = x_cols + z_cols;
    validate_eliminated_sector_against_fixture(instance, sector, base_cols, target_rows);

    if (dense_filename.empty()) {
        throw std::runtime_error("deformation operators require --dense-file");
    }
    DenseBlockFile dense_block = load_dense_block_file(dense_filename, instance.modulus);
    if (dense_block.rows != target_rows || dense_block.columns != 2) {
        throw std::runtime_error("dense block dimensions do not match the eliminated target sector");
    }

    vector<uint32_t> drop_columns;
    size_t kept_cols = base_cols;
    if (operator_name == "replacement") {
        if (drop_columns_filename.empty()) {
            throw std::runtime_error("replacement operator requires --drop-columns");
        }
        drop_columns = load_drop_columns_file(drop_columns_filename, dense_block.columns, base_cols);
        kept_cols = base_cols - drop_columns.size();
    } else if (!drop_columns_filename.empty()) {
        throw std::runtime_error("--drop-columns is only meaningful for replacement");
    }
    size_t operator_cols = operator_name == "augmented"
        ? base_cols + dense_block.columns
        : kept_cols + dense_block.columns;

    auto setup_stop = std::chrono::high_resolution_clock::now();
    auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_stop - setup_start).count();

    std::cout << "\n=== Deformation " << operator_name << " sector " << sector << " ===" << std::endl;
    std::cout << "Base F0 dimensions: " << target_rows << " x " << base_cols
              << " (x=" << x_cols << ", z=" << z_cols
              << ", y=" << y_rows << ", t=" << t_rows << ")" << std::endl;
    std::cout << "Deformation operator dimensions: " << target_rows
              << " x " << operator_cols
              << ", dense_columns=" << dense_block.columns
              << ", layout_ms=" << setup_ms
              << ", y_sector=" << y_sector
              << std::endl;
    if (!drop_columns.empty()) {
        std::cout << "Dropped base columns:";
        for (uint32_t col : drop_columns) {
            std::cout << " " << col;
        }
        std::cout << std::endl;
    }

    my_int storage_prime = static_cast<my_int>(instance.modulus);
    std::mt19937_64 gen(seeded_sector_value(random_seed, sector));
    vector<my_int> row_precond = generate_random_vector(target_rows, storage_prime, gen, true);
    vector<my_int> col_precond = generate_random_vector(operator_cols, storage_prime, gen, true);
    vector<my_int> initial_B = generate_random_vector(operator_cols * num_v, storage_prime, gen);
    vector<my_int> current_B = initial_B;
    RowMixPlan rowmix_plan = generate_rowmix_plan(
        target_rows,
        operator_name,
        sector,
        instance.modulus,
        rowmix_options
    );
    string wdm_filename = sector_filename(output_dir, operator_name, sector, num_v, rowmix_options);
    ensure_output_absent(wdm_filename, no_save);

    std::cout << "Dense block file: " << dense_filename << std::endl;
    std::cout << "Row mixing: rounds=" << rowmix_plan.rounds
              << ", seed=" << rowmix_plan.seed
              << ", realized_seed=" << rowmix_plan.realized_seed
              << ", plan_hash=" << hex_u64(rowmix_plan_hash(rowmix_plan))
              << std::endl;
    if (!no_save) {
        write_rowmix_sidecar(
            wdm_filename,
            operator_name,
            sector,
            target_rows,
            operator_cols,
            num_v,
            instance.modulus,
            random_seed,
            rowmix_plan,
            row_precond,
            col_precond,
            initial_B
        );
    }

    vector<my_int> base_source_precond;
    if (operator_name == "augmented") {
        base_source_precond.assign(col_precond.begin(), col_precond.begin() + base_cols);
    } else {
        base_source_precond.assign(base_cols, 0);
        size_t source_row = 0;
        size_t drop_idx = 0;
        for (size_t full_col = 0; full_col < base_cols; ++full_col) {
            if (drop_idx < drop_columns.size() && full_col == drop_columns[drop_idx]) {
                ++drop_idx;
                continue;
            }
            base_source_precond[full_col] = col_precond[source_row];
            ++source_row;
        }
        if (source_row != kept_cols) {
            throw std::runtime_error("replacement column complement construction failed");
        }
    }

    vector<my_int> dm_row_ones = ones_vector(t_rows);
    vector<my_int> dm_col_ones = ones_vector(x_cols);
    vector<my_int> d2_col_ones = ones_vector(y_rows);
    vector<my_int> d2_row_precond_for_operator =
        rowmix_plan.enabled() ? ones_vector(target_rows) : row_precond;

    std::cout << "Packaging base eliminated F0 D-operators for GPU..." << std::endl;
    CyclicPhiHostData<my_int> dm_host = make_operator_host_data(
        instance,
        common_dm,
        std::move(dm_layout),
        a0_map,
        a1_map,
        std::move(dm_row_ones),
        std::move(dm_col_ones)
    );
    CyclicPhiHostData<my_int> d2_host = make_operator_host_data(
        instance,
        common_d2,
        std::move(d2_layout),
        a0_map,
        a1_map,
        std::move(d2_row_precond_for_operator),
        std::move(d2_col_ones)
    );

    std::cout << "Loading deformation data onto GPU: D_m="
              << std::fixed << std::setprecision(2)
              << to_mib(cyclic_phi_host_data_memory_size(dm_host))
              << " MiB, D_(m-1)="
              << to_mib(cyclic_phi_host_data_memory_size(d2_host))
              << " MiB, dense block="
              << to_mib(dense_block.values.size() * sizeof(my_int))
              << " MiB" << std::endl;

    CudaCyclicPhiOperator<my_int> cuDm = CudaCyclicPhiOperator<my_int>::from_host(dm_host);
    CudaCyclicPhiOperator<my_int> cuD2 = CudaCyclicPhiOperator<my_int>::from_host(d2_host);
    CudaDenseMatrix<my_int> cuDenseBlock =
        CudaDenseMatrix<my_int>::from_host(dense_block.values, dense_block.rows, dense_block.columns);
    uint32_t* dZOffsets = copy_vector_to_device(z_layout.offsets, "deformation z offsets");
    CyclicColumnDescriptor* dZDescriptors =
        copy_vector_to_device(z_layout.descriptors, "deformation z descriptors");
    my_int* dRightInverse = copy_vector_to_device(
        instance.elimination_right_inverse_a0_by_a1,
        "elimination right inverse"
    );
    my_int* dKernelBasis = copy_vector_to_device(
        instance.elimination_kernel_basis_a0_by_kernel,
        "elimination kernel basis"
    );
    my_int* dOperatorSourcePrecond =
        copy_vector_to_device(col_precond, "deformation source preconditioner");
    my_int* dBaseSourcePrecond =
        copy_vector_to_device(base_source_precond, "base F0 source preconditioner");
    my_int* dTargetRowPrecond = nullptr;
    DeviceRowMixPlan device_rowmix;
    if (rowmix_plan.enabled()) {
        dTargetRowPrecond = copy_vector_to_device(row_precond, "deformation target row preconditioner");
        device_rowmix = copy_rowmix_plan_to_device(rowmix_plan);
    }

    size_t mem_op = operator_cols * num_v * sizeof(my_int);
    size_t mem_base = base_cols * num_v * sizeof(my_int);
    size_t mem_target = target_rows * num_v * sizeof(my_int);
    std::cout << "Dense block memory: source/D=" << to_mib(mem_op)
              << " MiB each, base=" << to_mib(mem_base)
              << " MiB, target=" << to_mib(mem_target)
              << " MiB" << std::endl;

    CudaDenseMatrix<my_int> cuB = CudaDenseMatrix<my_int>::from_host(current_B, operator_cols, num_v);
    CudaDenseMatrix<my_int> cuD = CudaDenseMatrix<my_int>::allocate_uninitialized(operator_cols, num_v);
    CudaDenseMatrix<my_int> cuTarget = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
    CudaDenseMatrix<my_int> cuT = CudaDenseMatrix<my_int>::allocate_uninitialized(t_rows, num_v);
    CudaDenseMatrix<my_int> cuY = CudaDenseMatrix<my_int>::allocate_uninitialized(y_rows, num_v);
    CudaDenseMatrix<my_int> cuH = CudaDenseMatrix<my_int>::allocate_uninitialized(y_rows, num_v);
    CudaDenseMatrix<my_int> cuTAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(t_rows, num_v);
    CudaDenseMatrix<my_int> cuXAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(x_cols, num_v);
    CudaDenseMatrix<my_int> cuZAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(z_cols, num_v);
    CudaDenseMatrix<my_int> cuBaseAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(base_cols, num_v);
    CudaDenseMatrix<my_int> cuScaledOp;
    CudaDenseMatrix<my_int> cuFullScaled;
    CudaDenseMatrix<my_int> cuExtraScaled;
    if (operator_name == "augmented") {
        cuScaledOp = CudaDenseMatrix<my_int>::allocate_uninitialized(operator_cols, num_v);
    } else {
        cuFullScaled = CudaDenseMatrix<my_int>::allocate_uninitialized(base_cols, num_v);
        cuExtraScaled = CudaDenseMatrix<my_int>::allocate_uninitialized(dense_block.columns, num_v);
    }
    CudaDenseMatrix<my_int> cuTargetMixed;
    if (rowmix_plan.enabled()) {
        cuTargetMixed = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
    }
    CudaDenseMatrix<accum_int> cuDenseTransposeScratch =
        CudaDenseMatrix<accum_int>::allocate_uninitialized(dense_block.columns, num_v);

    auto apply_forward_scaled_base = [&](CudaDenseMatrix<my_int>& scaled_base,
                                         CudaDenseMatrix<my_int>& output) {
        CudaDenseMatrix<my_int> scaled_x = cyclic_dense_view(scaled_base.d_data, x_cols, num_v);
        CudaDenseMatrix<my_int> scaled_z =
            cyclic_dense_view(scaled_base.d_data + x_cols * num_v, z_cols, num_v);
        cuDm.forward_scaled_columns(scaled_x, cuT, mod_params);
        cyclic_launch_elim_z_to_y(
            scaled_z,
            cuY,
            instance.elimination_kernel_weights.size(),
            cuD2.d_columnDescriptors,
            dZOffsets,
            dZDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_subtract_Rt(
            cuT,
            cuY,
            instance.n,
            cuD2.d_columnDescriptors,
            cuDm.d_rowOffsets,
            cuDm.d_rowDescriptors,
            dRightInverse,
            mod_params
        );
        cuD2.forward_scaled_columns(cuY, output, mod_params);
    };

    auto apply_backward_base = [&](CudaDenseMatrix<my_int>& input,
                                   CudaDenseMatrix<my_int>& output_base) {
        cuD2.transpose(input, cuH, mod_params);
        cyclic_launch_elim_RT_h(
            cuH,
            cuTAdj,
            instance.n,
            cuDm.d_rowDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dRightInverse,
            mod_params
        );
        cuDm.transpose(cuTAdj, cuXAdj, mod_params);
        cyclic_launch_elim_BT_h(
            cuH,
            cuZAdj,
            instance.elimination_kernel_weights.size(),
            dZDescriptors,
            cuD2.d_columnOffsets,
            cuD2.d_columnDescriptors,
            dKernelBasis,
            mod_params
        );
        cyclic_launch_elim_assemble_adjoint(cuXAdj, cuZAdj, dBaseSourcePrecond, output_base, mod_params);
    };

    auto apply_forward = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        if (operator_name == "augmented") {
            cyclic_launch_scale_rows(input, dOperatorSourcePrecond, cuScaledOp, mod_params);
            CudaDenseMatrix<my_int> scaled_base = cyclic_dense_view(cuScaledOp.d_data, base_cols, num_v);
            CudaDenseMatrix<my_int> scaled_extra =
                cyclic_dense_view(cuScaledOp.d_data + base_cols * num_v, dense_block.columns, num_v);
            apply_forward_scaled_base(scaled_base, output);
            cyclic_launch_dense_columns_axpy<my_int, true>(cuDenseBlock, scaled_extra, output, mod_params);
        } else {
            CudaDenseMatrix<my_int> input_kept = cyclic_dense_view(input.d_data, kept_cols, num_v);
            CudaDenseMatrix<my_int> input_extra =
                cyclic_dense_view(input.d_data + kept_cols * num_v, dense_block.columns, num_v);
            cyclic_launch_insert_complement_scaled(
                input_kept,
                drop_columns[0],
                drop_columns[1],
                dOperatorSourcePrecond,
                cuFullScaled,
                mod_params
            );
            cyclic_launch_scale_rows(
                input_extra,
                dOperatorSourcePrecond + kept_cols,
                cuExtraScaled,
                mod_params
            );
            apply_forward_scaled_base(cuFullScaled, output);
            cyclic_launch_dense_columns_axpy<my_int, false>(cuDenseBlock, cuExtraScaled, output, mod_params);
        }
    };

    auto apply_backward = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        if (operator_name == "augmented") {
            CudaDenseMatrix<my_int> output_base = cyclic_dense_view(output.d_data, base_cols, num_v);
            apply_backward_base(input, output_base);
            cyclic_launch_dense_columns_transpose<my_int, true>(
                cuDenseBlock,
                input,
                dOperatorSourcePrecond,
                base_cols,
                output,
                cuDenseTransposeScratch.d_data,
                gram_chunk_size,
                mod_params
            );
        } else {
            apply_backward_base(input, cuBaseAdj);
            CudaDenseMatrix<my_int> output_kept = cyclic_dense_view(output.d_data, kept_cols, num_v);
            cyclic_launch_gather_complement(
                cuBaseAdj,
                drop_columns[0],
                drop_columns[1],
                output_kept
            );
            cyclic_launch_dense_columns_transpose<my_int, false>(
                cuDenseBlock,
                input,
                dOperatorSourcePrecond,
                kept_cols,
                output,
                cuDenseTransposeScratch.d_data,
                gram_chunk_size,
                mod_params
            );
        }
    };

    size_t default_max_nlen = (2 * operator_cols) / num_v + 50;
    size_t max_nlen = max_nlen_override == 0 ? default_max_nlen : max_nlen_override;
    size_t sp_size = upper_triangular_size(num_v);
    size_t bigsp_len = max_nlen + 5;
    std::cout << "Sequence target: " << max_nlen
              << " entries; packed sequence GPU memory "
              << to_mib(bigsp_len * sp_size * sizeof(my_int)) << " MiB" << std::endl;
    CudaDenseMatrix<my_int> cuBigSp = CudaDenseMatrix<my_int>::allocate_uninitialized(bigsp_len, sp_size);
    CudaDenseMatrix<accum_int> cuGramScratch =
        CudaDenseMatrix<accum_int>::allocate_uninitialized(1, num_v * num_v);

    my_int* dBigSp = cuBigSp.d_data;
    size_t seq_position = 0;
    compute_and_push_bigsp2(cuB, cuB, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);

    auto apply_square = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        apply_forward(input, cuTarget);
        if (rowmix_plan.enabled()) {
            apply_rowmix_forward_device(cuTarget, device_rowmix, mod_params);
            cyclic_launch_scale_rows(cuTarget, dTargetRowPrecond, cuTargetMixed, mod_params);
            apply_rowmix_transpose_device(cuTargetMixed, device_rowmix, mod_params);
            apply_backward(cuTargetMixed, output);
        } else {
            apply_backward(cuTarget, output);
        }
    };

    auto computation_start = std::chrono::high_resolution_clock::now();
    long long last_save = 0;
    long long last_report = 0;
    long long report_interval = 1000;
    size_t last_nlen = seq_position;
    size_t rounds = ((max_nlen - seq_position) + 3) / 4;
    std::cout << "Computing " << rounds << " deformation " << operator_name
              << " BCW rounds for sector " << sector << "." << std::endl;

    for (size_t round = 0; round < rounds; ++round) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - computation_start).count();
        if (save_after > 0 && !no_save && elapsed - last_save > static_cast<long long>(save_after) * 1000) {
            std::cout << "\nSaving deformation " << operator_name
                      << " sector " << sector << " after " << elapsed / 1000 << " s" << std::endl;
            save_all_data(
                wdm_filename,
                target_rows,
                operator_cols,
                num_v,
                static_cast<my_int>(instance.modulus),
                row_precond,
                col_precond,
                initial_B,
                cuB.d_data,
                dBigSp,
                seq_position
            );
            last_save = elapsed;
        }
        if (elapsed - last_report > report_interval) {
            report_progress(
                elapsed,
                last_report,
                last_nlen,
                seq_position,
                max_nlen,
                num_v,
                operator_name + " sector " + std::to_string(sector)
            );
        }

        apply_square(cuB, cuD);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
        compute_and_push_bigsp2(cuD, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);

        apply_square(cuD, cuB);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
        compute_and_push_bigsp2(cuB, cuB, dBigSp, cuGramScratch.d_data, seq_position, mod_params, gram_chunk_size);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto computation_stop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(computation_stop - computation_start).count();
    std::cout << std::endl;
    std::cout << "Deformation " << operator_name << " sector " << sector
              << " BCW sequence runtime: " << milliseconds << " ms" << std::endl;
    if (milliseconds > 0) {
        std::cout << "Deformation " << operator_name << " sector " << sector
                  << " total throughput: "
                  << seq_position * num_v * 1e3 / milliseconds << "/s." << std::endl;
    }

    if (no_save) {
        std::cout << "Skipping final WDM save for deformation "
                  << operator_name << " sector " << sector << " due to --no-save." << std::endl;
    } else {
        save_all_data(
            wdm_filename,
            target_rows,
            operator_cols,
            num_v,
            static_cast<my_int>(instance.modulus),
            row_precond,
            col_precond,
            initial_B,
            cuB.d_data,
            dBigSp,
            seq_position
        );
    }

    cuDm.release();
    cuD2.release();
    cuDenseBlock.release();
    CHECK_CUDA(cudaFree(dZOffsets));
    CHECK_CUDA(cudaFree(dZDescriptors));
    CHECK_CUDA(cudaFree(dRightInverse));
    CHECK_CUDA(cudaFree(dKernelBasis));
    CHECK_CUDA(cudaFree(dOperatorSourcePrecond));
    CHECK_CUDA(cudaFree(dBaseSourcePrecond));
    if (rowmix_plan.enabled()) {
        CHECK_CUDA(cudaFree(dTargetRowPrecond));
        device_rowmix.release();
        cuTargetMixed.release();
    }
    cuB.release();
    cuD.release();
    cuTarget.release();
    cuT.release();
    cuY.release();
    cuH.release();
    cuTAdj.release();
    cuXAdj.release();
    cuZAdj.release();
    cuBaseAdj.release();
    if (operator_name == "augmented") {
        cuScaledOp.release();
    } else {
        cuFullScaled.release();
        cuExtraScaled.release();
    }
    cuDenseTransposeScratch.release();
    cuBigSp.release();
    cuGramScratch.release();
}

int main(int argc, char* argv[]) {
    try {
        CLI::App app{"CUDA deformation Wiedemann sequence computation for cyclic Prym-Green operators"};

        string instance_filename;
        string output_dir = ".";
        string operator_name = "eliminated";
        string dense_filename;
        string drop_columns_filename;
        bool no_save = false;
        int sector_arg = -1;
        int cuda_device_id = 0;
        size_t num_v = 4;
        size_t max_nlen_override = 0;
        size_t save_after = 0;
        size_t gram_chunk_override = 0;
        uint64_t random_seed = 1;
        size_t rowmix_rounds = 2;
        uint64_t rowmix_seed = 10001;
        bool validate_rowmix = false;
        size_t validate_columns = 4;

        app.add_option("instance", instance_filename, "Cyclic Prym fixture/instance JSON")->required();
        app.add_option(
            "--operator",
            operator_name,
            "Production operator: eliminated, augmented, or replacement."
        )->default_val("eliminated");
        app.add_option("--dense-file", dense_filename, "Dense deformation block JSON for augmented/replacement operators.");
        app.add_option("--drop-columns", drop_columns_filename, "JSON file with two base-sector columns to delete for replacement.");
        app.add_option("--sector", sector_arg, "Run one sector/character.")->required();
        app.add_option("--out", output_dir, "Output directory for per-sector WDM files.")->default_val(".");
        app.add_option("-d,--device", cuda_device_id, "Selects a CUDA device to use.")->default_val(0);
        app.add_option("-v", num_v, "Paper block size: 4.")->default_val(4);
        app.add_option("-s,--saveafter", save_after, "Trigger automatic saves every s seconds; 0 disables intermediate saves.")->default_val(0);
        app.add_option("--seed", random_seed, "Base deterministic RNG seed for preconditioners and starting vectors.")->default_val(1);
        app.add_option("--rowmix-rounds", rowmix_rounds, "Paper target row-mixing rounds: 2.")->default_val(2);
        app.add_option("--rowmix-seed", rowmix_seed, "Independent deterministic RNG seed for row mixing.")->default_val(10001);
        app.add_flag("--validate-rowmix", validate_rowmix, "Run focused CPU/CUDA square-operator validation before sequence generation.");

        CLI11_PARSE(app, argc, argv);

        if (rowmix_rounds != 2 || rowmix_seed != 10001) {
            throw std::runtime_error("The paper pipeline requires --rowmix-rounds 2 --rowmix-seed 10001.");
        }
        if (random_seed != 1 || save_after != 0) {
            throw std::runtime_error("The paper pipeline requires --seed 1 --saveafter 0.");
        }


        if (operator_name != "eliminated"
                && operator_name != "augmented"
                && operator_name != "replacement") {
            std::cerr << "--operator must be one of: eliminated, augmented, replacement." << std::endl;
            return 1;
        }
        if (num_v != 4) {
            std::cerr << "The paper pipeline requires exactly 4 block vectors." << std::endl;
            return 1;
        }


        if (sector_arg < 0) {
            std::cerr << "--sector must be nonnegative." << std::endl;
            return 1;
        }

        std::cout << "Loading cyclic Prym instance JSON: " << instance_filename << std::endl;
        CyclicInstance instance = load_cyclic_instance(instance_filename);
        if (sector_arg >= 0 && static_cast<size_t>(sector_arg) >= instance.r) {
            std::cerr << "--sector must be in 0.." << instance.r - 1 << std::endl;
            return 1;
        }
        if (!is_prime(instance.modulus)) {
            std::cerr << "Modulus " << instance.modulus << " is not prime." << std::endl;
            return 1;
        }

        size_t max_safe_gram_chunk = max_safe_gram_chunk_size(instance.modulus);
        size_t gram_chunk_size = gram_chunk_override == 0
            ? std::min<size_t>(AUTO_GRAM_CHUNK_CAP, max_safe_gram_chunk)
            : gram_chunk_override;
        uint64_t p64 = instance.modulus;
        uint64_t accum_max = std::numeric_limits<accum_int>::max() - 1ULL;
        if (!(gram_chunk_size > 0
                && p64 <= static_cast<uint64_t>(std::numeric_limits<my_int>::max())
                && static_cast<uint64_t>(gram_chunk_size) * p64 * p64 < accum_max)) {
            std::cerr << "Prime/chunk combination is not valid for uint16_t storage and uint32_t Gram accumulators." << std::endl;
            return 1;
        }

        CHECK_CUDA(cudaSetDevice(cuda_device_id));
        cudaDeviceProp device_prop;
        CHECK_CUDA(cudaGetDeviceProperties(&device_prop, cuda_device_id));
        std::cout << "Using CUDA device " << cuda_device_id << ": " << device_prop.name
                  << ", global memory " << std::fixed << std::setprecision(2)
                  << to_mib(device_prop.totalGlobalMem) << " MiB" << std::endl;

        std::cout << "Cyclic Phi_g data: g=" << instance.genus
                  << ", r=" << instance.r
                  << ", n=" << instance.n
                  << ", m=" << instance.m
                  << ", modulus=" << instance.modulus
                  << ", mu entries=" << instance.mu.size()
                  << std::endl;
        std::cout << "Gram dot chunk size: " << gram_chunk_size
                  << " (max safe for this prime: " << max_safe_gram_chunk << ")"
                  << std::endl;
        std::cout << "Row mixing configuration: rounds=" << rowmix_rounds
                  << ", seed=" << rowmix_seed
                  << ", algorithm=" << ROWMIX_ALGORITHM_VERSION
                  << std::endl;

        RowMixOptions rowmix_options{rowmix_rounds, rowmix_seed};
        ValidationOptions validation_options{validate_rowmix, validate_columns};

        CharacterMap a0_map = build_character_map(instance.weight_a0, instance.r, "A0");
        CharacterMap a1_map = build_character_map(instance.weight_a1, instance.r, "A1");
        std::cout << "A0 character entries: " << a0_map.indices.size()
                  << ", A1 character entries: " << a1_map.indices.size()
                  << std::endl;

        ModulusParams mod_params = make_modulus_params(instance.modulus);

        {
            if (!instance.has_elimination) {
                std::cerr << "Instance does not contain third_section_elimination data." << std::endl;
                return 1;
            }
            vector<size_t> u_indices;
            vector<uint16_t> u_weights;
            u_indices.reserve(instance.n - 1);
            u_weights.reserve(instance.n - 1);
            for (size_t i = 0; i < instance.n; ++i) {
                if (i == instance.elimination_w_index) {
                    continue;
                }
                u_indices.push_back(i);
                u_weights.push_back(instance.weight_v[i]);
            }
            CharacterMap kernel_map = build_character_map(
                instance.elimination_kernel_weights,
                instance.r,
                "ker(mu_w)"
            );
            std::cout << "Eliminated operator: w_V_index=" << instance.elimination_w_index
                      << ", w_weight=" << instance.elimination_w_weight
                      << ", U dimension=" << u_indices.size()
                      << ", kernel dimension=" << instance.elimination_kernel_weights.size()
                      << std::endl;

            auto dm_start = std::chrono::high_resolution_clock::now();
            std::cout << "Building eliminated D_m incidences on U..." << std::endl;
            CyclicCommonIncidence common_dm = build_common_incidences_generic(
                u_indices.size(),
                instance.m,
                instance.r,
                u_indices,
                u_weights,
                "eliminated D_m"
            );
            auto d2_start = std::chrono::high_resolution_clock::now();
            std::cout << "Building eliminated D_(m-1) incidences on U..." << std::endl;
            CyclicCommonIncidence common_d2 = build_common_incidences_generic(
                u_indices.size(),
                instance.m - 1,
                instance.r,
                u_indices,
                u_weights,
                "eliminated D_(m-1)"
            );
            auto d2_stop = std::chrono::high_resolution_clock::now();
            auto dm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d2_start - dm_start).count();
            auto d2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(d2_stop - d2_start).count();
            std::cout << "Eliminated incidences built: D_m forward="
                      << common_dm.row_incidence.size()
                      << ", D_m transpose=" << common_dm.column_incidence.size()
                      << ", elapsed_ms=" << dm_ms << std::endl;
            std::cout << "Eliminated incidences built: D_(m-1) forward="
                      << common_d2.row_incidence.size()
                      << ", D_(m-1) transpose=" << common_d2.column_incidence.size()
                      << ", elapsed_ms=" << d2_ms << std::endl;
            validate_accumulation_terms(
                instance.modulus,
                eliminated_accumulation_terms(common_dm, common_d2, a0_map, a1_map, kernel_map),
                "Eliminated cyclic operator"
            );

            if (operator_name == "augmented" || operator_name == "replacement") {
                run_deformation_eliminated_variant(
                    instance,
                    common_dm,
                    common_d2,
                    a0_map,
                    a1_map,
                    kernel_map,
                    static_cast<size_t>(sector_arg),
                    output_dir,
                    operator_name,
                    dense_filename,
                    drop_columns_filename,
                    num_v,
                    max_nlen_override,
                    save_after,
                    no_save,
                    random_seed,
                    rowmix_options,
                    gram_chunk_size,
                    mod_params
                );
            } else {
                run_eliminated_sector(instance, common_dm, common_d2, a0_map, a1_map, kernel_map,
                                      static_cast<size_t>(sector_arg), output_dir, num_v,
                                      max_nlen_override, save_after, no_save, random_seed,
                                      rowmix_options, validation_options,
                                      gram_chunk_size, mod_params);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cuprym_cyclic error: " << e.what() << std::endl;
        return 1;
    }
}
