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

string extract_string_value(const string& text, const string& key) {
    size_t colon = find_json_key_colon(text, key);
    size_t pos = skip_ws(text, colon + 1);
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("JSON key '" + key + "' is not a string");
    }
    ++pos;
    string out;
    bool escape = false;
    for (; pos < text.size(); ++pos) {
        char ch = text[pos];
        if (escape) {
            out.push_back(ch);
            escape = false;
        } else if (ch == '\\') {
            escape = true;
        } else if (ch == '"') {
            return out;
        } else {
            out.push_back(ch);
        }
    }
    throw std::runtime_error("unterminated JSON string for key '" + key + "'");
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

string rowmix_sidecar_filename(const string& wdm_filename) {
    string suffix = ".wdm.zst";
    if (wdm_filename.size() >= suffix.size()
            && wdm_filename.compare(wdm_filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return wdm_filename.substr(0, wdm_filename.size() - suffix.size()) + ".rowmix.json";
    }
    return wdm_filename + ".rowmix.json";
}

void require_condition(bool condition, const string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

my_int normalize_mod_i64(int64_t value, uint32_t prime) {
    int64_t p = static_cast<int64_t>(prime);
    value %= p;
    if (value < 0) {
        value += p;
    }
    return static_cast<my_int>(value);
}

string default_generators_filename(const string& wdm_filename) {
    constexpr const char* compressed_suffix = ".wdm.zst";
    if (wdm_filename.size() >= std::char_traits<char>::length(compressed_suffix)
            && wdm_filename.compare(
                wdm_filename.size() - std::char_traits<char>::length(compressed_suffix),
                std::char_traits<char>::length(compressed_suffix),
                compressed_suffix
            ) == 0) {
        return wdm_filename.substr(
            0,
            wdm_filename.size() - std::char_traits<char>::length(compressed_suffix)
        ) + "_generators.txt";
    }

    constexpr const char* plain_suffix = ".wdm";
    if (wdm_filename.size() >= std::char_traits<char>::length(plain_suffix)
            && wdm_filename.compare(
                wdm_filename.size() - std::char_traits<char>::length(plain_suffix),
                std::char_traits<char>::length(plain_suffix),
                plain_suffix
            ) == 0) {
        return wdm_filename.substr(
            0,
            wdm_filename.size() - std::char_traits<char>::length(plain_suffix)
        ) + "_generators.txt";
    }
    return wdm_filename + "_generators.txt";
}

string default_preconditioned_output_filename(const string& wdm_filename) {
    return wdm_filename + "_nullvectors_1.txt";
}

string default_original_output_filename(const string& wdm_filename) {
    return wdm_filename + "_nullvectors_2.txt";
}

void require_file_exists(const string& filename, const string& label) {
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error(label + " file does not exist: " + filename);
    }
}

vector<my_int> flatten_initial_vectors(
    const vector<vector<my_int>>& v_list,
    size_t vector_length,
    size_t num_v
) {
    require_condition(v_list.size() == num_v, "Initial vector count does not match WDM header");

    vector<my_int> flattened(checked_mul(vector_length, num_v, "flattened initial block"));
    for (size_t col = 0; col < num_v; ++col) {
        require_condition(
            v_list[col].size() == vector_length,
            "Initial vector length does not match WDM matrix column count"
        );
        for (size_t row = 0; row < vector_length; ++row) {
            flattened[row * num_v + col] = v_list[col][row];
        }
    }
    return flattened;
}

size_t load_generators(
    const string& generators_filename,
    size_t num_v,
    uint32_t prime,
    vector<my_int>& flat_generators
) {
    ifstream gen_file(generators_filename);
    if (!gen_file.is_open()) {
        throw std::runtime_error("failed to open generators file " + generators_filename);
    }

    flat_generators.clear();
    string line;
    size_t line_number = 0;
    while (getline(gen_file, line)) {
        ++line_number;
        if (line.find_first_not_of(" \t\r\n") == string::npos) {
            continue;
        }

        istringstream input(line);
        for (size_t i = 0; i < num_v * num_v; ++i) {
            int64_t value = 0;
            if (!(input >> value)) {
                throw std::runtime_error(
                    "generator line " + std::to_string(line_number) + " has too few entries"
                );
            }
            flat_generators.push_back(normalize_mod_i64(value, prime));
        }

        string extra;
        if (input >> extra) {
            throw std::runtime_error(
                "generator line " + std::to_string(line_number)
                    + " has more than num_v*num_v entries"
            );
        }
    }

    const size_t generator_size = checked_mul(num_v, num_v, "generator matrix size");
    require_condition(generator_size > 0, "generator matrix size must be positive");
    require_condition(
        flat_generators.size() % generator_size == 0,
        "generator file does not contain complete generator matrices"
    );
    size_t n_generators = flat_generators.size() / generator_size;
    require_condition(n_generators > 0, "generator file does not contain any generator matrices");
    return n_generators;
}

void validate_preconditioner_vector(
    const vector<my_int>& preconditioner,
    const string& name,
    uint32_t prime
) {
    for (size_t i = 0; i < preconditioner.size(); ++i) {
        if (static_cast<uint32_t>(preconditioner[i]) % prime == 0) {
            throw std::runtime_error(
                name + " has a zero entry modulo the prime at index " + std::to_string(i)
            );
        }
    }
}

vector<size_t> count_column_nonzeros(
    const vector<my_int>& matrix,
    size_t rows,
    size_t cols,
    uint32_t prime
) {
    require_condition(matrix.size() == rows * cols, "dense matrix buffer has unexpected size");

    vector<size_t> counts(cols, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            if (static_cast<uint32_t>(matrix[row * cols + col]) % prime != 0) {
                ++counts[col];
            }
        }
    }
    return counts;
}

size_t total_nonzeros(const vector<size_t>& counts) {
    size_t total = 0;
    for (size_t count : counts) {
        total += count;
    }
    return total;
}

void print_column_counts(const string& label, const vector<size_t>& counts) {
    cout << label;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i > 0) {
            cout << ", ";
        }
        cout << i << ":" << counts[i];
    }
    cout << endl;
}

void print_first_nonzero_entries(
    const vector<my_int>& matrix,
    size_t rows,
    size_t cols,
    uint32_t prime,
    size_t limit
) {
    size_t printed = 0;
    for (size_t row = 0; row < rows && printed < limit; ++row) {
        for (size_t col = 0; col < cols && printed < limit; ++col) {
            uint32_t value = static_cast<uint32_t>(matrix[row * cols + col]) % prime;
            if (value != 0) {
                cout << "  (" << row << ", " << col << ") = " << value << endl;
                ++printed;
            }
        }
    }
}

uint32_t cpu_mod_u64(uint64_t value, uint32_t prime);
uint32_t cpu_mul_mod(uint32_t a, uint32_t b, uint32_t prime);

void save_candidate_vectors(
    const string& filename,
    const vector<my_int>& dense_columns,
    size_t rows,
    size_t cols,
    const vector<my_int>* source_scaling,
    uint32_t prime
) {
    require_condition(dense_columns.size() == rows * cols, "candidate vector buffer has unexpected size");
    if (source_scaling != nullptr) {
        require_condition(
            source_scaling->size() == rows,
            "source preconditioner length does not match candidate vector length"
        );
    }

    std::filesystem::path output_path(filename);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    const string tmp_filename = filename + ".tmp";
    {
        ofstream output(tmp_filename);
        if (!output.is_open()) {
            throw std::runtime_error("failed to open output file " + tmp_filename);
        }

        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t value = static_cast<uint32_t>(dense_columns[row * cols + col]) % prime;
                if (source_scaling != nullptr) {
                    value = cpu_mul_mod(value, static_cast<uint32_t>((*source_scaling)[row]), prime);
                }
                if (row > 0) {
                    output << " ";
                }
                output << value;
            }
            output << "\n";
        }

        output.close();
        if (!output) {
            throw std::runtime_error("failed while writing output file " + tmp_filename);
        }
    }

    std::error_code error;
    std::filesystem::rename(tmp_filename, filename, error);
    if (error) {
        std::filesystem::remove(tmp_filename);
        throw std::runtime_error(
            "failed to rename " + tmp_filename + " to " + filename + ": " + error.message()
        );
    }
}

uint32_t mod_pow_u32(uint32_t base, uint32_t exponent, uint32_t prime) {
    uint64_t result = 1;
    uint64_t value = base % prime;
    while (exponent > 0) {
        if ((exponent & 1u) != 0) {
            result = (result * value) % prime;
        }
        value = (value * value) % prime;
        exponent >>= 1u;
    }
    return static_cast<uint32_t>(result);
}

uint32_t mod_inverse_u32(uint32_t value, uint32_t prime) {
    if (value % prime == 0) {
        throw std::runtime_error("attempted to invert zero modulo prime");
    }
    return mod_pow_u32(value, prime - 2, prime);
}

size_t rank_dense_candidate_columns(
    const vector<my_int>& dense_columns,
    size_t rows,
    size_t cols,
    uint32_t prime,
    vector<size_t>& pivot_columns
) {
    vector<my_int> scratch = dense_columns;
    pivot_columns.clear();
    size_t rank = 0;
    for (size_t col = 0; col < cols && rank < rows; ++col) {
        size_t pivot = rows;
        for (size_t row = rank; row < rows; ++row) {
            if (static_cast<uint32_t>(scratch[row * cols + col]) % prime != 0) {
                pivot = row;
                break;
            }
        }
        if (pivot == rows) {
            continue;
        }
        if (pivot != rank) {
            for (size_t c = 0; c < cols; ++c) {
                std::swap(scratch[rank * cols + c], scratch[pivot * cols + c]);
            }
        }

        uint32_t inv = mod_inverse_u32(static_cast<uint32_t>(scratch[rank * cols + col]), prime);
        for (size_t c = col; c < cols; ++c) {
            scratch[rank * cols + c] = static_cast<my_int>(
                cpu_mul_mod(static_cast<uint32_t>(scratch[rank * cols + c]), inv, prime)
            );
        }

        for (size_t row = 0; row < rows; ++row) {
            if (row == rank) {
                continue;
            }
            uint32_t factor = static_cast<uint32_t>(scratch[row * cols + col]) % prime;
            if (factor == 0) {
                continue;
            }
            for (size_t c = col; c < cols; ++c) {
                uint32_t current = static_cast<uint32_t>(scratch[row * cols + c]) % prime;
                uint32_t pivot_value = static_cast<uint32_t>(scratch[rank * cols + c]) % prime;
                uint32_t subtract = cpu_mul_mod(factor, pivot_value, prime);
                scratch[row * cols + c] = static_cast<my_int>(
                    current >= subtract ? current - subtract : current + prime - subtract
                );
            }
        }

        pivot_columns.push_back(col);
        ++rank;
    }
    return rank;
}

struct ResolvedRowMixOptions {
    RowMixOptions options;
    bool from_sidecar;
    string sidecar_filename;
};

ResolvedRowMixOptions resolve_rowmix_options(
    const string& wdm_filename,
    const string& operator_name,
    size_t sector,
    size_t rows,
    size_t cols,
    size_t num_v,
    uint32_t prime,
    const vector<my_int>& row_precond,
    const vector<my_int>& col_precond,
    const vector<my_int>& initial_block,
    int64_t cli_rounds,
    uint64_t cli_seed,
    bool cli_rounds_supplied,
    bool cli_seed_supplied
) {
    string sidecar = rowmix_sidecar_filename(wdm_filename);
    if (!std::filesystem::exists(sidecar)) {
        if (cli_rounds_supplied && cli_rounds < 0) {
            throw std::runtime_error("--rowmix-rounds must be nonnegative");
        }
        size_t rounds = cli_rounds_supplied ? static_cast<size_t>(cli_rounds) : 0;
        if (rounds > 0) {
            cout << "No rowmix sidecar found; using CLI rowmix settings." << endl;
        }
        return ResolvedRowMixOptions{RowMixOptions{rounds, cli_seed}, false, sidecar};
    }

    string text = read_text_file(sidecar);
    string sidecar_operator = extract_string_value(text, "operator");
    size_t sidecar_sector = checked_size(extract_u64(text, "sector"), "rowmix sidecar sector");
    size_t sidecar_rows = checked_size(extract_u64(text, "rows"), "rowmix sidecar rows");
    size_t sidecar_cols = checked_size(extract_u64(text, "columns"), "rowmix sidecar columns");
    size_t sidecar_num_v = checked_size(extract_u64(text, "vector_count"), "rowmix sidecar vector count");
    uint32_t sidecar_prime = static_cast<uint32_t>(extract_u64(text, "prime"));
    string rowmix_text = extract_object_text(text, "rowmix");
    size_t sidecar_rounds = checked_size(extract_u64(rowmix_text, "rounds"), "rowmix sidecar rounds");
    uint64_t sidecar_seed = extract_u64(rowmix_text, "seed");
    uint64_t sidecar_realized_seed = extract_u64(rowmix_text, "realized_seed");
    size_t sidecar_target_rows =
        checked_size(extract_u64(rowmix_text, "target_rows"), "rowmix sidecar target rows");
    string hashes_text = extract_object_text(text, "hashes");
    string row_hash = extract_string_value(hashes_text, "row_preconditioner_fnv1a64");
    string col_hash = extract_string_value(hashes_text, "column_preconditioner_fnv1a64");
    string initial_hash = extract_string_value(hashes_text, "initial_block_fnv1a64");

    require_condition(sidecar_operator == operator_name, "rowmix sidecar operator does not match CLI operator");
    require_condition(sidecar_sector == sector, "rowmix sidecar sector does not match CLI sector");
    require_condition(sidecar_rows == rows, "rowmix sidecar row count does not match WDM/operator");
    require_condition(sidecar_cols == cols, "rowmix sidecar column count does not match WDM/operator");
    require_condition(sidecar_num_v == num_v, "rowmix sidecar vector count does not match WDM");
    require_condition(sidecar_prime == prime, "rowmix sidecar prime does not match WDM");
    require_condition(sidecar_target_rows == rows, "rowmix sidecar target row count does not match operator");
    require_condition(
        row_hash == hex_u64(fnv1a64_vector_hash(row_precond)),
        "rowmix sidecar row-preconditioner hash does not match WDM"
    );
    require_condition(
        col_hash == hex_u64(fnv1a64_vector_hash(col_precond)),
        "rowmix sidecar column-preconditioner hash does not match WDM"
    );
    require_condition(
        initial_hash == hex_u64(fnv1a64_vector_hash(initial_block)),
        "rowmix sidecar initial-block hash does not match WDM"
    );

    if (cli_rounds_supplied) {
        require_condition(
            cli_rounds >= 0 && static_cast<size_t>(cli_rounds) == sidecar_rounds,
            "CLI --rowmix-rounds conflicts with rowmix sidecar"
        );
    }
    if (cli_seed_supplied) {
        require_condition(cli_seed == sidecar_seed, "CLI --rowmix-seed conflicts with rowmix sidecar");
    }

    RowMixOptions options{sidecar_rounds, sidecar_seed};
    RowMixPlan plan = generate_rowmix_plan(rows, operator_name, sector, prime, options);
    require_condition(
        plan.realized_seed == sidecar_realized_seed,
        "regenerated rowmix realized seed does not match sidecar"
    );
    require_condition(
        hex_u64(rowmix_plan_hash(plan)) == extract_string_value(rowmix_text, "hash_fnv1a64"),
        "regenerated rowmix hash does not match sidecar"
    );

    cout << "Loaded rowmix settings from " << sidecar << endl;
    return ResolvedRowMixOptions{options, true, sidecar};
}

void write_kernel_vector_metadata(
    const string& filename,
    const string& instance_filename,
    const string& wdm_filename,
    const string& generators_filename,
    const string& operator_name,
    size_t sector,
    size_t rows,
    size_t cols,
    size_t num_v,
    uint32_t prime,
    size_t sequence_length,
    size_t generator_count,
    const RowMixPlan& rowmix_plan,
    const vector<size_t>& candidate_nnz,
    const vector<size_t>& product_nnz,
    size_t candidate_rank,
    const vector<size_t>& pivot_columns,
    long long runtime_ms
) {
    std::filesystem::path output_path(filename);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const string tmp_filename = filename + ".tmp";
    ofstream out(tmp_filename);
    if (!out) {
        throw std::runtime_error("failed to open metadata output " + tmp_filename);
    }
    out << "{\n";
    out << "  \"type\": \"prym_cyclic_kernel_vectors_report\",\n";
    out << "  \"version\": 1,\n";
    out << "  \"instance\": \"" << instance_filename << "\",\n";
    out << "  \"wdm_file\": \"" << wdm_filename << "\",\n";
    out << "  \"generators_file\": \"" << generators_filename << "\",\n";
    out << "  \"operator\": \"" << operator_name << "\",\n";
    out << "  \"sector\": " << sector << ",\n";
    out << "  \"rows\": " << rows << ",\n";
    out << "  \"columns\": " << cols << ",\n";
    out << "  \"prime\": " << prime << ",\n";
    out << "  \"vector_count\": " << num_v << ",\n";
    out << "  \"wdm_sequence_length\": " << sequence_length << ",\n";
    out << "  \"generator_count\": " << generator_count << ",\n";
    out << "  \"runtime_ms\": " << runtime_ms << ",\n";
    out << "  \"rowmix\": {\n";
    out << "    \"rounds\": " << rowmix_plan.rounds << ",\n";
    out << "    \"seed\": " << rowmix_plan.seed << ",\n";
    out << "    \"realized_seed\": " << rowmix_plan.realized_seed << ",\n";
    out << "    \"hash_fnv1a64\": \"" << hex_u64(rowmix_plan_hash(rowmix_plan)) << "\"\n";
    out << "  },\n";
    out << "  \"candidate_nonzeros\": [";
    for (size_t i = 0; i < candidate_nnz.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << candidate_nnz[i];
    }
    out << "],\n";
    out << "  \"product_nonzeros\": [";
    for (size_t i = 0; i < product_nnz.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << product_nnz[i];
    }
    out << "],\n";
    out << "  \"candidate_rank\": " << candidate_rank << ",\n";
    out << "  \"pivot_columns\": [";
    for (size_t i = 0; i < pivot_columns.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << pivot_columns[i];
    }
    out << "]\n";
    out << "}\n";
    out.close();
    if (!out) {
        throw std::runtime_error("failed while writing metadata output " + tmp_filename);
    }
    std::error_code error;
    std::filesystem::rename(tmp_filename, filename, error);
    if (error) {
        std::filesystem::remove(tmp_filename);
        throw std::runtime_error(
            "failed to rename " + tmp_filename + " to " + filename + ": " + error.message()
        );
    }
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

template<typename T, int DENSE_COLS>
__global__ void kernel_vector_accumulate_generator_slice_kernel(
    size_t rows,
    const T* __restrict__ B,
    const T* __restrict__ generators,
    T* __restrict__ D,
    size_t slice_index,
    ModulusParams mod
) {
    size_t row = blockIdx.x * blockDim.y + threadIdx.y;
    int dense_col = threadIdx.x;
    if (row >= rows || dense_col >= DENSE_COLS) {
        return;
    }

    const T* G = generators + slice_index * DENSE_COLS * DENSE_COLS;
    uint64_t sum = static_cast<uint32_t>(D[row * DENSE_COLS + dense_col]);
    for (int i = 0; i < DENSE_COLS; ++i) {
        uint32_t b = static_cast<uint32_t>(B[row * DENSE_COLS + i]);
        uint32_t g = static_cast<uint32_t>(G[dense_col * DENSE_COLS + i]);
        sum += static_cast<uint64_t>(b) * static_cast<uint64_t>(g);
    }
    D[row * DENSE_COLS + dense_col] =
        static_cast<T>(cyclic_reduce_mod_u64(sum, mod));
}

template<typename T, int DENSE_COLS>
void launch_kernel_vector_accumulate_generator_slice_static(
    const CudaDenseMatrix<T>& B,
    const CudaDenseMatrix<T>& generators,
    CudaDenseMatrix<T>& D,
    size_t slice_index,
    ModulusParams mod
) {
    const unsigned int block_cols = dense_column_block_width(DENSE_COLS);
    const unsigned int block_rows = std::max(1u, 256u / block_cols);
    dim3 blockDim(block_cols, block_rows);
    dim3 gridDim((B.numRows + block_rows - 1) / block_rows);
    kernel_vector_accumulate_generator_slice_kernel<T, DENSE_COLS><<<gridDim, blockDim>>>(
        B.numRows,
        B.d_data,
        generators.d_data,
        D.d_data,
        slice_index,
        mod
    );
    CHECK_CUDA(cudaGetLastError());
}

template<typename T>
void accumulate_generator_slice(
    const CudaDenseMatrix<T>& B,
    const CudaDenseMatrix<T>& generators,
    CudaDenseMatrix<T>& D,
    size_t slice_index,
    ModulusParams mod
) {
    if (B.numRows != D.numRows || B.numCols != D.numCols) {
        throw std::runtime_error("kernel-vector generator accumulation dimensions do not match");
    }
    if (generators.numRows <= slice_index
            || generators.numCols != checked_mul(B.numCols, B.numCols, "generator slice width")) {
        throw std::runtime_error("kernel-vector generator storage dimensions do not match");
    }
    switch (B.numCols) {
        case 1: launch_kernel_vector_accumulate_generator_slice_static<T, 1>(B, generators, D, slice_index, mod); break;
        case 2: launch_kernel_vector_accumulate_generator_slice_static<T, 2>(B, generators, D, slice_index, mod); break;
        case 3: launch_kernel_vector_accumulate_generator_slice_static<T, 3>(B, generators, D, slice_index, mod); break;
        case 4: launch_kernel_vector_accumulate_generator_slice_static<T, 4>(B, generators, D, slice_index, mod); break;
        case 5: launch_kernel_vector_accumulate_generator_slice_static<T, 5>(B, generators, D, slice_index, mod); break;
        case 6: launch_kernel_vector_accumulate_generator_slice_static<T, 6>(B, generators, D, slice_index, mod); break;
        case 7: launch_kernel_vector_accumulate_generator_slice_static<T, 7>(B, generators, D, slice_index, mod); break;
        case 8: launch_kernel_vector_accumulate_generator_slice_static<T, 8>(B, generators, D, slice_index, mod); break;
        case 9: launch_kernel_vector_accumulate_generator_slice_static<T, 9>(B, generators, D, slice_index, mod); break;
        case 10: launch_kernel_vector_accumulate_generator_slice_static<T, 10>(B, generators, D, slice_index, mod); break;
        case 11: launch_kernel_vector_accumulate_generator_slice_static<T, 11>(B, generators, D, slice_index, mod); break;
        case 12: launch_kernel_vector_accumulate_generator_slice_static<T, 12>(B, generators, D, slice_index, mod); break;
        case 13: launch_kernel_vector_accumulate_generator_slice_static<T, 13>(B, generators, D, slice_index, mod); break;
        case 14: launch_kernel_vector_accumulate_generator_slice_static<T, 14>(B, generators, D, slice_index, mod); break;
        case 15: launch_kernel_vector_accumulate_generator_slice_static<T, 15>(B, generators, D, slice_index, mod); break;
        case 16: launch_kernel_vector_accumulate_generator_slice_static<T, 16>(B, generators, D, slice_index, mod); break;
        default:
            throw std::runtime_error("kernel-vector generator accumulation supports -v values from 1 to 16");
    }
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

template<typename ApplySquare, typename ApplyForward>
void recover_kernel_vectors_core(
    const string& instance_filename,
    const string& wdm_filename,
    const string& generators_filename,
    const string& preconditioned_output_filename,
    const string& original_output_filename,
    const string& metadata_output_filename,
    const string& operator_name,
    size_t sector,
    size_t rows,
    size_t cols,
    uint32_t prime,
    const vector<my_int>& row_precond,
    const vector<my_int>& col_precond,
    const vector<my_int>& initial_block,
    size_t num_v,
    size_t sequence_length,
    const vector<my_int>& flat_generators,
    size_t generator_count,
    const RowMixPlan& rowmix_plan,
    ApplySquare apply_square,
    ApplyForward apply_forward
) {
    cout << "Recovery dimensions: rows=" << rows
         << ", columns=" << cols
         << ", v=" << num_v
         << ", WDM sequence length=" << sequence_length
         << ", generator matrices=" << generator_count
         << endl;
    if (generator_count > sequence_length) {
        cout << "Warning: generator count exceeds WDM sequence length; continuing because rust_rank exported it." << endl;
    }

    CudaDenseMatrix<my_int> cuB = CudaDenseMatrix<my_int>::from_host(initial_block, cols, num_v);
    CudaDenseMatrix<my_int> cuNext = CudaDenseMatrix<my_int>::allocate_uninitialized(cols, num_v);
    CudaDenseMatrix<my_int> cuD = CudaDenseMatrix<my_int>::allocate(cols, num_v, 0);
    CudaDenseMatrix<my_int> cuForwardProduct = CudaDenseMatrix<my_int>::allocate_uninitialized(rows, num_v);
    CudaDenseMatrix<my_int> cuGenerators =
        CudaDenseMatrix<my_int>::from_host(flat_generators, generator_count, num_v * num_v);

    cout << "Dense recovery memory: B/next/candidates="
         << fixed << setprecision(2) << to_mib(cols * num_v * sizeof(my_int))
         << " MiB each, forward product="
         << to_mib(rows * num_v * sizeof(my_int))
         << " MiB, generators="
         << to_mib(flat_generators.size() * sizeof(my_int))
         << " MiB" << endl;

    const auto computation_start = std::chrono::high_resolution_clock::now();
    long long last_report = 0;
    size_t last_round = 0;
    const long long report_interval_ms = 1000;

    for (size_t round = 0; round < generator_count; ++round) {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - computation_start).count();
        if (elapsed - last_report > report_interval_ms) {
            double seconds = static_cast<double>(elapsed - last_report) / 1000.0;
            double speed = seconds > 0.0 ? static_cast<double>(round - last_round) / seconds : 0.0;
            double remaining = speed > 0.0 ? static_cast<double>(generator_count - round) / speed : 0.0;
            cout << "\rKernel-vector recovery: " << round << "/" << generator_count
                 << " | elapsed " << elapsed / 1000 << "s"
                 << " | throughput " << fixed << setprecision(2) << speed << "/s"
                 << " | remaining " << setprecision(1) << remaining << "s        " << flush;
            last_round = round;
            last_report = elapsed;
        }

        accumulate_generator_slice(cuB, cuGenerators, cuD, round, make_modulus_params(prime));
        if (round + 1 < generator_count) {
            apply_square(cuB, cuNext);
            std::swap(cuB.d_data, cuNext.d_data);
        }
    }
    cout << "\rKernel-vector recovery: " << generator_count << "/"
         << generator_count << " complete.                       " << endl;

    apply_forward(cuD, cuForwardProduct);
    CHECK_CUDA(cudaDeviceSynchronize());

    const auto computation_stop = std::chrono::high_resolution_clock::now();
    const auto runtime_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(computation_stop - computation_start).count();

    vector<my_int> product = copy_cuda_dense_to_host(cuForwardProduct);
    vector<size_t> product_nnz = count_column_nonzeros(product, rows, num_v, prime);
    print_column_counts("A*(source-preconditioned candidate) nonzeros by vector: ", product_nnz);
    if (total_nonzeros(product_nnz) != 0) {
        cout << "First nonzero entries of certificate product:" << endl;
        print_first_nonzero_entries(product, rows, num_v, prime, 10);
        throw std::runtime_error("kernel-vector certificate failed");
    }
    cout << "Kernel-vector certificate passed." << endl;

    vector<my_int> candidates = copy_cuda_dense_to_host(cuD);
    vector<size_t> candidate_nnz = count_column_nonzeros(candidates, cols, num_v, prime);
    print_column_counts("Candidate nonzeros by vector: ", candidate_nnz);
    require_condition(
        total_nonzeros(candidate_nnz) != 0,
        "all candidate vectors are zero; no kernel vector was produced"
    );

    vector<size_t> pivot_columns;
    size_t candidate_rank =
        rank_dense_candidate_columns(candidates, cols, num_v, prime, pivot_columns);
    cout << "Candidate block rank: " << candidate_rank << "/" << num_v << endl;
    cout << "Independent candidate columns:";
    for (size_t col : pivot_columns) {
        cout << " " << col;
    }
    cout << endl;

    cout << "GPU recovery runtime: " << runtime_ms << " ms" << endl;
    if (runtime_ms > 0) {
        cout << "Generator throughput: "
             << fixed << setprecision(2)
             << generator_count * 1000.0 / static_cast<double>(runtime_ms)
             << "/s" << endl;
    }

    cout << "Saving candidate vectors..." << endl;
    save_candidate_vectors(preconditioned_output_filename, candidates, cols, num_v, nullptr, prime);
    save_candidate_vectors(original_output_filename, candidates, cols, num_v, &col_precond, prime);
    write_kernel_vector_metadata(
        metadata_output_filename,
        instance_filename,
        wdm_filename,
        generators_filename,
        operator_name,
        sector,
        rows,
        cols,
        num_v,
        prime,
        sequence_length,
        generator_count,
        rowmix_plan,
        candidate_nnz,
        product_nnz,
        candidate_rank,
        pivot_columns,
        runtime_ms
    );
    cout << "Saved preconditioned candidates to " << preconditioned_output_filename << endl;
    cout << "Saved original-coordinate candidates to " << original_output_filename << endl;
    cout << "Saved metadata to " << metadata_output_filename << endl;

    cuB.release();
    cuNext.release();
    cuD.release();
    cuForwardProduct.release();
    cuGenerators.release();
}

void recover_eliminated_kernel_vectors(
    const CyclicInstance& instance,
    const CyclicCommonIncidence& common_dm,
    const CyclicCommonIncidence& common_d2,
    const CharacterMap& a0_map,
    const CharacterMap& a1_map,
    const CharacterMap& kernel_map,
    size_t sector,
    const string& instance_filename,
    const string& wdm_filename,
    const string& generators_filename,
    const string& preconditioned_output_filename,
    const string& original_output_filename,
    const string& metadata_output_filename,
    int64_t cli_rowmix_rounds,
    uint64_t cli_rowmix_seed,
    bool cli_rowmix_rounds_supplied,
    bool cli_rowmix_seed_supplied,
    const ValidationOptions& validation_options,
    ModulusParams mod_params
) {
    size_t y_sector = sub_mod(sector, instance.elimination_w_weight, instance.r);
    cout << "Building eliminated sector " << sector
         << " layouts: y_sector=" << y_sector << "..." << endl;
    CyclicSectorLayout dm_layout = build_sector_layout_generic(
        instance,
        common_dm,
        a0_map,
        a1_map,
        sector,
        "eliminated D_m"
    );
    CyclicSectorLayout d2_layout = build_sector_layout_generic(
        instance,
        common_d2,
        a0_map,
        a1_map,
        y_sector,
        "eliminated D_(m-1)"
    );
    CyclicTensorLayout z_layout = build_tensor_layout(
        common_d2.column_subset_weights,
        kernel_map,
        instance.r,
        y_sector,
        "eliminated kernel tensor"
    );

    size_t x_cols = dm_layout.column_descriptors.size();
    size_t z_cols = z_layout.descriptors.size();
    size_t total_cols = x_cols + z_cols;
    size_t target_rows = d2_layout.row_descriptors.size();
    validate_eliminated_sector_against_fixture(instance, sector, total_cols, target_rows);

    vector<my_int> row_precond;
    vector<my_int> col_precond;
    vector<vector<my_int>> v_list;
    size_t wdm_rows = 0;
    size_t wdm_cols = 0;
    size_t num_v = 0;
    size_t sequence_length = 0;
    uint32_t wdm_prime = 0;
    tie(wdm_prime, wdm_rows, wdm_cols, num_v, sequence_length) =
        load_wdm_initial_state(wdm_filename, row_precond, col_precond, v_list);
    require_condition(wdm_prime == instance.modulus, "WDM prime does not match instance modulus");
    require_condition(wdm_rows == target_rows, "WDM row count does not match eliminated sector");
    require_condition(wdm_cols == total_cols, "WDM column count does not match eliminated sector");
    require_condition(num_v > 0 && num_v <= 16, "WDM vector count must be in 1..16");
    validate_preconditioner_vector(row_precond, "row preconditioner", instance.modulus);
    validate_preconditioner_vector(col_precond, "column preconditioner", instance.modulus);
    vector<my_int> initial_block = flatten_initial_vectors(v_list, wdm_cols, num_v);

    ResolvedRowMixOptions resolved_rowmix = resolve_rowmix_options(
        wdm_filename,
        "eliminated",
        sector,
        wdm_rows,
        wdm_cols,
        num_v,
        instance.modulus,
        row_precond,
        col_precond,
        initial_block,
        cli_rowmix_rounds,
        cli_rowmix_seed,
        cli_rowmix_rounds_supplied,
        cli_rowmix_seed_supplied
    );
    RowMixPlan rowmix_plan = generate_rowmix_plan(
        wdm_rows,
        "eliminated",
        sector,
        instance.modulus,
        resolved_rowmix.options
    );

    vector<my_int> flat_generators;
    size_t generator_count = load_generators(generators_filename, num_v, instance.modulus, flat_generators);

    vector<my_int> dm_row_ones = ones_vector(dm_layout.row_descriptors.size());
    vector<my_int> dm_col_ones = ones_vector(x_cols);
    vector<my_int> d2_col_ones = ones_vector(d2_layout.column_descriptors.size());
    vector<my_int> d2_row_precond_for_operator =
        rowmix_plan.enabled() ? ones_vector(target_rows) : row_precond;

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

    cout << "Loading eliminated data onto GPU: D_m="
         << fixed << setprecision(2)
         << to_mib(cyclic_phi_host_data_memory_size(dm_host))
         << " MiB, D_(m-1)="
         << to_mib(cyclic_phi_host_data_memory_size(d2_host))
         << " MiB, z layout="
         << to_mib(z_layout.offsets.size() * sizeof(uint32_t)
             + z_layout.descriptors.size() * sizeof(CyclicColumnDescriptor))
         << " MiB" << endl;

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

    CudaDenseMatrix<my_int> cuScaled = CudaDenseMatrix<my_int>::allocate_uninitialized(total_cols, num_v);
    CudaDenseMatrix<my_int> cuTarget = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
    CudaDenseMatrix<my_int> cuT = CudaDenseMatrix<my_int>::allocate_uninitialized(dm_host.numRows, num_v);
    CudaDenseMatrix<my_int> cuY = CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numCols, num_v);
    CudaDenseMatrix<my_int> cuH = CudaDenseMatrix<my_int>::allocate_uninitialized(d2_host.numCols, num_v);
    CudaDenseMatrix<my_int> cuTAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(dm_host.numRows, num_v);
    CudaDenseMatrix<my_int> cuXAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(x_cols, num_v);
    CudaDenseMatrix<my_int> cuZAdj = CudaDenseMatrix<my_int>::allocate_uninitialized(z_cols, num_v);
    CudaDenseMatrix<my_int> cuTargetMixed;
    if (rowmix_plan.enabled()) {
        cuTargetMixed = CudaDenseMatrix<my_int>::allocate_uninitialized(target_rows, num_v);
    }

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

    auto apply_forward = [&](CudaDenseMatrix<my_int>& input, CudaDenseMatrix<my_int>& output) {
        cyclic_launch_scale_rows(input, dSourcePrecond, cuScaled, mod_params);
        CudaDenseMatrix<my_int> scaled_x = cyclic_dense_view(cuScaled.d_data, x_cols, num_v);
        CudaDenseMatrix<my_int> scaled_z =
            cyclic_dense_view(cuScaled.d_data + x_cols * num_v, z_cols, num_v);
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

    recover_kernel_vectors_core(
        instance_filename,
        wdm_filename,
        generators_filename,
        preconditioned_output_filename,
        original_output_filename,
        metadata_output_filename,
        "eliminated",
        sector,
        wdm_rows,
        wdm_cols,
        instance.modulus,
        row_precond,
        col_precond,
        initial_block,
        num_v,
        sequence_length,
        flat_generators,
        generator_count,
        rowmix_plan,
        apply_square,
        apply_forward
    );

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
        cuTargetMixed.release();
    }
    cuScaled.release();
    cuTarget.release();
    cuT.release();
    cuY.release();
    cuH.release();
    cuTAdj.release();
    cuXAdj.release();
    cuZAdj.release();
}

int main(int argc, char* argv[]) {
    try {
        CLI::App app{"CUDA matrix-free cyclic Prym-Green kernel-vector recovery"};

        string instance_filename;
        string wdm_filename;
        string generators_filename;
        string preconditioned_output_filename;
        string original_output_filename;
        string metadata_output_filename;
        string operator_name = "eliminated";
        int sector_arg = -1;
        int cuda_device_id = 0;
        int64_t rowmix_rounds = 2;
        uint64_t rowmix_seed = 10001;
        bool validate_rowmix = false;
        size_t validate_columns = 4;

        app.add_option("instance", instance_filename, "Cyclic Prym fixture/instance JSON")->required();
        app.add_option("-f,--wdm-file", wdm_filename, "WDM file used for the rank computation")->required();
        app.add_option(
            "-g,--generators-file",
            generators_filename,
            "Generator file produced by prym-rank --generator; defaults to WDM stem + _generators.txt"
        );
        app.add_option(
            "--preconditioned-output",
            preconditioned_output_filename,
            "Output file for kernel candidates in WDM/preconditioned source coordinates"
        );
        app.add_option(
            "--original-output",
            original_output_filename,
            "Output file for kernel candidates in original source coordinates"
        );
        app.add_option(
            "--metadata-output",
            metadata_output_filename,
            "JSON report for certificate, candidate rank, hashes, and provenance"
        );
        app.add_option("--operator", operator_name, "Production operator: eliminated.")->default_val("eliminated");
        app.add_option("--sector", sector_arg, "Cyclic sector/character.")->required();
        app.add_option("-d,--device", cuda_device_id, "Selects a CUDA device to use.")->default_val(0);
        auto* rowmix_rounds_option =
            app.add_option("--rowmix-rounds", rowmix_rounds, "Fallback row-mixing rounds when no WDM sidecar exists.");
        auto* rowmix_seed_option =
            app.add_option("--rowmix-seed", rowmix_seed, "Fallback row-mixing seed when no WDM sidecar exists.")->default_val(10001);
        app.add_flag("--validate-rowmix", validate_rowmix, "Run focused CPU/CUDA square-operator validation before recovery.");

        CLI11_PARSE(app, argc, argv);

        if (rowmix_rounds != 2 || rowmix_seed != 10001) {
            throw std::runtime_error("The paper pipeline requires --rowmix-rounds 2 --rowmix-seed 10001.");
        }

        if (operator_name != "eliminated") {
            cerr << "--operator must be eliminated." << endl;
            return 1;
        }
        if (sector_arg < 0) {
            cerr << "--sector is required and must be nonnegative." << endl;
            return 1;
        }

        if (rowmix_rounds_option->count() > 0 && rowmix_rounds < 0) {
            cerr << "--rowmix-rounds must be nonnegative." << endl;
            return 1;
        }

        if (generators_filename.empty()) {
            generators_filename = default_generators_filename(wdm_filename);
        }
        if (preconditioned_output_filename.empty()) {
            preconditioned_output_filename = default_preconditioned_output_filename(wdm_filename);
        }
        if (original_output_filename.empty()) {
            original_output_filename = default_original_output_filename(wdm_filename);
        }
        if (metadata_output_filename.empty()) {
            metadata_output_filename = wdm_filename + "_nullvectors_report.json";
        }

        require_file_exists(instance_filename, "Instance");
        require_file_exists(wdm_filename, "WDM");
        require_file_exists(generators_filename, "Generators");

        cout << "Loading cyclic Prym instance JSON: " << instance_filename << endl;
        CyclicInstance instance = load_cyclic_instance(instance_filename);
        size_t sector = static_cast<size_t>(sector_arg);
        if (sector >= instance.r) {
            cerr << "--sector must be in 0.." << instance.r - 1 << endl;
            return 1;
        }
        if (!is_prime(instance.modulus)) {
            cerr << "Modulus " << instance.modulus << " is not prime." << endl;
            return 1;
        }

        CHECK_CUDA(cudaSetDevice(cuda_device_id));
        cudaDeviceProp device_prop;
        CHECK_CUDA(cudaGetDeviceProperties(&device_prop, cuda_device_id));
        cout << "Using CUDA device " << cuda_device_id << ": " << device_prop.name
             << ", global memory " << fixed << setprecision(2)
             << to_mib(device_prop.totalGlobalMem) << " MiB" << endl;

        cout << "Cyclic Prym data: g=" << instance.genus
             << ", r=" << instance.r
             << ", n=" << instance.n
             << ", m=" << instance.m
             << ", modulus=" << instance.modulus
             << ", operator=" << operator_name
             << ", sector=" << sector
             << endl;

        CharacterMap a0_map = build_character_map(instance.weight_a0, instance.r, "A0");
        CharacterMap a1_map = build_character_map(instance.weight_a1, instance.r, "A1");
        ModulusParams mod_params = make_modulus_params(instance.modulus);
        ValidationOptions validation_options{validate_rowmix, validate_columns};
        bool cli_rowmix_rounds_supplied = rowmix_rounds_option->count() > 0;
        bool cli_rowmix_seed_supplied = rowmix_seed_option->count() > 0;

        {
            if (!instance.has_elimination) {
                cerr << "Instance does not contain third_section_elimination data." << endl;
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
            cout << "Building eliminated common incidences..." << endl;
            CyclicCommonIncidence common_dm = build_common_incidences_generic(
                u_indices.size(),
                instance.m,
                instance.r,
                u_indices,
                u_weights,
                "eliminated D_m"
            );
            CyclicCommonIncidence common_d2 = build_common_incidences_generic(
                u_indices.size(),
                instance.m - 1,
                instance.r,
                u_indices,
                u_weights,
                "eliminated D_(m-1)"
            );
            validate_accumulation_terms(
                instance.modulus,
                eliminated_accumulation_terms(common_dm, common_d2, a0_map, a1_map, kernel_map),
                "Eliminated cyclic operator"
            );
            recover_eliminated_kernel_vectors(
                instance,
                common_dm,
                common_d2,
                a0_map,
                a1_map,
                kernel_map,
                sector,
                instance_filename,
                wdm_filename,
                generators_filename,
                preconditioned_output_filename,
                original_output_filename,
                metadata_output_filename,
                rowmix_rounds,
                rowmix_seed,
                cli_rowmix_rounds_supplied,
                cli_rowmix_seed_supplied,
                validation_options,
                mod_params
            );
        }

        return 0;
    } catch (const std::exception& e) {
        cerr << "cuprym_cyclic_kernel_vectors error: " << e.what() << endl;
        return 1;
    }
}
