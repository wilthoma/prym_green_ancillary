#ifndef BCW_ZSTD_COMPAT_H
#define BCW_ZSTD_COMPAT_H

#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

struct ZSTD_DStream_s;
using ZSTD_DStream = ZSTD_DStream_s;

struct ZSTD_CStream_s;
using ZSTD_CStream = ZSTD_CStream_s;

struct ZSTD_inBuffer {
    const void* src;
    size_t size;
    size_t pos;
};

struct ZSTD_outBuffer {
    void* dst;
    size_t size;
    size_t pos;
};

extern "C" {
ZSTD_DStream* ZSTD_createDStream();
size_t ZSTD_freeDStream(ZSTD_DStream* zds);
size_t ZSTD_initDStream(ZSTD_DStream* zds);
size_t ZSTD_decompressStream(ZSTD_DStream* zds, ZSTD_outBuffer* output, ZSTD_inBuffer* input);
size_t ZSTD_DStreamInSize();
size_t ZSTD_DStreamOutSize();

ZSTD_CStream* ZSTD_createCStream();
size_t ZSTD_freeCStream(ZSTD_CStream* zcs);
size_t ZSTD_initCStream(ZSTD_CStream* zcs, int compressionLevel);
size_t ZSTD_compressStream(ZSTD_CStream* zcs, ZSTD_outBuffer* output, ZSTD_inBuffer* input);
size_t ZSTD_endStream(ZSTD_CStream* zcs, ZSTD_outBuffer* output);
size_t ZSTD_CStreamOutSize();

unsigned int ZSTD_isError(size_t code);
const char* ZSTD_getErrorName(size_t code);
}

namespace bcw_zstd {

constexpr std::array<unsigned char, 4> ZSTD_MAGIC{{0x28, 0xb5, 0x2f, 0xfd}};

inline bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool path_requests_compression(const std::string& filename) {
    return ends_with(filename, ".zst") || ends_with(filename, ".zst.tmp");
}

inline bool stream_has_zstd_magic(std::istream& input) {
    std::array<unsigned char, ZSTD_MAGIC.size()> magic{};
    input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
    const std::streamsize bytes_read = input.gcount();
    input.clear();
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error("Failed to seek WDM file after zstd magic check");
    }
    return bytes_read == static_cast<std::streamsize>(magic.size()) && magic == ZSTD_MAGIC;
}

inline std::streamsize checked_streamsize(size_t size, const std::string& context) {
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(context + " is too large for std::streamsize");
    }
    return static_cast<std::streamsize>(size);
}

class InputBuffer : public std::streambuf {
public:
    explicit InputBuffer(std::istream& input)
        : input_(input),
          input_buffer_(ZSTD_DStreamInSize()),
          output_buffer_(ZSTD_DStreamOutSize())
    {
        dstream_ = ZSTD_createDStream();
        if (!dstream_) {
            throw std::runtime_error("Failed to allocate zstd decompression stream");
        }
        const size_t init_result = ZSTD_initDStream(dstream_);
        if (ZSTD_isError(init_result)) {
            throw std::runtime_error(
                std::string("Failed to initialize zstd decompression stream: ")
                + ZSTD_getErrorName(init_result)
            );
        }
        setg(output_buffer_.data(), output_buffer_.data(), output_buffer_.data());
    }

    ~InputBuffer() override {
        if (dstream_) {
            ZSTD_freeDStream(dstream_);
        }
    }

protected:
    int_type underflow() override {
        if (gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }
        if (frame_finished_) {
            return traits_type::eof();
        }

        while (true) {
            if (input_pos_ == input_size_) {
                input_.read(
                    input_buffer_.data(),
                    checked_streamsize(input_buffer_.size(), "zstd input buffer")
                );
                const std::streamsize got = input_.gcount();
                if (got <= 0) {
                    if (input_.bad()) {
                        throw std::runtime_error("Failed to read compressed WDM data");
                    }
                    throw std::runtime_error(
                        "Compressed WDM ended before the zstd frame was complete"
                    );
                }
                input_size_ = static_cast<size_t>(got);
                input_pos_ = 0;
            }

            ZSTD_inBuffer zstd_input{input_buffer_.data(), input_size_, input_pos_};
            ZSTD_outBuffer zstd_output{output_buffer_.data(), output_buffer_.size(), 0};
            const size_t result = ZSTD_decompressStream(dstream_, &zstd_output, &zstd_input);
            input_pos_ = zstd_input.pos;
            if (ZSTD_isError(result)) {
                throw std::runtime_error(
                    std::string("zstd decompression failed: ") + ZSTD_getErrorName(result)
                );
            }

            if (result == 0) {
                frame_finished_ = true;
                if (zstd_input.pos != zstd_input.size) {
                    throw std::runtime_error("Compressed WDM has trailing bytes after zstd frame");
                }
            }

            if (zstd_output.pos > 0) {
                setg(
                    output_buffer_.data(),
                    output_buffer_.data(),
                    output_buffer_.data() + zstd_output.pos
                );
                return traits_type::to_int_type(*gptr());
            }

            if (frame_finished_) {
                return traits_type::eof();
            }
        }
    }

private:
    std::istream& input_;
    ZSTD_DStream* dstream_ = nullptr;
    std::vector<char> input_buffer_;
    std::vector<char> output_buffer_;
    size_t input_pos_ = 0;
    size_t input_size_ = 0;
    bool frame_finished_ = false;
};

class InputStream {
public:
    explicit InputStream(const std::string& filename)
        : file_(filename, std::ios::binary),
          stream_(&file_)
    {
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        if (stream_has_zstd_magic(file_)) {
            zstd_buffer_ = std::make_unique<InputBuffer>(file_);
            zstd_stream_ = std::make_unique<std::istream>(zstd_buffer_.get());
            stream_ = zstd_stream_.get();
        }
    }

    std::istream& stream() {
        return *stream_;
    }

private:
    std::ifstream file_;
    std::unique_ptr<InputBuffer> zstd_buffer_;
    std::unique_ptr<std::istream> zstd_stream_;
    std::istream* stream_;
};

class OutputStream {
public:
    explicit OutputStream(
        const std::string& filename,
        bool compressed,
        int compression_level = 3
    )
        : file_(filename, std::ios::binary),
          compressed_(compressed)
    {
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        if (compressed_) {
            cstream_ = ZSTD_createCStream();
            if (!cstream_) {
                throw std::runtime_error("Failed to allocate zstd compression stream");
            }
            const size_t init_result = ZSTD_initCStream(cstream_, compression_level);
            if (ZSTD_isError(init_result)) {
                throw std::runtime_error(
                    std::string("Failed to initialize zstd compression stream: ")
                    + ZSTD_getErrorName(init_result)
                );
            }
            output_buffer_.resize(ZSTD_CStreamOutSize());
        }
    }

    ~OutputStream() {
        try {
            finish();
        } catch (...) {
        }
        if (cstream_) {
            ZSTD_freeCStream(cstream_);
        }
    }

    void write_raw(const char* data, size_t size) {
        if (finished_) {
            throw std::runtime_error("Attempted to write to a finished WDM stream");
        }
        if (size == 0) {
            return;
        }

        if (!compressed_) {
            file_.write(data, checked_streamsize(size, "WDM write"));
            return;
        }

        ZSTD_inBuffer input{data, size, 0};
        while (input.pos < input.size) {
            ZSTD_outBuffer output{output_buffer_.data(), output_buffer_.size(), 0};
            const size_t result = ZSTD_compressStream(cstream_, &output, &input);
            if (ZSTD_isError(result)) {
                throw std::runtime_error(
                    std::string("zstd compression failed: ") + ZSTD_getErrorName(result)
                );
            }
            if (output.pos > 0) {
                file_.write(output_buffer_.data(), checked_streamsize(output.pos, "zstd output"));
            }
        }
    }

    void write_text(const char* text) {
        write_raw(text, std::char_traits<char>::length(text));
    }

    void write_space() {
        write_raw(" ", 1);
    }

    void write_newline() {
        write_raw("\n", 1);
    }

    template<typename Number>
    void write_number(Number value) {
        const std::string text = std::to_string(value);
        write_raw(text.data(), text.size());
    }

    void finish() {
        if (finished_) {
            return;
        }

        if (compressed_) {
            size_t remaining = 1;
            while (remaining != 0) {
                ZSTD_outBuffer output{output_buffer_.data(), output_buffer_.size(), 0};
                remaining = ZSTD_endStream(cstream_, &output);
                if (ZSTD_isError(remaining)) {
                    throw std::runtime_error(
                        std::string("zstd finalization failed: ") + ZSTD_getErrorName(remaining)
                    );
                }
                if (output.pos > 0) {
                    file_.write(
                        output_buffer_.data(),
                        checked_streamsize(output.pos, "zstd final output")
                    );
                }
            }
        }

        file_.flush();
        if (!file_) {
            throw std::runtime_error("Failed to write WDM stream");
        }
        finished_ = true;
    }

private:
    std::ofstream file_;
    bool compressed_;
    ZSTD_CStream* cstream_ = nullptr;
    std::vector<char> output_buffer_;
    bool finished_ = false;
};

} // namespace bcw_zstd

#endif // BCW_ZSTD_COMPAT_H
