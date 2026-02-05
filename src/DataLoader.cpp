#include "DataLoader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <future>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>

namespace fast_finrl {

using namespace std;

// ============================================================================
// CsvLoader Implementation
// ============================================================================

void CsvLoader::load(const string& path, MyDataFrame& df) {
    ifstream infile(path);
    if (!infile.is_open()) {
        throw runtime_error("Cannot open file: " + path);
    }

    // Read header line
    string header_line;
    getline(infile, header_line);

    // Parse column names
    vector<string> col_names;
    stringstream header_ss(header_line);
    string col_name;
    while (getline(header_ss, col_name, ',')) {
        col_names.push_back(col_name);
    }

    size_t num_cols = col_names.size();

    // Prepare column storage
    vector<vector<string>> string_cols(num_cols);
    vector<vector<double>> double_cols(num_cols);
    vector<bool> is_string_col(num_cols, false);

    // Identify string columns
    for (size_t i = 0; i < num_cols; ++i) {
        if (col_names[i] == "date" || col_names[i] == "tic") {
            is_string_col[i] = true;
        }
    }

    // Read all rows
    string line;
    size_t row_count = 0;
    while (getline(infile, line)) {
        if (line.empty()) continue;

        stringstream line_ss(line);
        string cell;
        size_t col_idx = 0;

        while (getline(line_ss, cell, ',') && col_idx < num_cols) {
            if (is_string_col[col_idx]) {
                string_cols[col_idx].push_back(cell);
            } else {
                double_cols[col_idx].push_back(cell.empty() ? 0.0 : stod(cell));
            }
            col_idx++;
        }
        row_count++;
    }
    infile.close();

    // Create index
    vector<unsigned long> index(row_count);
    for (size_t i = 0; i < row_count; ++i) index[i] = i;
    df.load_index(move(index));

    // Load columns into DataFrame
    for (size_t i = 0; i < num_cols; ++i) {
        if (is_string_col[i]) {
            df.load_column(col_names[i].c_str(), move(string_cols[i]));
        } else {
            df.load_column(col_names[i].c_str(), move(double_cols[i]));
        }
    }
}

// ============================================================================
// ParquetLoader Implementation
// ============================================================================

namespace {
    // Template to extract column data from Arrow chunked array
    template<typename ArrowType, typename CppType>
    vector<CppType> extract_column(const shared_ptr<arrow::ChunkedArray>& chunked, size_t num_rows) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        vector<CppType> col_data;
        col_data.reserve(num_rows);
        for (int c = 0; c < chunked->num_chunks(); ++c) {
            auto array = static_pointer_cast<ArrayType>(chunked->chunk(c));
            for (int64_t j = 0; j < array->length(); ++j) {
                if constexpr (is_same_v<CppType, string>) {
                    col_data.push_back(string(array->GetView(j)));
                } else {
                    col_data.push_back(static_cast<CppType>(array->Value(j)));
                }
            }
        }
        return col_data;
    }

    // Convert numeric to string
    template<typename ArrowType>
    vector<string> extract_as_string(const shared_ptr<arrow::ChunkedArray>& chunked, size_t num_rows) {
        using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
        vector<string> col_data;
        col_data.reserve(num_rows);
        for (int c = 0; c < chunked->num_chunks(); ++c) {
            auto array = static_pointer_cast<ArrayType>(chunked->chunk(c));
            for (int64_t j = 0; j < array->length(); ++j) {
                col_data.push_back(to_string(array->Value(j)));
            }
        }
        return col_data;
    }
}

void ParquetLoader::load(const string& path, MyDataFrame& df) {
    PARQUET_ASSIGN_OR_THROW(auto infile, arrow::io::ReadableFile::Open(path));

    // Create ParquetFileReader first
    auto parquet_reader = parquet::ParquetFileReader::Open(infile);

    // Enable parallel column reading
    parquet::ArrowReaderProperties props(true);  // use_threads = true
    props.set_pre_buffer(true);  // Pre-buffer for better I/O

    // Create FileReader with properties
    PARQUET_ASSIGN_OR_THROW(auto reader,
        parquet::arrow::FileReader::Make(arrow::default_memory_pool(), move(parquet_reader), props));
    reader->set_use_threads(true);  // Enable parallel decoding

    shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    size_t num_rows = table->num_rows();
    int num_cols = table->num_columns();

    // Create index
    vector<unsigned long> index(num_rows);
    for (size_t i = 0; i < num_rows; ++i) index[i] = i;
    df.load_index(move(index));

    // Pre-extract column info
    vector<string> col_names(num_cols);
    vector<arrow::Type::type> col_types(num_cols);
    for (int i = 0; i < num_cols; ++i) {
        col_names[i] = table->field(i)->name();
        col_types[i] = table->column(i)->type()->id();
    }

    // Extract columns in parallel using std::async
    vector<vector<double>> double_cols(num_cols);
    vector<vector<string>> string_cols(num_cols);
    vector<bool> is_string_col(num_cols, false);
    vector<future<void>> futures;
    futures.reserve(num_cols);

    for (int i = 0; i < num_cols; ++i) {
        futures.push_back(async(launch::async, [&, i]() {
            auto chunked = table->column(i);
            auto type_id = col_types[i];
            const auto& col_name = col_names[i];

            if (col_name == "date" || col_name == "tic") {
                is_string_col[i] = true;
                if (type_id == arrow::Type::STRING) {
                    string_cols[i] = extract_column<arrow::StringType, string>(chunked, num_rows);
                } else if (type_id == arrow::Type::INT64) {
                    string_cols[i] = extract_as_string<arrow::Int64Type>(chunked, num_rows);
                }
            } else {
                if (type_id == arrow::Type::DOUBLE) {
                    double_cols[i] = extract_column<arrow::DoubleType, double>(chunked, num_rows);
                } else if (type_id == arrow::Type::INT64) {
                    double_cols[i] = extract_column<arrow::Int64Type, double>(chunked, num_rows);
                } else if (type_id == arrow::Type::FLOAT) {
                    double_cols[i] = extract_column<arrow::FloatType, double>(chunked, num_rows);
                }
            }
        }));
    }

    // Wait for all extractions to complete
    for (auto& f : futures) f.get();

    // Load columns into DataFrame (sequential - DataFrame not thread-safe)
    for (int i = 0; i < num_cols; ++i) {
        if (is_string_col[i]) {
            df.load_column(col_names[i].c_str(), move(string_cols[i]));
        } else {
            df.load_column(col_names[i].c_str(), move(double_cols[i]));
        }
    }
}

// ============================================================================
// Factory Function
// ============================================================================

unique_ptr<IDataLoader> create_loader(const string& path) {
    bool is_parquet = (path.size() >= 8 && path.substr(path.size() - 8) == ".parquet");

    if (is_parquet) {
        return make_unique<ParquetLoader>();
    } else {
        return make_unique<CsvLoader>();
    }
}

} // namespace fast_finrl
