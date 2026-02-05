#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/writer.h>
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> [output.parquet]" << std::endl;
        return 1;
    }

    std::string csv_path = argv[1];
    std::string parquet_path = argc > 2 ? argv[2] : csv_path.substr(0, csv_path.rfind('.')) + ".parquet";

    std::cout << "Converting " << csv_path << " to " << parquet_path << "..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Open CSV file
    auto maybe_input = arrow::io::ReadableFile::Open(csv_path);
    if (!maybe_input.ok()) {
        std::cerr << "Error opening file: " << maybe_input.status().message() << std::endl;
        return 1;
    }
    auto input = *maybe_input;

    // Configure CSV reader
    auto read_options = arrow::csv::ReadOptions::Defaults();
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    auto convert_options = arrow::csv::ConvertOptions::Defaults();

    // Create CSV reader
    auto maybe_reader = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(),
        input,
        read_options,
        parse_options,
        convert_options
    );
    if (!maybe_reader.ok()) {
        std::cerr << "Error creating reader: " << maybe_reader.status().message() << std::endl;
        return 1;
    }
    auto reader = *maybe_reader;

    // Read CSV into Arrow Table
    auto maybe_table = reader->Read();
    if (!maybe_table.ok()) {
        std::cerr << "Error reading CSV: " << maybe_table.status().message() << std::endl;
        return 1;
    }
    auto table = *maybe_table;

    auto read_time = std::chrono::high_resolution_clock::now();
    std::cout << "  Read CSV: " << std::chrono::duration<double>(read_time - start).count()
              << "s (" << table->num_rows() << " rows, " << table->num_columns() << " columns)" << std::endl;

    // Write Parquet file
    auto maybe_output = arrow::io::FileOutputStream::Open(parquet_path);
    if (!maybe_output.ok()) {
        std::cerr << "Error creating output: " << maybe_output.status().message() << std::endl;
        return 1;
    }
    auto output = *maybe_output;

    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output, 1024 * 1024);
    if (!status.ok()) {
        std::cerr << "Error writing Parquet: " << status.message() << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "  Total time: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
    std::cout << "  Output: " << parquet_path << std::endl;

    return 0;
}
