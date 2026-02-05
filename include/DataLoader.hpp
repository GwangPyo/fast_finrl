#pragma once

#include <string>
#include <memory>
#include <DataFrame/DataFrame.h>

namespace fast_finrl {

using MyDataFrame = hmdf::StdDataFrame<unsigned long>;

class IDataLoader {
public:
    virtual ~IDataLoader() = default;
    virtual void load(const std::string& path, MyDataFrame& df) = 0;
};

class CsvLoader : public IDataLoader {
public:
    void load(const std::string& path, MyDataFrame& df) override;
};

class ParquetLoader : public IDataLoader {
public:
    void load(const std::string& path, MyDataFrame& df) override;
};

// Factory function - auto-detect based on file extension
std::unique_ptr<IDataLoader> create_loader(const std::string& path);

} // namespace fast_finrl
