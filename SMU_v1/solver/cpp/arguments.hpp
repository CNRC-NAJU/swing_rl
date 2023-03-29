#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace Swing {

/* Read argument file and store to single vector */
template <typename T>
const std::vector<T> read_arg_file(
    const std::string& t_arg_file_name, const uint64_t& t_num_args, const T t_dummy
) {
    std::ifstream arg_file(t_arg_file_name);
    std::vector<T> args;
    args.reserve(t_num_args);

    if constexpr (std::is_same<T, float>::value) {
        std::string tmp_data;
        while (getline(arg_file, tmp_data)) {
            args.emplace_back(std::stof(tmp_data));
        }
    } else {
        std::string tmp_data;
        while (getline(arg_file, tmp_data)) {
            args.emplace_back(std::stod(tmp_data));
        }
    }
    arg_file.close();

    return args;
}

}  // namespace Swing
