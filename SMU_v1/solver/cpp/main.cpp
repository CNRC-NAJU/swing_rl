#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "arguments.hpp"
#include "parameters.hpp"
#include "solver.hpp"

namespace Swing{

template <typename T>
void solve(
    const std::string& t_solver_name,
    const uint64_t& t_num_nodes,
    const uint64_t& t_num_edges,
    const uint64_t& t_num_steps,
    const std::string& t_arg_file_name,
    const T t_dummy
) {
    //* Read argument file and store to a single vector
    const auto args = read_arg_file(
        t_arg_file_name, 5 * t_num_nodes + 3 * t_num_edges + t_num_steps, t_dummy
    );
    const Parameters params(args, t_num_nodes, t_num_edges, t_num_steps);

    //* Run Runge-Kutta solver
    std::vector<std::vector<T>> trajectories;
    if (t_solver_name.find("rk1") != std::string::npos) {
        trajectories = Swing::solve_rk1(
            params.weighted_edge_list,
            {params.phase, params.dphase},
            {params.power, params.gamma, params.mass},
            params.dts
        );
    } else if (t_solver_name.find("rk2") != std::string::npos) {
        trajectories = Swing::solve_rk2(
            params.weighted_edge_list,
            {params.phase, params.dphase},
            {params.power, params.gamma, params.mass},
            params.dts
        );
    } else {
        trajectories = Swing::solve_rk4(
            params.weighted_edge_list,
            {params.phase, params.dphase},
            {params.power, params.gamma, params.mass},
            params.dts
        );
    }

    //* Report result with maximum precision
    std::cout << std::setprecision(std::numeric_limits<T>::digits10 + 1);
    for (const auto& trajectory : trajectories) {
        for (const auto& traj : trajectory) {
            std::cout << traj << " ";
        }
    }
}

}

int main(int argc, char* argv[]) {
    //* Get input variables
    const std::string solver_name = argv[1];
    const uint64_t num_nodes = std::stoull(argv[2]);
    const uint64_t num_edges = std::stoull(argv[3]);
    const uint64_t num_steps = std::stoull(argv[4]);
    const uint64_t precision = std::stoull(argv[5]);
    const std::string arg_file_name = argv[6];

    if (precision == 32) {
        Swing::solve(solver_name, num_nodes, num_edges, num_steps, arg_file_name, (float)0.0);
    } else {
        Swing::solve(solver_name, num_nodes, num_edges, num_steps, arg_file_name, (double)0.0);
    }

    return 0;
}