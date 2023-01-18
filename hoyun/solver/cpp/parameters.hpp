#pragma once
#include <vector>

#include "weighted_edge.hpp"

namespace Swing {

template <typename T>
struct Parameters {
    //* Parameters: node properties
    std::vector<T> phase, dphase, power, gamma, mass;

    //* Parameters: network properties
    std::vector<WeightedEdge<T>> weighted_edge_list;

    //* Parameters: times
    std::vector<T> dts;

    Parameters();
    Parameters(
        const std::vector<T> t_args,
        const uint64_t& t_num_nodes,
        const uint64_t& t_num_edges,
        const uint64_t& t_num_steps
    ) {
        //* Parameter locations at argument file
        const uint64_t phase_start_idx = 0;
        const uint64_t dphase_start_idx = phase_start_idx + t_num_nodes;
        const uint64_t power_start_idx = dphase_start_idx + t_num_nodes;
        const uint64_t gamma_start_idx = power_start_idx + t_num_nodes;
        const uint64_t mass_start_idx = gamma_start_idx + t_num_nodes;
        const uint64_t wedge_list_start_idx = mass_start_idx + t_num_nodes;
        const uint64_t dt_start_idx = wedge_list_start_idx + 3 * t_num_edges;

        //* Parameters: node properties
        phase.assign(t_num_nodes, 0.0);
        dphase.assign(t_num_nodes, 0.0);
        power.assign(t_num_nodes, 0.0);
        gamma.assign(t_num_nodes, 0.0);
        mass.assign(t_num_nodes, 0.0);
        for (uint64_t i = 0; i < t_num_nodes; ++i) {
            phase[i] = t_args[phase_start_idx + i];
            dphase[i] = t_args[dphase_start_idx + i];
            power[i] = t_args[power_start_idx + i];
            gamma[i] = t_args[gamma_start_idx + i];
            mass[i] = t_args[mass_start_idx + i];
        }

        //* Parameters: network properties
        weighted_edge_list.reserve(t_num_edges);
        for (uint64_t e = 0; e < t_num_edges; ++e) {
            const uint64_t node1 = (uint64_t)t_args[wedge_list_start_idx + 3 * e];
            const uint64_t node2 = (uint64_t)t_args[wedge_list_start_idx + 3 * e + 1];
            const T weight = t_args[wedge_list_start_idx + 3 * e + 2];
            weighted_edge_list.emplace_back(WeightedEdge<T>(node1, node2, weight));
        }

        //* Parameters: times
        dts.assign(t_num_steps, 0.0);
        for (uint64_t t = 0; t < t_num_steps; ++t) {
            dts[t] = t_args[dt_start_idx + t];
        }
    }
};

}  // namespace Swing
