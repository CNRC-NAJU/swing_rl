/*
Solve swing equation

m_i * d^2 theta_i / dt^2 = P_i - gamma_i * d theta_i/dt + sum_j K_ij * A_ij *
sin(theta_j-theta_i)
*/

#pragma once

#include <cmath>
#include <vector>

#include "linear_algebra.hpp"
#include "weighted_edge.hpp"

template <typename T>
using mat = std::vector<std::vector<T>>;

namespace Swing {

template <typename T>
std::vector<T> get_acceleration(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_state,
    const std::vector<std::vector<T>>& t_params
) {
    /*
    t_weighted_edge_list: (E,) If (0, 1) is at edge_list, (1, 0) is not
    t_state: (2, N), phase, dphase of each node
    t_params: (3, N), node features of power, gamma, mass
    */

    using namespace LinearAlgebra;

    const uint64_t num_nodes = t_state[0].size();
    std::vector<T> force(num_nodes, 0.0);

    // P - gamma * velocity
    for (uint64_t node = 0; node < num_nodes; ++node) {
        force[node] = t_params[0][node] - t_params[1][node] * t_state[1][node];
    }

    for (const WeightedEdge<T>& weighted_edge : t_weighted_edge_list) {
        const uint64_t node1 = weighted_edge.node1;
        const uint64_t node2 = weighted_edge.node2;
        const T weight = weighted_edge.weight;

        // Calculate interaction: KA sin(theta_j - theta_i)
        const T interaction = weight * std::sin(t_state[0][node2] - t_state[0][node1]);

        // Use symmetric property
        force[node1] += interaction;
        force[node2] -= interaction;
    }

    // a = F / m
    for (uint64_t node = 0; node < num_nodes; ++node) {
        force[node] /= t_params[2][node];
    }

    return force;
}

template <typename T>
std::vector<std::vector<T>> step_rk1(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_state,
    const std::vector<std::vector<T>>& t_params,
    const T& dt
) {
    using namespace LinearAlgebra;

    const std::vector<T> velocity = t_state[1];
    const std::vector<T> acceleration =
        get_acceleration(t_weighted_edge_list, t_state, t_params);

    // Result
    return {t_state[0] + dt * velocity, t_state[1] + dt * acceleration};
}

template <typename T>
std::vector<std::vector<T>> step_rk2(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_state,
    const std::vector<std::vector<T>>& t_params,
    const T& dt
) {
    using namespace LinearAlgebra;

    const uint64_t num_nodes = t_weighted_edge_list.size();
    std::vector<std::vector<T>> temp_state = {
        std::vector<T>(num_nodes, 0.0), std::vector<T>(num_nodes, 0.0)};

    // Stage 1
    const std::vector<T> velocity1 = t_state[1];
    const std::vector<T> acceleration1 =
        get_acceleration(t_weighted_edge_list, t_state, t_params);

    // Stage 2
    temp_state[0] = t_state[0] + dt * velocity1;
    temp_state[1] = t_state[1] + dt * acceleration1;
    const std::vector<T> velocity2 = temp_state[1];
    const std::vector<T> acceleration2 =
        get_acceleration(t_weighted_edge_list, temp_state, t_params);

    // Result
    const std::vector<T> velocity = 0.5 * (velocity1 + velocity2);
    const std::vector<T> acceleration = 0.5 * (acceleration1 + acceleration2);
    return {t_state[0] + dt * velocity, t_state[1] + dt * acceleration};
}

template <typename T>
std::vector<std::vector<T>> step_rk4(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_state,
    const std::vector<std::vector<T>>& t_params,
    const T& dt
) {
    using namespace LinearAlgebra;

    const uint64_t num_nodes = t_weighted_edge_list.size();
    std::vector<std::vector<T>> temp_state = {
        std::vector<T>(num_nodes, 0.0), std::vector<T>(num_nodes, 0.0)};

    // Stage 1
    const std::vector<T> velocity1 = t_state[1];
    const std::vector<T> acceleration1 =
        get_acceleration(t_weighted_edge_list, t_state, t_params);

    // Stage 2
    temp_state[0] = t_state[0] + 0.5 * dt * velocity1;
    temp_state[1] = t_state[1] + 0.5 * dt * acceleration1;
    const std::vector<T> velocity2 = temp_state[1];
    const std::vector<T> acceleration2 =
        get_acceleration(t_weighted_edge_list, temp_state, t_params);

    // Stage 3
    temp_state[0] = t_state[0] + 0.5 * dt * velocity2;
    temp_state[1] = t_state[1] + 0.5 * dt * acceleration2;
    const std::vector<T> velocity3 = temp_state[1];
    const std::vector<T> acceleration3 =
        get_acceleration(t_weighted_edge_list, temp_state, t_params);

    // Stage 4
    temp_state[0] = t_state[0] + dt * velocity3;
    temp_state[1] = t_state[1] + dt * acceleration3;
    const std::vector<T> velocity4 = temp_state[1];
    const std::vector<T> acceleration4 =
        get_acceleration(t_weighted_edge_list, temp_state, t_params);

    // Result
    const std::vector<T> velocity =
        (velocity1 + 2.0 * velocity2 + 2.0 * velocity3 + velocity4) / 6.0;
    const std::vector<T> acceleration =
        (acceleration1 + 2.0 * acceleration2 + 2.0 * acceleration3 + acceleration4) /
        6.0;
    return {t_state[0] + dt * velocity, t_state[1] + dt * acceleration};
}

template <typename T>
std::vector<std::vector<T>> solve_rk1(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_initial_state,
    const std::vector<std::vector<T>>& t_params,
    const std::vector<T>& t_dts
) {
    /*
    t_weighted_edge_list: (N, N), adjacency matrix wegithed by coupling constant
    t_state: (2, N), phase, dphase of each node
    t_params: (3, N), node features of power, gamma, mass
    t_dts: (S, ), dt for each time step

    Return
    (S+1, 2 * N), phases, dphases of each node at each time step
    */

    std::vector<std::vector<T>> trajectory;  // (S+1, 2*N)
    trajectory.reserve(t_dts.size() + 1);
    trajectory.emplace_back(LinearAlgebra::flatten(t_initial_state));

    std::vector<std::vector<T>> state = t_initial_state;
    for (const auto& dt : t_dts) {
        state = step_rk1(t_weighted_edge_list, state, t_params, dt);
        trajectory.emplace_back(LinearAlgebra::flatten(state));
    }

    return trajectory;
}

template <typename T>
std::vector<std::vector<T>> solve_rk2(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_initial_state,
    const std::vector<std::vector<T>>& t_params,
    const std::vector<T>& t_dts
) {
    /*
    t_weighted_edge_list: (N, N), adjacency matrix wegithed by coupling constant
    t_state: (2, N), phase, dphase of each node
    t_params: (3, N), node features of power, gamma, mass
    t_dts: (S, ), dt for each time step

    Return
    (S+1, 2 * N), phases, dphases of each node at each time step
    */

    std::vector<std::vector<T>> trajectory;  // (S+1, 2*N)
    trajectory.reserve(t_dts.size() + 1);
    trajectory.emplace_back(LinearAlgebra::flatten(t_initial_state));

    std::vector<std::vector<T>> state = t_initial_state;
    for (const auto& dt : t_dts) {
        state = step_rk2(t_weighted_edge_list, state, t_params, dt);
        trajectory.emplace_back(LinearAlgebra::flatten(state));
    }

    return trajectory;
}

template <typename T>
std::vector<std::vector<T>> solve_rk4(
    const std::vector<WeightedEdge<T>>& t_weighted_edge_list,
    const std::vector<std::vector<T>>& t_initial_state,
    const std::vector<std::vector<T>>& t_params,
    const std::vector<T>& t_dts
) {
    /*
    t_weighted_edge_list: (E,) If (0, 1) is at edge_list, (1, 0) is not
    t_state: (2, N), phase, dphase of each node
    t_params: (3, N), node features of power, gamma, mass
    t_dts: (S, ), dt for each time step

    Return
    (S+1, 2 * N), phase1, ... phaseN, dphase1,...,dphaseN at each time step
    */

    std::vector<std::vector<T>> trajectory;  // (S+1, 2*N)
    trajectory.reserve(t_dts.size() + 1);
    trajectory.emplace_back(LinearAlgebra::flatten(t_initial_state));

    std::vector<std::vector<T>> state = t_initial_state;
    for (const auto& dt : t_dts) {
        state = step_rk4(t_weighted_edge_list, state, t_params, dt);
        trajectory.emplace_back(LinearAlgebra::flatten(state));
    }

    return trajectory;
}

}  // namespace Swing