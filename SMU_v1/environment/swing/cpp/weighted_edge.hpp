#pragma once

namespace Swing {

template <typename T>
struct WeightedEdge {
    uint64_t node1;
    uint64_t node2;
    T weight;

    WeightedEdge();
    WeightedEdge(const uint64_t& t_node1, const uint64_t& t_node2, const T& t_weight)
        : node1(t_node1), node2(t_node2), weight(t_weight) {}
};

}  // namespace Swing