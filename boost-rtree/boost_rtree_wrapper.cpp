// boost_rtree_wrapper.cpp
// C++ wrapper for Boost.Geometry R-tree providing a C-compatible interface for Rust FFI.
// Uses Boost's highly-optimized R-tree implementation for spatial indexing.

#include "boost_rtree_wrapper.h"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// Helper to set coordinates at compile-time using index sequence
template<size_t DIM, size_t... Is>
void set_coords_impl(bg::model::point<float, DIM, bg::cs::cartesian>& pt,
                     const float* coords, std::index_sequence<Is...>) {
    ((bg::set<Is>(pt, coords[Is])), ...);
}

template<size_t DIM>
void set_coords(bg::model::point<float, DIM, bg::cs::cartesian>& pt, const float* coords) {
    set_coords_impl(pt, coords, std::make_index_sequence<DIM>{});
}

// Helper to calculate squared distance between point and coordinate array
template<size_t DIM, size_t... Is>
float distance_squared_impl(const bg::model::point<float, DIM, bg::cs::cartesian>& pt,
                           const float* coords, std::index_sequence<Is...>) {
    return (((bg::get<Is>(pt) - coords[Is]) * (bg::get<Is>(pt) - coords[Is])) + ...);
}

template<size_t DIM>
float distance_squared(const bg::model::point<float, DIM, bg::cs::cartesian>& pt, const float* coords) {
    return distance_squared_impl(pt, coords, std::make_index_sequence<DIM>{});
}

// Template implementation for different dimensions
template<size_t DIM>
class BoostRTreeImpl {
public:
    using Point = bg::model::point<float, DIM, bg::cs::cartesian>;
    using Value = std::pair<Point, size_t>;  // (point, index)
    using RTree = bgi::rtree<Value, bgi::quadratic<16>>;

    BoostRTreeImpl(const float* points, size_t num_points, size_t dimensions)
        : num_points_(num_points), dimensions_(dimensions) {
        rebuild(points, num_points);
    }

    void rebuild(const float* points, size_t num_points) {
        num_points_ = num_points;
        std::vector<Value> values;
        values.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            Point pt;
            set_coords<DIM>(pt, &points[i * DIM]);
            values.emplace_back(pt, i);
        }

        tree_ = std::make_unique<RTree>(values.begin(), values.end());
    }

    size_t radius_search(
        const float* query_point,
        float radius_squared,
        size_t* out_indices,
        float* out_distances_squared,
        size_t max_results
    ) {
        Point query_pt;
        set_coords<DIM>(query_pt, query_point);

        float radius = std::sqrt(radius_squared);

        // Create bounding box around query point
        Point min_corner, max_corner;
        float min_coords[DIM], max_coords[DIM];
        for (size_t d = 0; d < DIM; ++d) {
            min_coords[d] = query_point[d] - radius;
            max_coords[d] = query_point[d] + radius;
        }
        set_coords<DIM>(min_corner, min_coords);
        set_coords<DIM>(max_corner, max_coords);
        bg::model::box<Point> search_box(min_corner, max_corner);

        // Query with bounding box and distance predicate
        std::vector<Value> results;
        tree_->query(
            bgi::within(search_box) &&
            bgi::satisfies([&](const Value& v) {
                return distance_squared<DIM>(v.first, query_point) <= radius_squared;
            }),
            std::back_inserter(results)
        );

        // Limit results and copy to output arrays
        size_t count = std::min(results.size(), max_results);
        for (size_t i = 0; i < count; ++i) {
            out_indices[i] = results[i].second;
            out_distances_squared[i] = distance_squared<DIM>(results[i].first, query_point);
        }

        return count;
    }

    size_t point_count() const { return num_points_; }
    size_t dimensions() const { return dimensions_; }

private:
    std::unique_ptr<RTree> tree_;
    size_t num_points_;
    size_t dimensions_;
};

// Opaque struct holding the implementation
struct BoostRTreeIndex {
    void* impl;
    size_t dimensions;
};

// Helper function to create implementation based on dimension
template<size_t DIM>
BoostRTreeIndex* create_impl(const float* points, size_t num_points, size_t dimensions) {
    auto* index = new BoostRTreeIndex;
    index->impl = new BoostRTreeImpl<DIM>(points, num_points, dimensions);
    index->dimensions = dimensions;
    return index;
}

// C API implementations
extern "C" {

BoostRTreeIndex* boost_rtree_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions
) {
    // Runtime dispatch based on dimensions
    switch (dimensions) {
        case 2: return create_impl<2>(points, num_points, dimensions);
        case 3: return create_impl<3>(points, num_points, dimensions);
        case 4: return create_impl<4>(points, num_points, dimensions);
        case 8: return create_impl<8>(points, num_points, dimensions);
        default:
            // For unsupported dimensions, return nullptr
            return nullptr;
    }
}

void boost_rtree_destroy_index(BoostRTreeIndex* index) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:
            delete static_cast<BoostRTreeImpl<2>*>(index->impl);
            break;
        case 3:
            delete static_cast<BoostRTreeImpl<3>*>(index->impl);
            break;
        case 4:
            delete static_cast<BoostRTreeImpl<4>*>(index->impl);
            break;
        case 8:
            delete static_cast<BoostRTreeImpl<8>*>(index->impl);
            break;
    }

    delete index;
}

size_t boost_rtree_radius_search(
    const BoostRTreeIndex* index,
    const float* query_point,
    float radius_squared,
    size_t* out_indices,
    float* out_distances_squared,
    size_t max_results
) {
    if (!index) return 0;

    switch (index->dimensions) {
        case 2:
            return static_cast<BoostRTreeImpl<2>*>(index->impl)->radius_search(
                query_point, radius_squared, out_indices, out_distances_squared, max_results
            );
        case 3:
            return static_cast<BoostRTreeImpl<3>*>(index->impl)->radius_search(
                query_point, radius_squared, out_indices, out_distances_squared, max_results
            );
        case 4:
            return static_cast<BoostRTreeImpl<4>*>(index->impl)->radius_search(
                query_point, radius_squared, out_indices, out_distances_squared, max_results
            );
        case 8:
            return static_cast<BoostRTreeImpl<8>*>(index->impl)->radius_search(
                query_point, radius_squared, out_indices, out_distances_squared, max_results
            );
        default:
            return 0;
    }
}

void boost_rtree_update_points(
    BoostRTreeIndex* index,
    const float* points,
    size_t num_points
) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:
            static_cast<BoostRTreeImpl<2>*>(index->impl)->rebuild(points, num_points);
            break;
        case 3:
            static_cast<BoostRTreeImpl<3>*>(index->impl)->rebuild(points, num_points);
            break;
        case 4:
            static_cast<BoostRTreeImpl<4>*>(index->impl)->rebuild(points, num_points);
            break;
        case 8:
            static_cast<BoostRTreeImpl<8>*>(index->impl)->rebuild(points, num_points);
            break;
    }
}

size_t boost_rtree_point_count(const BoostRTreeIndex* index) {
    if (!index) return 0;

    switch (index->dimensions) {
        case 2:
            return static_cast<BoostRTreeImpl<2>*>(index->impl)->point_count();
        case 3:
            return static_cast<BoostRTreeImpl<3>*>(index->impl)->point_count();
        case 4:
            return static_cast<BoostRTreeImpl<4>*>(index->impl)->point_count();
        case 8:
            return static_cast<BoostRTreeImpl<8>*>(index->impl)->point_count();
        default:
            return 0;
    }
}

size_t boost_rtree_dimensions(const BoostRTreeIndex* index) {
    return index ? index->dimensions : 0;
}

}  // extern "C"
