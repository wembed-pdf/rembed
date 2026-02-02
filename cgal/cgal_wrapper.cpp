// cgal_wrapper.cpp
// C++ wrapper for CGAL Kd-tree providing a C-compatible interface for Rust FFI.
// Uses CGAL's Epick_d kernel (Exact Predicates Inexact Constructions in d-dimensions)
// with Eigen3 for efficient high-dimensional spatial indexing.

#include "cgal_wrapper.h"
#include <CGAL/Epick_d.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>

// Template implementation for different dimensions
template<int DIM>
class CgalKdTreeImpl {
public:
    typedef CGAL::Epick_d<CGAL::Dimension_tag<DIM>> Kernel;
    typedef CGAL::Search_traits_d<Kernel> Traits;
    typedef typename Kernel::Point_d Point_d;
    typedef CGAL::Kd_tree<Traits> Tree;

    // Hash and equality for Point_d to enable O(1) index lookup
    struct PointHash {
        size_t operator()(const Point_d& p) const {
            size_t h = 0;
            for (int d = 0; d < DIM; ++d) {
                h ^= std::hash<double>{}(p[d]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    struct PointEqual {
        bool operator()(const Point_d& a, const Point_d& b) const {
            for (int d = 0; d < DIM; ++d) {
                if (a[d] != b[d]) return false;
            }
            return true;
        }
    };

    CgalKdTreeImpl(const float* points, size_t num_points, size_t dimensions)
        : num_points_(num_points), dimensions_(dimensions) {
        rebuild(points, num_points);
    }

    void rebuild(const float* points, size_t num_points) {
        num_points_ = num_points;
        points_.clear();
        points_.reserve(num_points);
        point_to_index_.clear();
        point_to_index_.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            std::vector<double> coords(DIM);
            for (int d = 0; d < DIM; ++d) {
                coords[d] = static_cast<double>(points[i * DIM + d]);
            }
            Point_d pt(DIM, coords.begin(), coords.end());
            points_.push_back(pt);
            point_to_index_[pt] = i;
        }

        // Build the Kd-tree from the stored points
        tree_ = std::make_unique<Tree>(points_.begin(), points_.end());
        tree_->build();
    }

    size_t radius_search(
        const float* query_point,
        float radius_squared,
        float epsilon,
        size_t* out_indices,
        float* out_distances_squared,
        size_t max_results
    ) {
        // Convert query point to CGAL point
        std::vector<double> query_coords(DIM);
        for (int d = 0; d < DIM; ++d) {
            query_coords[d] = static_cast<double>(query_point[d]);
        }
        Point_d query_pt(DIM, query_coords.begin(), query_coords.end());

        float radius = std::sqrt(radius_squared);

        // Create fuzzy sphere for radius search (epsilon=0 for exact)
        CGAL::Fuzzy_sphere<Traits> fuzzy_sphere(query_pt, radius, epsilon);

        std::vector<Point_d> results;
        tree_->search(std::back_inserter(results), fuzzy_sphere);

        // Map results back to indices using hash map
        size_t count = 0;
        for (const auto& result_pt : results) {
            if (count >= max_results) break;

            auto it = point_to_index_.find(result_pt);
            if (it == point_to_index_.end()) continue;

            // Calculate exact squared distance in float precision
            float dist_sq = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                float diff = static_cast<float>(result_pt[d] - query_pt[d]);
                dist_sq += diff * diff;
            }

            // Verify within radius (fuzzy sphere may include points slightly outside)
            if (dist_sq <= radius_squared) {
                out_indices[count] = it->second;
                out_distances_squared[count] = dist_sq;
                ++count;
            }
        }

        return count;
    }

    size_t point_count() const { return num_points_; }
    size_t dimensions() const { return dimensions_; }

private:
    std::unique_ptr<Tree> tree_;
    std::vector<Point_d> points_;
    std::unordered_map<Point_d, size_t, PointHash, PointEqual> point_to_index_;
    size_t num_points_;
    size_t dimensions_;
};

// Opaque struct holding the implementation
struct CgalKdTreeIndex {
    void* impl;
    size_t dimensions;
};

// Helper function to create implementation based on dimension
template<int DIM>
CgalKdTreeIndex* create_impl(const float* points, size_t num_points, size_t dimensions) {
    auto* index = new CgalKdTreeIndex;
    index->impl = new CgalKdTreeImpl<DIM>(points, num_points, dimensions);
    index->dimensions = dimensions;
    return index;
}

// C API implementations
extern "C" {

CgalKdTreeIndex* cgal_kdtree_create_index(
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

void cgal_kdtree_destroy_index(CgalKdTreeIndex* index) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:
            delete static_cast<CgalKdTreeImpl<2>*>(index->impl);
            break;
        case 3:
            delete static_cast<CgalKdTreeImpl<3>*>(index->impl);
            break;
        case 4:
            delete static_cast<CgalKdTreeImpl<4>*>(index->impl);
            break;
        case 8:
            delete static_cast<CgalKdTreeImpl<8>*>(index->impl);
            break;
    }

    delete index;
}

size_t cgal_kdtree_radius_search(
    const CgalKdTreeIndex* index,
    const float* query_point,
    float radius_squared,
    float epsilon,
    size_t* out_indices,
    float* out_distances_squared,
    size_t max_results
) {
    if (!index) return 0;

    switch (index->dimensions) {
        case 2:
            return static_cast<CgalKdTreeImpl<2>*>(index->impl)->radius_search(
                query_point, radius_squared, epsilon, out_indices, out_distances_squared, max_results
            );
        case 3:
            return static_cast<CgalKdTreeImpl<3>*>(index->impl)->radius_search(
                query_point, radius_squared, epsilon, out_indices, out_distances_squared, max_results
            );
        case 4:
            return static_cast<CgalKdTreeImpl<4>*>(index->impl)->radius_search(
                query_point, radius_squared, epsilon, out_indices, out_distances_squared, max_results
            );
        case 8:
            return static_cast<CgalKdTreeImpl<8>*>(index->impl)->radius_search(
                query_point, radius_squared, epsilon, out_indices, out_distances_squared, max_results
            );
        default:
            return 0;
    }
}

void cgal_kdtree_update_points(
    CgalKdTreeIndex* index,
    const float* points,
    size_t num_points
) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:
            static_cast<CgalKdTreeImpl<2>*>(index->impl)->rebuild(points, num_points);
            break;
        case 3:
            static_cast<CgalKdTreeImpl<3>*>(index->impl)->rebuild(points, num_points);
            break;
        case 4:
            static_cast<CgalKdTreeImpl<4>*>(index->impl)->rebuild(points, num_points);
            break;
        case 8:
            static_cast<CgalKdTreeImpl<8>*>(index->impl)->rebuild(points, num_points);
            break;
    }
}

size_t cgal_kdtree_point_count(const CgalKdTreeIndex* index) {
    if (!index) return 0;

    switch (index->dimensions) {
        case 2:
            return static_cast<CgalKdTreeImpl<2>*>(index->impl)->point_count();
        case 3:
            return static_cast<CgalKdTreeImpl<3>*>(index->impl)->point_count();
        case 4:
            return static_cast<CgalKdTreeImpl<4>*>(index->impl)->point_count();
        case 8:
            return static_cast<CgalKdTreeImpl<8>*>(index->impl)->point_count();
        default:
            return 0;
    }
}

size_t cgal_kdtree_dimensions(const CgalKdTreeIndex* index) {
    return index ? index->dimensions : 0;
}

}  // extern "C"
