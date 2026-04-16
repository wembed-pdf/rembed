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

    CgalKdTreeResult radius_search(
        const float* query_point,
        float radius_squared,
        float epsilon
    ) {
        CgalKdTreeResult result = {nullptr, nullptr, 0};

        // Convert query point to CGAL point
        std::vector<double> query_coords(DIM);
        for (int d = 0; d < DIM; ++d) {
            query_coords[d] = static_cast<double>(query_point[d]);
        }
        Point_d query_pt(DIM, query_coords.begin(), query_coords.end());

        float radius = std::sqrt(radius_squared);

        // Create fuzzy sphere for radius search (epsilon=0 for exact)
        CGAL::Fuzzy_sphere<Traits> fuzzy_sphere(query_pt, radius, epsilon);

        std::vector<Point_d> search_results;
        tree_->search(std::back_inserter(search_results), fuzzy_sphere);

        // Collect valid results
        std::vector<size_t> indices;
        std::vector<float> distances;

        for (const auto& result_pt : search_results) {
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
                indices.push_back(it->second);
                distances.push_back(dist_sq);
            }
        }

        // Allocate and copy results
        result.count = indices.size();
        if (result.count > 0) {
            result.indices = new size_t[result.count];
            result.distances_squared = new float[result.count];
            for (size_t i = 0; i < result.count; ++i) {
                result.indices[i] = indices[i];
                result.distances_squared[i] = distances[i];
            }
        }

        return result;
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
        case 2:  return create_impl<2>(points, num_points, dimensions);
        case 3:  return create_impl<3>(points, num_points, dimensions);
        case 4:  return create_impl<4>(points, num_points, dimensions);
        case 5:  return create_impl<5>(points, num_points, dimensions);
        case 6:  return create_impl<6>(points, num_points, dimensions);
        case 7:  return create_impl<7>(points, num_points, dimensions);
        case 8:  return create_impl<8>(points, num_points, dimensions);
        case 9:  return create_impl<9>(points, num_points, dimensions);
        case 10: return create_impl<10>(points, num_points, dimensions);
        case 11: return create_impl<11>(points, num_points, dimensions);
        case 12: return create_impl<12>(points, num_points, dimensions);
        case 13: return create_impl<13>(points, num_points, dimensions);
        case 14: return create_impl<14>(points, num_points, dimensions);
        case 15: return create_impl<15>(points, num_points, dimensions);
        case 16: return create_impl<16>(points, num_points, dimensions);
        default:
            // For unsupported dimensions, return nullptr
            return nullptr;
    }
}

void cgal_kdtree_destroy_index(CgalKdTreeIndex* index) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:  delete static_cast<CgalKdTreeImpl<2>*>(index->impl);  break;
        case 3:  delete static_cast<CgalKdTreeImpl<3>*>(index->impl);  break;
        case 4:  delete static_cast<CgalKdTreeImpl<4>*>(index->impl);  break;
        case 5:  delete static_cast<CgalKdTreeImpl<5>*>(index->impl);  break;
        case 6:  delete static_cast<CgalKdTreeImpl<6>*>(index->impl);  break;
        case 7:  delete static_cast<CgalKdTreeImpl<7>*>(index->impl);  break;
        case 8:  delete static_cast<CgalKdTreeImpl<8>*>(index->impl);  break;
        case 9:  delete static_cast<CgalKdTreeImpl<9>*>(index->impl);  break;
        case 10: delete static_cast<CgalKdTreeImpl<10>*>(index->impl); break;
        case 11: delete static_cast<CgalKdTreeImpl<11>*>(index->impl); break;
        case 12: delete static_cast<CgalKdTreeImpl<12>*>(index->impl); break;
        case 13: delete static_cast<CgalKdTreeImpl<13>*>(index->impl); break;
        case 14: delete static_cast<CgalKdTreeImpl<14>*>(index->impl); break;
        case 15: delete static_cast<CgalKdTreeImpl<15>*>(index->impl); break;
        case 16: delete static_cast<CgalKdTreeImpl<16>*>(index->impl); break;
    }

    delete index;
}

CgalKdTreeResult cgal_kdtree_radius_search(
    const CgalKdTreeIndex* index,
    const float* query_point,
    float radius_squared,
    float epsilon
) {
    CgalKdTreeResult result = {nullptr, nullptr, 0};
    if (!index) return result;

    switch (index->dimensions) {
        case 2:  return static_cast<CgalKdTreeImpl<2>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 3:  return static_cast<CgalKdTreeImpl<3>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 4:  return static_cast<CgalKdTreeImpl<4>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 5:  return static_cast<CgalKdTreeImpl<5>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 6:  return static_cast<CgalKdTreeImpl<6>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 7:  return static_cast<CgalKdTreeImpl<7>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 8:  return static_cast<CgalKdTreeImpl<8>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 9:  return static_cast<CgalKdTreeImpl<9>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 10: return static_cast<CgalKdTreeImpl<10>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 11: return static_cast<CgalKdTreeImpl<11>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 12: return static_cast<CgalKdTreeImpl<12>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 13: return static_cast<CgalKdTreeImpl<13>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 14: return static_cast<CgalKdTreeImpl<14>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 15: return static_cast<CgalKdTreeImpl<15>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        case 16: return static_cast<CgalKdTreeImpl<16>*>(index->impl)->radius_search(query_point, radius_squared, epsilon);
        default: return result;
    }
}

void cgal_kdtree_free_result(CgalKdTreeResult* result) {
    if (result) {
        delete[] result->indices;
        delete[] result->distances_squared;
        result->indices = nullptr;
        result->distances_squared = nullptr;
        result->count = 0;
    }
}

void cgal_kdtree_update_points(
    CgalKdTreeIndex* index,
    const float* points,
    size_t num_points
) {
    if (!index) return;

    switch (index->dimensions) {
        case 2:  static_cast<CgalKdTreeImpl<2>*>(index->impl)->rebuild(points, num_points);  break;
        case 3:  static_cast<CgalKdTreeImpl<3>*>(index->impl)->rebuild(points, num_points);  break;
        case 4:  static_cast<CgalKdTreeImpl<4>*>(index->impl)->rebuild(points, num_points);  break;
        case 5:  static_cast<CgalKdTreeImpl<5>*>(index->impl)->rebuild(points, num_points);  break;
        case 6:  static_cast<CgalKdTreeImpl<6>*>(index->impl)->rebuild(points, num_points);  break;
        case 7:  static_cast<CgalKdTreeImpl<7>*>(index->impl)->rebuild(points, num_points);  break;
        case 8:  static_cast<CgalKdTreeImpl<8>*>(index->impl)->rebuild(points, num_points);  break;
        case 9:  static_cast<CgalKdTreeImpl<9>*>(index->impl)->rebuild(points, num_points);  break;
        case 10: static_cast<CgalKdTreeImpl<10>*>(index->impl)->rebuild(points, num_points); break;
        case 11: static_cast<CgalKdTreeImpl<11>*>(index->impl)->rebuild(points, num_points); break;
        case 12: static_cast<CgalKdTreeImpl<12>*>(index->impl)->rebuild(points, num_points); break;
        case 13: static_cast<CgalKdTreeImpl<13>*>(index->impl)->rebuild(points, num_points); break;
        case 14: static_cast<CgalKdTreeImpl<14>*>(index->impl)->rebuild(points, num_points); break;
        case 15: static_cast<CgalKdTreeImpl<15>*>(index->impl)->rebuild(points, num_points); break;
        case 16: static_cast<CgalKdTreeImpl<16>*>(index->impl)->rebuild(points, num_points); break;
    }
}

size_t cgal_kdtree_point_count(const CgalKdTreeIndex* index) {
    if (!index) return 0;

    switch (index->dimensions) {
        case 2:  return static_cast<CgalKdTreeImpl<2>*>(index->impl)->point_count();
        case 3:  return static_cast<CgalKdTreeImpl<3>*>(index->impl)->point_count();
        case 4:  return static_cast<CgalKdTreeImpl<4>*>(index->impl)->point_count();
        case 5:  return static_cast<CgalKdTreeImpl<5>*>(index->impl)->point_count();
        case 6:  return static_cast<CgalKdTreeImpl<6>*>(index->impl)->point_count();
        case 7:  return static_cast<CgalKdTreeImpl<7>*>(index->impl)->point_count();
        case 8:  return static_cast<CgalKdTreeImpl<8>*>(index->impl)->point_count();
        case 9:  return static_cast<CgalKdTreeImpl<9>*>(index->impl)->point_count();
        case 10: return static_cast<CgalKdTreeImpl<10>*>(index->impl)->point_count();
        case 11: return static_cast<CgalKdTreeImpl<11>*>(index->impl)->point_count();
        case 12: return static_cast<CgalKdTreeImpl<12>*>(index->impl)->point_count();
        case 13: return static_cast<CgalKdTreeImpl<13>*>(index->impl)->point_count();
        case 14: return static_cast<CgalKdTreeImpl<14>*>(index->impl)->point_count();
        case 15: return static_cast<CgalKdTreeImpl<15>*>(index->impl)->point_count();
        case 16: return static_cast<CgalKdTreeImpl<16>*>(index->impl)->point_count();
        default: return 0;
    }
}

size_t cgal_kdtree_dimensions(const CgalKdTreeIndex* index) {
    return index ? index->dimensions : 0;
}

}  // extern "C"
