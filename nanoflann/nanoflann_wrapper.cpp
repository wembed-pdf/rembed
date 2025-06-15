// nanoflann_wrapper.cpp
#include "nanoflann_wrapper.h"
#include "nanoflann.hpp"
#include <vector>
#include <memory>
#include <cstring>

// Point cloud adaptor for interfacing with Rust
template<typename T, int DIM>
struct RustPointCloud {
    std::vector<std::vector<T>> points;
    
    inline size_t kdtree_get_point_count() const { 
        return points.size(); 
    }
    
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    }
    
    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { 
        return false; 
    }
};

// Internal wrapper struct that holds the KD-tree index
// Use size_t as IndexType to match our C interface
template<int DIM>
struct NanoflannWrapperImpl {
    using PointCloud = RustPointCloud<float, DIM>;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud, float, size_t>,
        PointCloud,
        DIM,
        size_t  // Use size_t for IndexType to match our interface
    >;
    
    std::unique_ptr<PointCloud> cloud;
    std::unique_ptr<KDTree> index;
    size_t dimensions;
    size_t leaf_max_size;
    
    NanoflannWrapperImpl(const float* points, size_t num_points, size_t dims, size_t leaf_size) 
        : dimensions(dims), leaf_max_size(leaf_size)
    {
        cloud = std::make_unique<PointCloud>();
        cloud->points.resize(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].resize(dims);
            for (size_t d = 0; d < dims; ++d) {
                cloud->points[i][d] = points[i * dims + d];
            }
        }
        
        index = std::make_unique<KDTree>(
            static_cast<int>(dims), *cloud, 
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size)
        );
    }
    
    void rebuild(const float* points, size_t num_points) {
        cloud->points.resize(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].resize(dimensions);
            for (size_t d = 0; d < dimensions; ++d) {
                cloud->points[i][d] = points[i * dimensions + d];
            }
        }
        
        // Rebuild the index
        index = std::make_unique<KDTree>(
            static_cast<int>(dimensions), *cloud, 
            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size)
        );
    }
};

// Opaque struct exposed to C
struct NanoflannIndex {
    void* impl;
    int dimensions;
    
    NanoflannIndex(const float* points, size_t num_points, size_t dims, size_t leaf_max_size)
        : dimensions(static_cast<int>(dims))
    {
        switch (dims) {
            case 2:
                impl = new NanoflannWrapperImpl<2>(points, num_points, dims, leaf_max_size);
                break;
            case 3:
                impl = new NanoflannWrapperImpl<3>(points, num_points, dims, leaf_max_size);
                break;
            case 4:
                impl = new NanoflannWrapperImpl<4>(points, num_points, dims, leaf_max_size);
                break;
            case 8:
                impl = new NanoflannWrapperImpl<8>(points, num_points, dims, leaf_max_size);
                break;
            default:
                // For dynamic dimensions, use -1 template parameter
                impl = new NanoflannWrapperImpl<-1>(points, num_points, dims, leaf_max_size);
                break;
        }
    }
    
    ~NanoflannIndex() {
        switch (dimensions) {
            case 2: delete static_cast<NanoflannWrapperImpl<2>*>(impl); break;
            case 3: delete static_cast<NanoflannWrapperImpl<3>*>(impl); break;
            case 4: delete static_cast<NanoflannWrapperImpl<4>*>(impl); break;
            case 8: delete static_cast<NanoflannWrapperImpl<8>*>(impl); break;
            default: delete static_cast<NanoflannWrapperImpl<-1>*>(impl); break;
        }
    }
    
    template<int DIM>
    size_t knn_search_impl(const float* query_point, size_t k, size_t* indices, float* distances) const {
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);
        return w->index->knnSearch(query_point, k, indices, distances);
    }
    
    template<int DIM>
    size_t radius_search_impl(const float* query_point, float radius_sq, size_t* indices, float* distances, size_t max_results) const {
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);
        std::vector<nanoflann::ResultItem<size_t, float>> matches;
        const size_t num_matches = w->index->radiusSearch(query_point, radius_sq, matches);
        
        const size_t num_to_copy = std::min(num_matches, max_results);
        for (size_t i = 0; i < num_to_copy; ++i) {
            indices[i] = matches[i].first;
            distances[i] = matches[i].second;  // Already squared distance
        }
        return num_to_copy;
    }
    
    template<int DIM>
    void update_points_impl(const float* points, size_t num_points) {
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);
        w->rebuild(points, num_points);
    }
    
    template<int DIM>
    size_t point_count_impl() const {
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);
        return w->cloud->kdtree_get_point_count();
    }
    
    size_t knn_search(const float* query_point, size_t k, size_t* indices, float* distances) const {
        switch (dimensions) {
            case 2: return knn_search_impl<2>(query_point, k, indices, distances);
            case 3: return knn_search_impl<3>(query_point, k, indices, distances);
            case 4: return knn_search_impl<4>(query_point, k, indices, distances);
            case 8: return knn_search_impl<8>(query_point, k, indices, distances);
            default: return knn_search_impl<-1>(query_point, k, indices, distances);
        }
    }
    
    size_t radius_search(const float* query_point, float radius_sq, size_t* indices, float* distances, size_t max_results) const {
        switch (dimensions) {
            case 2: return radius_search_impl<2>(query_point, radius_sq, indices, distances, max_results);
            case 3: return radius_search_impl<3>(query_point, radius_sq, indices, distances, max_results);
            case 4: return radius_search_impl<4>(query_point, radius_sq, indices, distances, max_results);
            case 8: return radius_search_impl<8>(query_point, radius_sq, indices, distances, max_results);
            default: return radius_search_impl<-1>(query_point, radius_sq, indices, distances, max_results);
        }
    }
    
    void update_points(const float* points, size_t num_points) {
        switch (dimensions) {
            case 2: update_points_impl<2>(points, num_points); break;
            case 3: update_points_impl<3>(points, num_points); break;
            case 4: update_points_impl<4>(points, num_points); break;
            case 8: update_points_impl<8>(points, num_points); break;
            default: update_points_impl<-1>(points, num_points); break;
        }
    }
    
    size_t point_count() const {
        switch (dimensions) {
            case 2: return point_count_impl<2>();
            case 3: return point_count_impl<3>();
            case 4: return point_count_impl<4>();
            case 8: return point_count_impl<8>();
            default: return point_count_impl<-1>();
        }
    }
};

// C interface functions
extern "C" {

NanoflannIndex* nanoflann_create_index(const float* points, size_t num_points, size_t dimensions, size_t leaf_max_size) {
    try {
        if (!points || num_points == 0 || dimensions == 0) {
            return nullptr;
        }
        return new NanoflannIndex(points, num_points, dimensions, leaf_max_size);
    } catch (...) {
        return nullptr;
    }
}

void nanoflann_destroy_index(NanoflannIndex* index) {
    delete index;
}

size_t nanoflann_knn_search(const NanoflannIndex* index, const float* query_point, size_t k, size_t* out_indices, float* out_distances_squared) {
    if (!index || !query_point || !out_indices || !out_distances_squared || k == 0) {
        return 0;
    }
    return index->knn_search(query_point, k, out_indices, out_distances_squared);
}

size_t nanoflann_radius_search(const NanoflannIndex* index, const float* query_point, float radius_squared, size_t* out_indices, float* out_distances_squared, size_t max_results) {
    if (!index || !query_point || !out_indices || !out_distances_squared || max_results == 0) {
        return 0;
    }
    return index->radius_search(query_point, radius_squared, out_indices, out_distances_squared, max_results);
}

void nanoflann_update_points(NanoflannIndex* index, const float* points, size_t num_points) {
    if (index && points && num_points > 0) {
        index->update_points(points, num_points);
    }
}

size_t nanoflann_memory_usage(const NanoflannIndex* index) {
    if (!index) return 0;
    // Rough estimate - could be made more precise
    return index->point_count() * static_cast<size_t>(index->dimensions) * sizeof(float) + sizeof(NanoflannIndex);
}

size_t nanoflann_point_count(const NanoflannIndex* index) {
    if (!index) return 0;
    return index->point_count();
}

size_t nanoflann_dimensions(const NanoflannIndex* index) {
    if (!index) return 0;
    return static_cast<size_t>(index->dimensions);
}

} // extern "C"
