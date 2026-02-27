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
            case 6:
                impl = new NanoflannWrapperImpl<6>(points, num_points, dims, leaf_max_size);
                break;
            case 8:
                impl = new NanoflannWrapperImpl<8>(points, num_points, dims, leaf_max_size);
                break;
            case 10:
                impl = new NanoflannWrapperImpl<10>(points, num_points, dims, leaf_max_size);
                break;
            case 12:
                impl = new NanoflannWrapperImpl<12>(points, num_points, dims, leaf_max_size);
                break;
            case 14:
                impl = new NanoflannWrapperImpl<14>(points, num_points, dims, leaf_max_size);
                break;
            case 16:
                impl = new NanoflannWrapperImpl<16>(points, num_points, dims, leaf_max_size);
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
            case 6: delete static_cast<NanoflannWrapperImpl<6>*>(impl); break;
            case 8: delete static_cast<NanoflannWrapperImpl<8>*>(impl); break;
            case 10: delete static_cast<NanoflannWrapperImpl<10>*>(impl); break;
            case 12: delete static_cast<NanoflannWrapperImpl<12>*>(impl); break;
            case 14: delete static_cast<NanoflannWrapperImpl<14>*>(impl); break;
            case 16: delete static_cast<NanoflannWrapperImpl<16>*>(impl); break;
            default: delete static_cast<NanoflannWrapperImpl<-1>*>(impl); break;
        }
    }

    template<int DIM>
    NanoflannResult knn_search_impl(const float* query_point, size_t k) const {
        NanoflannResult result = {nullptr, nullptr, 0};
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);

        std::vector<size_t> indices(k);
        std::vector<float> distances(k);
        size_t num_found = w->index->knnSearch(query_point, k, indices.data(), distances.data());

        result.count = num_found;
        if (num_found > 0) {
            result.indices = new size_t[num_found];
            result.distances_squared = new float[num_found];
            for (size_t i = 0; i < num_found; ++i) {
                result.indices[i] = indices[i];
                result.distances_squared[i] = distances[i];
            }
        }
        return result;
    }

    template<int DIM>
    NanoflannResult radius_search_impl(const float* query_point, float radius_sq) const {
        NanoflannResult result = {nullptr, nullptr, 0};
        auto* w = static_cast<NanoflannWrapperImpl<DIM>*>(impl);

        std::vector<nanoflann::ResultItem<size_t, float>> matches;
        w->index->radiusSearch(query_point, radius_sq, matches);

        result.count = matches.size();
        if (result.count > 0) {
            result.indices = new size_t[result.count];
            result.distances_squared = new float[result.count];
            for (size_t i = 0; i < result.count; ++i) {
                result.indices[i] = matches[i].first;
                result.distances_squared[i] = matches[i].second;
            }
        }
        return result;
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

    NanoflannResult knn_search(const float* query_point, size_t k) const {
        switch (dimensions) {
            case 2: return knn_search_impl<2>(query_point, k);
            case 3: return knn_search_impl<3>(query_point, k);
            case 4: return knn_search_impl<4>(query_point, k);
            case 6: return knn_search_impl<6>(query_point, k);
            case 8: return knn_search_impl<8>(query_point, k);
            case 10: return knn_search_impl<10>(query_point, k);
            case 12: return knn_search_impl<12>(query_point, k);
            case 14: return knn_search_impl<14>(query_point, k);
            case 16: return knn_search_impl<16>(query_point, k);
            default: return knn_search_impl<-1>(query_point, k);
        }
    }

    NanoflannResult radius_search(const float* query_point, float radius_sq) const {
        switch (dimensions) {
            case 2: return radius_search_impl<2>(query_point, radius_sq);
            case 3: return radius_search_impl<3>(query_point, radius_sq);
            case 4: return radius_search_impl<4>(query_point, radius_sq);
            case 6: return radius_search_impl<6>(query_point, radius_sq);
            case 8: return radius_search_impl<8>(query_point, radius_sq);
            case 10: return radius_search_impl<10>(query_point, radius_sq);
            case 12: return radius_search_impl<12>(query_point, radius_sq);
            case 14: return radius_search_impl<14>(query_point, radius_sq);
            case 16: return radius_search_impl<16>(query_point, radius_sq);
            default: return radius_search_impl<-1>(query_point, radius_sq);
        }
    }

    void update_points(const float* points, size_t num_points) {
        switch (dimensions) {
            case 2: update_points_impl<2>(points, num_points); break;
            case 3: update_points_impl<3>(points, num_points); break;
            case 4: update_points_impl<4>(points, num_points); break;
            case 6: update_points_impl<6>(points, num_points); break;
            case 8: update_points_impl<8>(points, num_points); break;
            case 10: update_points_impl<10>(points, num_points); break;
            case 12: update_points_impl<12>(points, num_points); break;
            case 14: update_points_impl<14>(points, num_points); break;
            case 16: update_points_impl<16>(points, num_points); break;
            default: update_points_impl<-1>(points, num_points); break;
        }
    }

    size_t point_count() const {
        switch (dimensions) {
            case 2: return point_count_impl<2>();
            case 3: return point_count_impl<3>();
            case 4: return point_count_impl<4>();
            case 6: return point_count_impl<6>();
            case 8: return point_count_impl<8>();
            case 10: return point_count_impl<10>();
            case 12: return point_count_impl<12>();
            case 14: return point_count_impl<14>();
            case 16: return point_count_impl<16>();
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

NanoflannResult nanoflann_knn_search(const NanoflannIndex* index, const float* query_point, size_t k) {
    NanoflannResult result = {nullptr, nullptr, 0};
    if (!index || !query_point || k == 0) {
        return result;
    }
    return index->knn_search(query_point, k);
}

NanoflannResult nanoflann_radius_search(const NanoflannIndex* index, const float* query_point, float radius_squared) {
    NanoflannResult result = {nullptr, nullptr, 0};
    if (!index || !query_point) {
        return result;
    }
    return index->radius_search(query_point, radius_squared);
}

void nanoflann_free_result(NanoflannResult* result) {
    if (result) {
        delete[] result->indices;
        delete[] result->distances_squared;
        result->indices = nullptr;
        result->distances_squared = nullptr;
        result->count = 0;
    }
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
