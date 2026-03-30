// wembed_snn_wrapper.cpp
// C++ wrapper for wembed SNN providing a C-compatible interface for Rust FFI.
// Wraps the original implementation by Stefan Guettel and Xinye Chen (MIT License, 2022)

#include "wembed_snn_wrapper.h"
#include "snn.h"
#include <vector>
#include <memory>

// Wrapper struct holding the SnnModel and metadata
struct WembedSnnIndex {
    std::unique_ptr<SnnModel> model;
    size_t num_points;
    size_t dimensions;
    // Buffers for query operations (reused to avoid allocations)
    mutable Eigen::VectorXf query_buffer;
    mutable Eigen::VectorXf distance_buffer;
};

extern "C" {

WembedSnnIndex* wembed_snn_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions
) {
    if (num_points == 0 || dimensions == 0) {
        return nullptr;
    }

    // Convert from row-major to column-major
    // SnnModel expects column-major: data[r + rows * c]
    std::vector<float> data(num_points * dimensions);
    for (size_t r = 0; r < num_points; ++r) {
        for (size_t c = 0; c < dimensions; ++c) {
            // Input is row-major: points[r * dimensions + c]
            // Output is column-major: data[r + num_points * c]
            data[r + num_points * c] = points[r * dimensions + c];
        }
    }

    auto* index = new WembedSnnIndex;
    index->model = std::make_unique<SnnModel>(
        data.data(),
        static_cast<int>(num_points),
        static_cast<int>(dimensions)
    );
    index->num_points = num_points;
    index->dimensions = dimensions;
    index->query_buffer.resize(dimensions);
    index->distance_buffer.resize(num_points);

    return index;
}

void wembed_snn_destroy_index(WembedSnnIndex* index) {
    delete index;
}

WembedSnnResult wembed_snn_radius_search(
    WembedSnnIndex* index,
    const float* query_point,
    float radius
) {
    WembedSnnResult result = {nullptr, 0};

    if (!index || !index->model) {
        return result;
    }

    // Wrap query point directly (no conversion needed)
    std::vector<float> query(query_point, query_point + index->dimensions);

    // Perform the query - results vector grows dynamically
    std::vector<size_t> results;
    index->model->radius_single_query(
        query,
        radius,
        results,
        [](int id) -> size_t { return static_cast<size_t>(id); },
        index->query_buffer,
        index->distance_buffer
    );

    // Allocate and copy results
    result.count = results.size();
    if (result.count > 0) {
        result.indices = new size_t[result.count];
        for (size_t i = 0; i < result.count; ++i) {
            result.indices[i] = results[i];
        }
    }

    return result;
}

void wembed_snn_free_result(WembedSnnResult* result) {
    if (result && result->indices) {
        delete[] result->indices;
        result->indices = nullptr;
        result->count = 0;
    }
}

size_t wembed_snn_point_count(const WembedSnnIndex* index) {
    return index ? index->num_points : 0;
}

size_t wembed_snn_dimensions(const WembedSnnIndex* index) {
    return index ? index->dimensions : 0;
}

}  // extern "C"
