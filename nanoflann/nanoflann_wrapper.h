// nanoflann_wrapper.h
#ifndef NANOFLANN_WRAPPER_H
#define NANOFLANN_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the nanoflann index
typedef struct NanoflannIndex NanoflannIndex;

// Create a new nanoflann index
// points: flat array of points [x1, y1, z1, x2, y2, z2, ...]
// num_points: number of points
// dimensions: dimensionality of each point
// leaf_max_size: maximum points per leaf node
NanoflannIndex* nanoflann_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions,
    size_t leaf_max_size
);

// Destroy a nanoflann index
void nanoflann_destroy_index(NanoflannIndex* index);

// Perform k-nearest neighbor search
// Returns the number of neighbors found (may be less than k if not enough points)
size_t nanoflann_knn_search(
    const NanoflannIndex* index,
    const float* query_point,
    size_t k,
    size_t* out_indices,
    float* out_distances_squared
);

// Perform radius search
// Returns the number of neighbors found within radius
size_t nanoflann_radius_search(
    const NanoflannIndex* index,
    const float* query_point,
    float radius_squared,
    size_t* out_indices,
    float* out_distances_squared,
    size_t max_results
);

// Update the index with new point positions
// Note: This rebuilds the entire index
void nanoflann_update_points(
    NanoflannIndex* index,
    const float* points,
    size_t num_points
);

// Get memory usage statistics
size_t nanoflann_memory_usage(const NanoflannIndex* index);

// Get the number of points in the index
size_t nanoflann_point_count(const NanoflannIndex* index);

// Get the dimensionality of the index
size_t nanoflann_dimensions(const NanoflannIndex* index);

#ifdef __cplusplus
}
#endif

#endif // NANOFLANN_WRAPPER_H
