// wembed_snn_wrapper.h
// C-compatible wrapper interface for the wembed SNN (Sorted Nearest Neighbors) data structure.
// Wraps the original implementation by Stefan Guettel and Xinye Chen (MIT License, 2022)

#ifndef WEMBED_SNN_WRAPPER_H
#define WEMBED_SNN_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the SNN index
typedef struct WembedSnnIndex WembedSnnIndex;

// Result structure for radius search
typedef struct WembedSnnResult {
    size_t* indices;    // Dynamically allocated array of neighbor indices
    size_t count;       // Number of neighbors found
} WembedSnnResult;

// Create a new SNN index
// points: flat array of points in row-major order [x1, y1, z1, x2, y2, z2, ...]
// num_points: number of points
// dimensions: dimensionality of each point
WembedSnnIndex* wembed_snn_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions
);

// Destroy an SNN index
void wembed_snn_destroy_index(WembedSnnIndex* index);

// Perform radius search
// query_point: the query point (flat array of size dimensions)
// radius: the search radius (NOT squared)
// Returns a WembedSnnResult with dynamically allocated indices array
// Caller must free the result with wembed_snn_free_result()
WembedSnnResult wembed_snn_radius_search(
    WembedSnnIndex* index,
    const float* query_point,
    float radius
);

// Free the result from a radius search
void wembed_snn_free_result(WembedSnnResult* result);

// Get the number of points in the index
size_t wembed_snn_point_count(const WembedSnnIndex* index);

// Get the dimensionality of the index
size_t wembed_snn_dimensions(const WembedSnnIndex* index);

#ifdef __cplusplus
}
#endif

#endif // WEMBED_SNN_WRAPPER_H
