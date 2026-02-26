// boost_rtree_wrapper.h
#ifndef BOOST_RTREE_WRAPPER_H
#define BOOST_RTREE_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the Boost R-tree index
typedef struct BoostRTreeIndex BoostRTreeIndex;

// Result structure for radius search
typedef struct BoostRTreeResult {
    size_t* indices;              // Dynamically allocated array of neighbor indices
    float* distances_squared;     // Dynamically allocated array of squared distances
    size_t count;                 // Number of neighbors found
} BoostRTreeResult;

// Create a new Boost R-tree index
// points: flat array of points [x1, y1, z1, x2, y2, z2, ...]
// num_points: number of points
// dimensions: dimensionality of each point
BoostRTreeIndex* boost_rtree_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions
);

// Destroy a Boost R-tree index
void boost_rtree_destroy_index(BoostRTreeIndex* index);

// Perform radius search
// Returns a BoostRTreeResult with dynamically allocated arrays
// Caller must free the result with boost_rtree_free_result()
BoostRTreeResult boost_rtree_radius_search(
    const BoostRTreeIndex* index,
    const float* query_point,
    float radius_squared
);

// Free the result from a radius search
void boost_rtree_free_result(BoostRTreeResult* result);

// Update the index with new point positions
// Note: This rebuilds the entire index
void boost_rtree_update_points(
    BoostRTreeIndex* index,
    const float* points,
    size_t num_points
);

// Get the number of points in the index
size_t boost_rtree_point_count(const BoostRTreeIndex* index);

// Get the dimensionality of the index
size_t boost_rtree_dimensions(const BoostRTreeIndex* index);

#ifdef __cplusplus
}
#endif

#endif // BOOST_RTREE_WRAPPER_H
