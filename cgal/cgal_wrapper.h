// cgal_wrapper.h
#ifndef CGAL_WRAPPER_H
#define CGAL_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to the CGAL Kd-tree index
typedef struct CgalKdTreeIndex CgalKdTreeIndex;

// Result structure for radius search
typedef struct CgalKdTreeResult {
    size_t* indices;              // Dynamically allocated array of neighbor indices
    float* distances_squared;     // Dynamically allocated array of squared distances
    size_t count;                 // Number of neighbors found
} CgalKdTreeResult;

// Create a new CGAL Kd-tree index
// points: flat array of points [x1, y1, z1, x2, y2, z2, ...]
// num_points: number of points
// dimensions: dimensionality of each point
CgalKdTreeIndex* cgal_kdtree_create_index(
    const float* points,
    size_t num_points,
    size_t dimensions
);

// Destroy a CGAL Kd-tree index
void cgal_kdtree_destroy_index(CgalKdTreeIndex* index);

// Perform fuzzy sphere radius search
// epsilon: tolerance for fuzzy sphere (use 0.0 for exact radius)
// Returns a CgalKdTreeResult with dynamically allocated arrays
// Caller must free the result with cgal_kdtree_free_result()
CgalKdTreeResult cgal_kdtree_radius_search(
    const CgalKdTreeIndex* index,
    const float* query_point,
    float radius_squared,
    float epsilon
);

// Free the result from a radius search
void cgal_kdtree_free_result(CgalKdTreeResult* result);

// Update the index with new point positions
// Note: This rebuilds the entire index
void cgal_kdtree_update_points(
    CgalKdTreeIndex* index,
    const float* points,
    size_t num_points
);

// Get the number of points in the index
size_t cgal_kdtree_point_count(const CgalKdTreeIndex* index);

// Get the dimensionality of the index
size_t cgal_kdtree_dimensions(const CgalKdTreeIndex* index);

#ifdef __cplusplus
}
#endif

#endif // CGAL_WRAPPER_H
