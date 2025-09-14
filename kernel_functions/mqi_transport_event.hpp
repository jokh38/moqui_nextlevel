#ifndef MQI_TRANSPORT_EVENT_HPP
#define MQI_TRANSPORT_EVENT_HPP

#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_track_stack.hpp>
#include <moqui/base/mqi_node.hpp>
#include <moqui/kernel_functions/mqi_physics_processes.cuh>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include <moqui/base/mqi_error_check.hpp>

namespace mc {

/// \brief Event-by-event transport kernel using Woodcock tracking.
/// \tparam R Data type (e.g., float or double).
template<typename R>
CUDA_GLOBAL void
transport_event_by_event_kernel(cudaTextureObject_t tex,
                                float             max_sigma,
                                mqi::thrd_t*      threads,
                                mqi::node_t<R>*   world,
                                mqi::track_t<R>*  tracks,
                                const uint32_t    n_vtx,
                                uint32_t*         tracked_particles,
                                uint32_t*         scorer_offset_vector = nullptr,
                                bool              score_local_deposit  = true,
                                uint32_t          total_threads        = 1,
                                uint32_t          thread_id            = 0)
{
#if defined(__CUDACC__)
    thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    total_threads = (blockDim.x * gridDim.x);
#endif

    const mqi::vec2<uint32_t> h_range = mqi::start_and_length(total_threads, n_vtx, thread_id);
    mqi::mqi_rng* thread_rng = &threads[thread_id].rnd_generator;

    // Shared memory for secondary particle stack
    __shared__ mqi::track_t<R> secondary_stack[256];
    __shared__ int stack_top;
    if (threadIdx.x == 0) {
        stack_top = -1;
    }
    __syncthreads();

    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
        mqi::track_t<R>* primary = &tracks[i];
        mqi::track_stack_t<R> local_stack;
        local_stack.push_secondary(*primary);

        while (!local_stack.is_empty()) {
            mqi::track_t<R> track = local_stack.pop();

            while (!track.is_stopped()) {
                // Find current node and geometry
                // This simplified loop assumes the particle is in one of the children of the world.
                // A more robust geometry navigator would be needed for complex cases.
                bool in_geometry = false;
                for (uint32_t c_ind = 0; c_ind < world->n_children; c_ind++) {
                     mqi::grid3d<mqi::density_t, R>& c_geo = *(world->children[c_ind]->geo);
                     track.c_node = world->children[c_ind];

                     // Transform particle to geometry's local coordinates
                     mqi::vec3<R> original_pos = track.vtx0.pos;
                     mqi::vec3<R> original_dir = track.vtx0.dir;
                     track.vtx0.pos = c_geo.rotation_matrix_inv * (track.vtx0.pos - c_geo.translation_vector);
                     track.vtx0.dir = c_geo.rotation_matrix_inv * track.vtx0.dir;
                     track.vtx0.dir.normalize();
                     track.vtx1 = track.vtx0;

                     if (c_geo.is_inside(track.vtx0.pos)) {
                        in_geometry = true;
                        mqi::cnb_t cnb = c_geo.ijk2cnb(c_geo.index(track.vtx0.pos));
                        R rho_mass = c_geo[cnb];

                        // Woodcock Tracking: sample distance to next potential interaction site
                        R random_val = curand_uniform(thread_rng);
                        R dist_to_interaction = -log(random_val) / max_sigma;

                        // Find distance to the boundary of the current geometry voxel
                        mqi::intersect_t<R> its = c_geo.intersect(track.vtx0.pos, track.vtx0.dir);
                        R dist_to_boundary = its.dist;

                        // The step length is the minimum of the two distances
                        R step_length = min(dist_to_interaction, dist_to_boundary);
                        track.update_post_vertex_position(step_length);

                        // Apply continuous processes (energy loss) over the step
                        mc::apply_continuous_processes(&track, step_length, rho_mass, tex, c_ind);

                        // Scoring
                        for (uint8_t s = 0; s < track.c_node->n_scorers; ++s) {
                             if (track.c_node->scorers[s]->roi_->idx(cnb) > 0) {
                                insert_hashtable<R>(
                                  track.c_node->scorers[s]->data_,
                                  cnb,
                                  scorer_offset_vector ? scorer_offset_vector[i] : mqi::empty_pair,
                                  track.dE,
                                  c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                                  track.c_node->scorers[s]->max_capacity_);
                            }
                        }

                        // If the step ended on an interaction site, sample the interaction
                        if (dist_to_interaction < dist_to_boundary) {
                            mc::interaction_type_t interaction = mc::sample_discrete_interaction(
                                tex, track.vtx1.ke, c_ind, max_sigma, thread_rng);

                            switch (interaction) {
                                case mc::ELASTIC:
                                    // Final state model for elastic scattering would be called here.
                                    // For now, stop the particle as a placeholder.
                                    track.stop();
                                    break;
                                case mc::INELASTIC:
                                    // Final state model for inelastic reaction would be called here.
                                    // For now, stop the particle as a placeholder.
                                    track.stop();
                                    break;
                                case mc::NULL_COLLISION:
                                    // Do nothing, continue tracking
                                    break;
                            }
                        }

                        track.move();
                     }

                     // Transform back to world coordinates
                     track.vtx0.pos = c_geo.rotation_matrix_fwd * track.vtx0.pos + c_geo.translation_vector;
                     track.vtx0.dir = c_geo.rotation_matrix_fwd * track.vtx0.dir;
                     track.vtx1 = track.vtx0;

                     if (in_geometry) break;
                }
                if (!in_geometry || track.vtx1.ke <= 0.0) {
                    track.stop();
                }
            }
        }
#if defined(__CUDACC__)
        atomicAdd(tracked_particles, 1);
#else
        tracked_particles[0] += 1;
#endif
    }
}

} // namespace mc

#endif // MQI_TRANSPORT_EVENT_HPP
