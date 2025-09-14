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
    constexpr int STACK_SIZE = 256;
    __shared__ mqi::track_t<R> secondary_stack[STACK_SIZE];
    __shared__ int stack_top;
    if (threadIdx.x == 0) {
        stack_top = 0;
    }
    __syncthreads();

    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
        mqi::track_t<R> current_track = tracks[i];

        do {
            while (!current_track.is_stopped()) {
                // Find current node and geometry
                bool in_geometry = false;
                for (uint32_t c_ind = 0; c_ind < world->n_children; c_ind++) {
                     mqi::grid3d<mqi::density_t, R>& c_geo = *(world->children[c_ind]->geo);
                     current_track.c_node = world->children[c_ind];

                     // Transform particle to geometry's local coordinates
                     mqi::vec3<R> original_pos = current_track.vtx0.pos;
                     mqi::vec3<R> original_dir = current_track.vtx0.dir;
                     current_track.vtx0.pos = c_geo.rotation_matrix_inv * (current_track.vtx0.pos - c_geo.translation_vector);
                     current_track.vtx0.dir = c_geo.rotation_matrix_inv * current_track.vtx0.dir;
                     current_track.vtx0.dir.normalize();
                     current_track.vtx1 = current_track.vtx0;

                     if (c_geo.is_inside(current_track.vtx0.pos)) {
                        in_geometry = true;
                        mqi::cnb_t cnb = c_geo.ijk2cnb(c_geo.index(current_track.vtx0.pos));
                        R rho_mass = c_geo[cnb];

                        // Woodcock Tracking
                        R random_val = curand_uniform(thread_rng);
                        R dist_to_interaction = -log(random_val) / max_sigma;

                        // Find distance to boundary
                        mqi::intersect_t<R> its = c_geo.intersect(current_track.vtx0.pos, current_track.vtx0.dir);
                        R dist_to_boundary = its.dist;

                        R step_length = min(dist_to_interaction, dist_to_boundary);
                        current_track.update_post_vertex_position(step_length);

                        // Apply continuous processes
                        mc::apply_continuous_processes(&current_track, step_length, rho_mass, tex, c_ind);

                        // Scoring
                        for (uint8_t s = 0; s < current_track.c_node->n_scorers; ++s) {
                             if (current_track.c_node->scorers[s]->roi_->idx(cnb) > 0) {
                                insert_hashtable<R>(
                                  current_track.c_node->scorers[s]->data_,
                                  cnb,
                                  scorer_offset_vector ? scorer_offset_vector[i] : mqi::empty_pair,
                                  current_track.dE,
                                  c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                                  current_track.c_node->scorers[s]->max_capacity_);
                            }
                        }

                        // Sample discrete interaction
                        if (dist_to_interaction < dist_to_boundary) {
                            mc::interaction_type_t interaction = mc::sample_discrete_interaction(
                                tex, current_track.vtx1.ke, c_ind, max_sigma, thread_rng);

                            switch (interaction) {
                                case mc::ELASTIC:
                                    // Placeholder
                                    current_track.stop();
                                    break;
                                case mc::INELASTIC:
                                    mc::execute_inelastic_reaction(&current_track,
                                                                   secondary_stack,
                                                                   stack_top,
                                                                   STACK_SIZE,
                                                                   thread_rng);
                                    break;
                                case mc::NULL_COLLISION:
                                    // Continue tracking
                                    break;
                            }
                        }

                        current_track.move();
                     }

                     // Transform back to world coordinates
                     current_track.vtx0.pos = c_geo.rotation_matrix_fwd * current_track.vtx0.pos + c_geo.translation_vector;
                     current_track.vtx0.dir = c_geo.rotation_matrix_fwd * current_track.vtx0.dir;
                     current_track.vtx1 = current_track.vtx0;

                     if (in_geometry) break;
                }
                if (!in_geometry || current_track.vtx1.ke <= 0.0) {
                    current_track.stop();
                }
            }
        } while (mc::pop_from_stack(secondary_stack, stack_top, current_track));
#if defined(__CUDACC__)
        atomicAdd(tracked_particles, 1);
#else
        tracked_particles[0] += 1;
#endif
    }
}

} // namespace mc

#endif // MQI_TRANSPORT_EVENT_HPP
