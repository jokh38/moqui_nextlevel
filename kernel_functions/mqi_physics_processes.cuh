#ifndef MQI_PHYSICS_PROCESSES_CUH
#define MQI_PHYSICS_PROCESSES_CUH

#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include <moqui/base/mqi_error_check.hpp>

namespace mc
{

///< Enum to represent the result of a discrete interaction sampling
enum interaction_type_t
{
    NULL_COLLISION,
    ELASTIC,
    INELASTIC
};

/// \brief Applies continuous processes to a particle over a step.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ void
apply_continuous_processes(mqi::track_t<R>*   track,
                           R                  step_length,
                           R                  material_density,
                           cudaTextureObject_t tex,
                           int                material_idx)
{
    // Physics process index for stopping power
    constexpr float STOPPING_POWER_LAYER = 0.0f;

    // Fetch stopping power from the 3D texture using the energy at the start of the step
    const R stopping_power = tex3D<R>(
        tex, track->vtx1.ke, static_cast<R>(material_idx), STOPPING_POWER_LAYER);

    // Calculate energy loss over the step
    const R energy_loss = stopping_power * material_density * step_length;

    // Update the track's energy and deposit the energy lost in the current step
    track->update_post_vertex_energy(energy_loss);
    track->deposit(energy_loss);

    // TODO: Implement Multiple Coulomb Scattering (MCS) logic here.
    // This would involve calculating a scattering angle based on the material properties
    // and step length, and then updating the track's direction vector.
}

/// \brief Samples the discrete interaction type at the end of a step.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ interaction_type_t
sample_discrete_interaction(cudaTextureObject_t tex,
                            R                 energy,
                            int               material_idx,
                            R                 max_sigma,
                            mqi::mqi_rng*     rng)
{
    // Physics process indices for texture lookup, as per the plan
    constexpr float ELASTIC_XS_LAYER = 1.0f;
    constexpr float INELASTIC_XS_LAYER = 2.0f;

    // Fetch cross-sections from the 3D texture
    // u (coord.x) = energy, v (coord.y) = material index, w (coord.z) = physics process
    // The plan is ambiguous about normalization, but existing code uses energy directly.
    // We assume texture coordinates are set up accordingly.
    const R elastic_xs   = tex3D<R>(tex, energy, static_cast<R>(material_idx), ELASTIC_XS_LAYER);
    const R inelastic_xs = tex3D<R>(tex, energy, static_cast<R>(material_idx), INELASTIC_XS_LAYER);

    const R total_cs = elastic_xs + inelastic_xs;

    // If there is no cross-section, it's a null event
    if (total_cs <= 0.0f) {
        return NULL_COLLISION;
    }

    // Woodcock tracking: check if a real interaction occurs (null-collision test)
    if (curand_uniform(rng) > total_cs / max_sigma) {
        return NULL_COLLISION;
    }

    // A real interaction occurred. Sample which one it is.
    if (curand_uniform(rng) < elastic_xs / total_cs) {
        return ELASTIC;
    } else {
        return INELASTIC;
    }
}

} // namespace mc

#endif // MQI_PHYSICS_PROCESSES_CUH
