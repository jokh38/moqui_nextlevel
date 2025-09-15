#ifndef MQI_PHYSICS_PROCESSES_CUH
#define MQI_PHYSICS_PROCESSES_CUH

#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_math.hpp> // For mqi::PI

namespace mqi {
///< PI constant
constexpr float PI = 3.14159265358979323846f;
}

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

/// \brief Pushes a secondary particle onto the shared memory stack.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ void
push_to_stack(mqi::track_t<R>*  secondary_stack,
              int&              stack_top,
              const int         stack_size,
              const mqi::track_t<R>& new_track)
{
    // Use atomicAdd to get a unique index on the stack.
    // This is the 'push' operation.
    int index = atomicAdd(&stack_top, 1);
    if (index < stack_size) {
        secondary_stack[index] = new_track;
    }
    // If index >= stack_size, the stack is full. The particle is lost.
    // A more robust implementation might handle this case.
}

/// \brief Pops a particle from the shared memory stack for the current thread to process.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ bool
pop_from_stack(mqi::track_t<R>*  secondary_stack,
               int&              stack_top,
               mqi::track_t<R>&  current_track)
{
    // Atomically decrement the stack counter. The returned value is the count *before* decrementing.
    // The index of the item to pop is `count - 1`.
    int index = atomicSub(&stack_top, 1) - 1;

    if (index >= 0) {
        // If the index is valid, a particle was successfully reserved.
        // Copy the track data to the thread's local memory.
        current_track = secondary_stack[index];
        return true;
    } else {
        // The stack was empty. The atomicSub made the counter negative. Reset it to 0.
        atomicExch(&stack_top, 0);
        return false; // No particle was popped.
    }
}

/// \brief Generates a random isotropic direction vector.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ mqi::vec3<R>
get_isotropic_direction(mqi::mqi_rng* rng)
{
    R z = 2.0 * curand_uniform(rng) - 1.0;
    R phi = 2.0 * mqi::PI * curand_uniform(rng);
    R r = sqrt(1.0 - z*z);
    R x = r * cos(phi);
    R y = r * sin(phi);
    return mqi::vec3<R>(x, y, z);
}


/// \brief Implements the final state model for an inelastic reaction.
/// \tparam R Data type (e.g., float or double).
template<typename R>
__device__ void
execute_inelastic_reaction(mqi::track_t<R>*  primary_track,
                           mqi::track_t<R>*  secondary_stack,
                           int&              stack_top,
                           const int         stack_size,
                           mqi::mqi_rng*     rng)
{
    // 1. The primary proton is stopped.
    primary_track->stop();

    // 2. Create a new secondary proton.
    mqi::track_t<R> secondary;

    // 3. The secondary's starting vertex is the primary's interaction point.
    secondary.vtx0 = primary_track->vtx1;
    secondary.vtx1 = primary_track->vtx1; // vtx0 and vtx1 are the same initially.

    // 4. The secondary's energy is sampled from a simple distribution.
    secondary.vtx1.ke = primary_track->vtx1.ke * curand_uniform(rng);

    // 5. The secondary's direction is isotropic.
    secondary.vtx1.dir = get_isotropic_direction<R>(rng);

    // 6. Set other properties for the secondary.
    secondary.particle = mqi::PROTON;
    secondary.primary = false;
    secondary.status = mqi::CREATED;
    secondary.c_node = primary_track->c_node; // Assume it starts in the same node.

    // 7. Push the new secondary to the shared stack.
    push_to_stack(secondary_stack, stack_top, stack_size, secondary);
}


} // namespace mc

#endif // MQI_PHYSICS_PROCESSES_CUH
