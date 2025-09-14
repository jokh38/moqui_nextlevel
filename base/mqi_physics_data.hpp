#ifndef MQI_PHYSICS_DATA_HPP
#define MQI_PHYSICS_DATA_HPP

#include <cuda_runtime.h>
#include <memory>

#include "mqi_error_check.hpp"

namespace mqi {

///< A class to manage physics data and textures on the GPU.
class physics_data_manager {
public:
    /// Constructor: allocates CUDA array and creates texture object.
    physics_data_manager();

    /// Destructor: cleans up CUDA resources.
    ~physics_data_manager();

    /// Returns the CUDA texture object.
    cudaTextureObject_t
    get_texture_object() const {
        return tex_object_;
    }

private:
    /// The CUDA texture object for physics data.
    cudaTextureObject_t tex_object_ = 0;

    /// The CUDA array storing the physics data on the GPU.
    cudaArray* cu_array_ = nullptr;
};

}   // namespace mqi

#endif   // MQI_PHYSICS_DATA_HPP
