#include "mqi_physics_data.hpp"

#include <vector>
#include <fstream>
#include <iostream>

namespace mqi {

// A simple structure for the binary file header
// This MUST match the structure in the g4_data_parser utility
struct DataHeader {
    int num_materials;
    int num_energy_bins;
    int num_physics_processes;
};

physics_data_manager::physics_data_manager() {
    // Define the input file path
    const std::string input_path = "data/physics/proton_physics.bin";

    // Open the file for reading in binary mode
    std::ifstream in_file(input_path, std::ios::binary);
    if (!in_file) {
        std::cerr << "FATAL: Could not open physics data file: " << input_path << std::endl;
        // In a real application, you'd want to handle this more gracefully
        // For now, we'll just exit.
        exit(1);
    }

    // Read the header
    DataHeader header;
    in_file.read(reinterpret_cast<char*>(&header), sizeof(DataHeader));
    if (!in_file) {
        std::cerr << "FATAL: Could not read header from " << input_path << std::endl;
        exit(1);
    }

    // Read the data into a host vector
    const size_t data_size = static_cast<size_t>(header.num_materials) * header.num_energy_bins * header.num_physics_processes;
    std::vector<float> host_data(data_size);
    in_file.read(reinterpret_cast<char*>(host_data.data()), data_size * sizeof(float));
    if (!in_file) {
        std::cerr << "FATAL: Could not read data from " << input_path << std::endl;
        exit(1);
    }
    in_file.close();

    // Calculate max_sigma for Woodcock tracking.
    // This should be the maximum total cross-section across all materials and energies.
    max_sigma_ = 0.0f;
    for (int e = 0; e < header.num_energy_bins; ++e) {
        for (int m = 0; m < header.num_materials; ++m) {
            float total_cs_for_this_config = 0.0f;
            // Sum all processes that are cross-sections.
            // Assuming process 0 is stopping power, and all others are cross-sections.
            for (int p = 1; p < header.num_physics_processes; ++p) {
                size_t idx = (p * header.num_materials * header.num_energy_bins) +
                             (m * header.num_energy_bins) + e;
                if (idx < host_data.size()) {
                    total_cs_for_this_config += host_data[idx];
                }
            }
            if (total_cs_for_this_config > max_sigma_) {
                max_sigma_ = total_cs_for_this_config;
            }
        }
    }

    // Create CUDA 3D array
    cudaExtent extent = make_cudaExtent(header.num_energy_bins, header.num_materials, header.num_physics_processes);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMalloc3DArray(&cu_array_, &channelDesc, extent));

    // Copy data to CUDA array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(host_data.data(), extent.width * sizeof(float), extent.width, extent.height);
    copyParams.dstArray = cu_array_;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array_;

    // Specify texture object parameters for 3D texture
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // Energy
    texDesc.addressMode[1] = cudaAddressModeClamp; // Material
    texDesc.addressMode[2] = cudaAddressModeClamp; // Physics Process
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&tex_object_, &resDesc, &texDesc, NULL));
}

physics_data_manager::~physics_data_manager() {
    if (tex_object_) {
        cudaDestroyTextureObject(tex_object_);
    }
    if (cu_array_) {
        cudaFreeArray(cu_array_);
    }
}

} // namespace mqi
