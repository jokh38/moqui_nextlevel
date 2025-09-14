#include <iostream>
#include <fstream>
#include <vector>

// A simple structure for the binary file header
struct DataHeader {
    int num_materials;
    int num_energy_bins;
    int num_physics_processes;
};

int main() {
    // Define the dimensions of our dummy data
    const int num_materials = 2;
    const int num_energy_bins = 100;
    const int num_physics_processes = 3; // Stopping Power, Elastic XS, Inelastic XS

    // Create the header
    DataHeader header = {num_materials, num_energy_bins, num_physics_processes};

    // Generate some dummy data
    std::vector<float> data(num_materials * num_energy_bins * num_physics_processes);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i); // Simple dummy data
    }

    // Define the output file path
    // We assume the executable is run from the build directory
    const std::string output_path = "../../data/physics/proton_physics.bin";

    // Open the file for writing in binary mode
    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) {
        std::cerr << "Error: Could not open file for writing: " << output_path << std::endl;
        return 1;
    }

    // Write the header
    out_file.write(reinterpret_cast<const char*>(&header), sizeof(DataHeader));

    // Write the data
    out_file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

    // Close the file
    out_file.close();

    std::cout << "Successfully created dummy physics file at: " << output_path << std::endl;

    return 0;
}
