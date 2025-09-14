# CMakeLists.txt for the G4 data parser utility

# Define the project
project(G4DataParser)

# Add the executable
add_executable(g4_data_parser main.cpp)

# It's good practice to specify the C++ standard
set_target_properties(g4_data_parser PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

# Define where the executable will be placed
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
