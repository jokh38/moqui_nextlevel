
## **MOQUI Next-Gen GPU Transport Engine: Phased Execution Plan**

This document outlines the detailed execution plans for the MOQUI Next-Generation GPU Transport Engine. Each phase includes objectives, quantitative success metrics, key technologies, target modules with specific variables, detailed procedures, and an expanded **impact analysis with concrete mitigation strategies** to ensure development stability.

---

### **Phase 1: Texture Memory Implementation**

This phase focuses on transitioning the core data access mechanism to CUDA texture memory for immediate performance improvements.

* **Objective**: Convert the physics data access in the `Condensed History` kernel to use **texture memory**, establishing a technical foundation for future optimizations.
* **Success Metrics**:
    * **Primary**: Achieve a **15% or greater reduction** in the execution time of the `transport_particles_patient` kernel compared to the baseline.
    * **Secondary**: Maintain physics result consistency (e.g., dose distribution) with a difference of less than 0.1% compared to the baseline.
* **Key Technology**: **CUDA Texture Memory**

#### **1. Target Modules and Files**

| File Path | Class / Function | Description of Changes |
| :--- | :--- | :--- |
| `base/mqi_physics_data.hpp` | `physics_data_manager` | Declare the new class for managing physics data and textures. |
| `base/mqi_physics_data.cu` | `physics_data_manager` | Implement the class (data loading, texture object creation). |
| `kernel_functions/mqi_transport.hpp` | `transport_particles_patient` | Modify the data access method within the kernel. |
| `base/mqi_treatment_session.cpp` | `treatment_session` | Instantiate `physics_data_manager` and pass the texture object to the kernel. |
| `CMakeLists.txt` | - | Add the new `.hpp` and `.cu` files to the build system. |

#### **2. Key Variables and Logic**

* **New Class**: `mqi::physics_data_manager`
* **Key Member Variables** (`physics_data_manager`):
    * `cudaTextureObject_t tex_object;`: Stores the created CUDA texture object.
    * `cudaArray* cu_array;`: Stores the physics data on the GPU for texture binding.
* **Texture Coordinate Mapping**:
    * Physics data (cross-sections, stopping power) will be mapped to a 2D texture.
    * Mapping Strategy: `u` coordinate for **energy index**, `v` coordinate for **material index**.
* **Kernel Modification**:
    * Global memory array lookups will be replaced with `tex2D(tex_object, u, v)` calls.

#### **3. Execution Procedure**

1.  **Implement `physics_data_manager`**: Create `.hpp`/`.cu` files and implement the class, including CUDA array creation, texture object management, and cleanup logic.
2.  **Modify Kernel**: Update the `transport_particles_patient` kernel to accept a `cudaTextureObject_t` argument and replace data access logic with `tex2D()` calls based on the defined coordinate mapping.
3.  **Update Session Logic**: In `mqi_treatment_session.cpp`, instantiate the manager, prepare the texture object, and pass it to the kernel launch.
4.  **Update Build System**: Add the new files to `CMakeLists.txt` to ensure they are compiled and linked.

#### **4. Impact Analysis and Mitigation Plan**

* **Potential Issue 1: CUDA API Errors & Missing Headers**
    * **Symptom**: Runtime failures from calls like `cudaMallocArray` or `cudaCreateTextureObject`; compilation errors due to undefined functions.
    * **Mitigation**:
        * Wrap all CUDA API calls with the `check_cuda_last_error` macro to ensure immediate error reporting.
        * **Action**: Verify that `mqi_error_check.hpp` is correctly included and that its path is accessible. Ensure necessary headers like `cuda_runtime.h` are included in files making direct CUDA API calls.
* **Potential Issue 2: Kernel Signature Mismatch**
    * **Symptom**: Compilation error indicating that the function call to `transport_particles_patient` does not match the function's definition.
    * **Mitigation**: As planned, the kernel launch code in `mqi_treatment_session.cpp` must be updated simultaneously to pass the new `cudaTextureObject_t` argument correctly.
* **Potential Issue 3: Build System Errors**
    * **Symptom**: "Undefined reference" linker errors for the `physics_data_manager` class.
    * **Mitigation**: The build system update is an explicit step in the procedure. This ensures the new class and its implementation are correctly linked.

---

### **Phase 2: Adaptive Particle Sorting Implementation**

This phase introduces a sorting algorithm to optimize thread and memory behavior based on particle energy.

* **Objective**: Implement an **Adaptive Particle Sorting** algorithm to minimize thread divergence and non-coalesced memory access, thereby maximizing GPU utilization.
* **Success Metrics**:
    * **Primary**: Achieve an **additional 10% or greater reduction** in total simulation time for scenarios with high initial energy spread.
    * **Secondary**: Demonstrate a measurable decrease in thread divergence through GPU profiling tools (e.g., NVIDIA Nsight Compute).
* **Key Technology**: **Parallel Sort using the Thrust Library**

#### **1. Target Modules and Files**

| File Path | Class / Function | Description of Changes |
| :--- | :--- | :--- |
| `base/mqi_treatment_session.cpp` | `run_simulation` | Replace single kernel launch with a `while` loop containing adaptive sorting logic. |
| `base/mqi_track.hpp` | `track_t` | Add a comparison functor for sorting by energy. |
| `CMakeLists.txt` | - | Add Thrust library dependencies and necessary compiler flags. |

#### **2. Key Variables and Logic**

* **Control Flow**: The `run_simulation` function will be refactored into a `while` loop that continues until all particles are inactive.
* **Adaptive Logic Variables** (`run_simulation`):
    * `float divergence_rate;`: Stores the measured warp divergence rate from the profiler API or a custom kernel.
    * `const float DIVERGENCE_THRESHOLD = 0.2;`: An empirically determined threshold to trigger sorting.
* **Sorting Functor** (`mqi_track.hpp`): A `struct by_energy` with a `__host__ __device__` operator() will be implemented to compare the `kinetic_energy` of two `track_t` objects.
* **Active Particle Tracking**: The number of active particles will be tracked on the GPU using an **atomic counter** updated within the transport kernel, which will then be copied to the host to control the `while` loop.

#### **3. Execution Procedure**

1.  **Implement Comparison Logic**: Define the `by_energy` comparator in `mqi_track.hpp`.
2.  **Refactor Simulation Loop**: In `mqi_treatment_session.cpp`, implement the `while` loop. At the start of each iteration, measure divergence. If it exceeds `DIVERGENCE_THRESHOLD`, call `thrust::sort` on the particle array.
3.  **Implement Termination Condition**: Add the atomic counter logic to the kernel and the host-side check to terminate the loop correctly.
4.  **Update Build System**: Modify `CMakeLists.txt` to link Thrust and add required NVCC compiler flags.

#### **4. Impact Analysis and Mitigation Plan**

* **Potential Issue 1: Thrust Comparator Compilation Errors**
    * **Symptom**: Compilation fails with errors related to lambda functions or functors being inaccessible from device code.
    * **Mitigation**:
        * **Action**: Ensure the comparison functor or lambda function passed to `thrust::sort` is explicitly marked with `__host__ __device__` specifiers.
        * **Action**: In `CMakeLists.txt`, add the `--extended-lambda` flag to the NVCC compiler options to enable the use of modern C++ lambdas in device code.
* **Potential Issue 2: Missing Class Members in Comparator**
    * **Symptom**: Compilation fails with an error stating that a member (e.g., `kinetic_energy`) is not part of the `mqi::track_t` struct.
    * **Mitigation**:
        * **Action**: Before implementation, inspect the `mqi::track_t` class definition in `mqi_track.hpp`. Verify the exact member variable name for kinetic energy and use it in the comparator.
* **Potential Issue 3: Infinite Loop**
    * **Symptom**: The simulation runs indefinitely and never terminates.
    * **Mitigation**: The active particle counting mechanism must be robust. The use of an atomic counter on a dedicated GPU variable, which is then copied to the host after each kernel execution, will ensure a reliable termination condition.

---

### **Phase 3: `Event-by-Event` Kernel Integration**

This phase involves developing a new, high-performance kernel and integrating it as a selectable option.

* **Objective**: Develop a high-performance **`Event-by-Event` transport kernel** and integrate it, allowing runtime selection between transport models.
* **Success Metrics**:
    * **Primary**: For sparse, heterogeneous geometries, the `Event-by-Event` kernel should demonstrate **superior performance** compared to the optimized `Condensed History` kernel from Phase 2.
    * **Secondary**: Physics validation against a known standard (e.g., Geant4) for a set of benchmark cases must show agreement within 2% for key dosimetric indicators.
* **Key Technologies**: **Woodcock Tracking**, **Shared Memory Stack**, **Warp-Level Primitives**

#### **1. Target Modules and Files**

| File Path | Class / Function | Description of Changes |
| :--- | :--- | :--- |
| `kernel_functions/mqi_transport_event.hpp` | `transport_event_by_event_kernel` | Create the new `Event-by-Event` kernel. |
| `base/mqi_treatment_session.hpp` | `treatment_session` | Add a `transport_model` enum class to select the simulation type. |
| `base/mqi_treatment_session.cpp` | `run_simulation` | Implement branching logic to launch the correct kernel. |
| `base/mqi_physics_data.hpp` | `physics_data_manager` | Add `max_sigma` variable for Woodcock tracking. |

#### **2. Key Variables and Logic**

* **New Variable** (`physics_data_manager`):
    * `float max_sigma;`: Stores the maximum total cross-section across all materials, required for Woodcock tracking.
* **Shared Memory Stack** (`transport_event_by_event_kernel`):
    * `__shared__ track_t secondary_stack[256];`: A fixed-size stack in shared memory for managing secondary particles at the block level.
* **Configuration**:
    * An `enum class transport_model { CONDENSED_HISTORY, EVENT_BY_EVENT };` will be added.
    * The model will be selected via a JSON configuration key: `"transport_model": "event_by_event"`.

#### **3. Execution Procedure**

1.  **Enhance Physics Manager**: Add logic to calculate and store `max_sigma` during data loading.
2.  **Develop New Kernel**: Create `mqi_transport_event.hpp` and implement the new kernel, utilizing Woodcock tracking and the shared memory stack.
3.  **Optimize Launch Parameters**: Profile the new kernel to determine the optimal block size that balances shared memory usage and thread occupancy.
4.  **Integrate and Validate**: Add the `enum` and the `switch` statement in `run_simulation` for kernel selection. Perform the physics validation against benchmarks as defined in the success metrics.

#### **4. Impact Analysis and Mitigation Plan**

* **Potential Issue 1: Complex Kernel Bugs (e.g., Race Conditions)**
    * **Symptom**: Incorrect physics results, memory corruption, or kernel crashes detected by `cuda-memcheck`.
    * **Mitigation**: Use `cuda-memcheck` extensively during development to find memory errors. Implement unit tests with simple, verifiable physics (e.g., single particle in a water phantom) to validate the core algorithms before integrating them.
* **Potential Issue 2: Incorrect Kernel Dispatch**
    * **Symptom**: The wrong simulation model is executed despite the user's configuration.
    * **Mitigation**: Add explicit logging at the start of the simulation to print which transport model (`Condensed History` or `Event-by-Event`) was selected. Maintain regression tests for both models.
* **Potential Issue 3: Function Signature and Template Errors**
    * **Symptom**: Hard-to-debug compilation errors related to template instantiation or function overloading when calling existing utility functions from the new kernel.
    * **Mitigation**:
        * **Action**: When calling utility functions like `mqi::io::save_npz` or `mqi::grid3d::index` from the new kernel, pay strict attention to argument types. During code review, a checklist item will be to verify that the function call signatures perfectly match the function declarations.