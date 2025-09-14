## MOQUI 차세대 GPU 수송 엔진: 단계별 실행 계획 (영향 분석 및 완화 방안 포함)

`수정계획_F.md` 문서에 명시된 MOQUI 차세대 GPU 수송 엔진 개발 계획을 바탕으로, 각 단계별 상세 실행 계획을 개별 문서로 명확하게 정리했습니다. 각 계획은 단계별 목표, 핵심 기술, 수정 대상 모듈과 변수, 구체적인 실행 절차를 포함합니다.

특히, **각 단계의 수정이 다른 시스템 영역에 미칠 수 있는 영향을 사전에 분석**하고, **발생 가능한 문제들을 방지하기 위한 완화 방안을 코드 수정 계획에 명시적으로 통합**하여 개발의 안정성을 극대화하도록 설계되었습니다.

---

### **Phase 1: 텍스처 메모리 적용 실행 계획서**

이 문서는 MOQUI 개발 계획의 **Phase 1**에 대한 상세 실행 계획을 기술합니다.

* **목표**: 기존 `Condensed History` 커널의 물리 데이터 접근 방식을 **텍스처 메모리**로 전환하여 즉각적인 성능 향상을 달성하고, 후속 개발 단계의 기술적 기반을 마련합니다.
* **핵심 기술**: **CUDA 텍스처 메모리 (Texture Memory)**
    * **원리**: 물리 데이터(상호작용 단면적, 저지능 등)를 일반 전역 메모리(Global Memory)가 아닌 2D 텍스처 메모리에 바인딩(binding)하여 접근합니다.
    * **기대 효과**:
        1.  **하드웨어 가속 보간 (Hardware-accelerated Interpolation)**: 텍스처 하드웨어를 통해 자동으로 선형 보간을 수행하여, 소프트웨어 방식보다 월등히 빠른 속도를 제공합니다.
        2.  **전용 캐시 활용**: 텍스처 전용 L1 캐시를 사용하여 공간적 지역성(spatial locality)이 있는 데이터 접근 효율을 극대화합니다.

#### **1. 수정 대상 모듈 및 파일**

| 파일 경로                              | 클래스 / 함수                   | 변경 내용                                       |
| :------------------------------------- | :------------------------------ | :---------------------------------------------- |
| `physics/mqi_physics_data.hpp`         | `physics_data_manager`          | 신규 클래스 선언                                |
| `physics/mqi_physics_data.cu`          | `physics_data_manager`          | 클래스 구현 (데이터 로드, 텍스처 객체 생성)     |
| `kernel_functions/mqi_transport.hpp`   | `transport_particles_patient`   | 커널 내 데이터 접근 방식 수정                   |
| `base/mqi_treatment_session.cpp`       | `treatment_session`             | `physics_data_manager` 객체 생성 및 커널 인자 전달 |
| `CMakeLists.txt`                       | -                               | 신규 파일 (`.hpp`, `.cu`) 빌드 시스템에 추가    |

#### **2. 주요 변수 및 함수 변경 사항**

* **신규 클래스**:
    * `mqi::physics_data_manager`: 물리 데이터를 로드하고 CUDA 텍스처 객체를 관리하는 클래스를 `mqi_physics_data.hpp`에 선언하고 `mqi_physics_data.cu`에 구현합니다.
* **신규 멤버 변수 (`physics_data_manager` 클래스 내)**:
    * `cudaTextureObject_t tex_object;`: 생성된 텍스처 객체를 저장할 변수입니다.
    * `cudaArray* cu_array;`: GPU 메모리에 텍스처 데이터를 저장할 CUDA 배열입니다.
* **커널 함수 수정**:
    * `transport_particles_patient` 커널 내에서 전역 메모리 배열을 통해 물리 데이터를 읽어오던 로직을 `tex2D()` 함수를 사용하여 텍스처 메모리에서 데이터를 조회하도록 변경합니다.

#### **3. 실행 절차**

1.  **`mqi_physics_data.hpp`/`.cu` 파일 생성 및 구현**:
    * `physics` 디렉터리에 `mqi_physics_data.hpp`, `mqi_physics_data.cu` 파일을 생성하고 `physics_data_manager` 클래스를 정의 및 구현합니다.
    * 클래스에는 물리 데이터 로딩, CUDA 배열 생성, 텍스처 객체 생성 및 소멸을 위한 멤버 함수를 선언하고 구현합니다.
2.  **`mqi_transport.hpp` 커널 수정**:
    * `transport_particles_patient` 커널의 인자로 `cudaTextureObject_t` 타입의 텍스처 객체를 받도록 수정합니다.
    * 커널 내부에서 물리 데이터 테이블에 접근하는 모든 코드를 `tex2D()` 함수 호출로 변경합니다.
3.  **`mqi_treatment_session` 수정**:
    * 시뮬레이션 초기화 단계에서 `physics_data_manager` 객체를 생성하고, 물리 데이터를 로드 및 텍스처 객체로 변환하는 코드를 추가합니다.
    * `transport_particles_patient` 커널 호출 시 생성된 텍스처 객체를 인자로 전달합니다.
4.  **빌드 시스템 업데이트**:
    * `CMakeLists.txt`에 새로 추가된 `mqi_physics_data.hpp`와 `mqi_physics_data.cu` 파일이 빌드에 포함되도록 수정합니다.

#### **4. 영향 분석 및 완화 방안**

* **예상 문제 1: CUDA API 오류 및 누락**
    * **영향**: `cudaMallocArray`, `cudaCreateTextureObject` 등 새로운 CUDA API 호출 시 실패하면 런타임 오류가 발생하며, 관련 헤더가 누락되면 컴파일 오류가 발생합니다.
    * **완화 방안**: 모든 CUDA API 호출 직후에 에러 체크 함수(예: `check_cuda_last_error`)를 추가하여 오류 발생 시 즉시 감지하고 원인을 출력하도록 합니다. `mqi_error_check.hpp`와 같은 공용 에러 핸들링 모듈을 적극 활용하고, `cuda_runtime.h` 헤더가 포함되었는지 확인합니다.
* **예상 문제 2: 커널 시그니처 변경으로 인한 호출 오류**
    * **영향**: `transport_particles_patient` 커널의 인자가 변경되므로, 이 커널을 호출하는 `mqi_treatment_session.cpp` 등의 기존 코드는 컴파일 오류를 유발합니다.
    * **완화 방안**: 계획에 명시된 대로 `mqi_treatment_session.cpp`에서 커널을 호출하는 부분을 반드시 수정하여, 새로 생성된 `cudaTextureObject_t` 객체를 정확한 인자로 전달하도록 합니다.
* **예상 문제 3: 빌드 시스템 누락**
    * **영향**: 새로 추가된 `mqi_physics_data.hpp`/`.cu` 파일이 `CMakeLists.txt`에 등록되지 않으면 '정의되지 않은 참조(undefined reference)' 링킹 오류가 발생합니다.
    * **완화 방안**: 실행 절차에 명시된 대로, 파일 생성과 동시에 `CMakeLists.txt`에 해당 파일들을 추가하는 작업을 수행하여 빌드 과정에 포함시킵니다.

---

### **Phase 2: 적응형 입자 재정렬 실행 계획서**

이 문서는 MOQUI 개발 계획의 **Phase 2**에 대한 상세 실행 계획을 기술합니다.

* **목표**: Phase 1의 결과물에 **적응형 입자 재정렬(Adaptive Particle Sorting)** 알고리즘을 추가하여 스레드 분기(Thread Divergence) 및 비병합 메모리 접근(Non-coalesced Memory Access) 문제를 해결하고 GPU 활용률을 극대화합니다.
* **핵심 기술**: **Thrust 라이브러리를 이용한 병렬 정렬 (Parallel Sort)**
    * **원리**: 수송 커널 실행 직전, GPU 상의 전체 입자 배열을 에너지 순(내림차순)으로 정렬하여 한 워프(warp) 내 스레드들이 유사한 에너지의 입자를 처리하게 만듭니다.
    * **기대 효과**:
        1.  **스레드 분기 최소화**: 유사 에너지 입자들의 물리적 상호작용 확률을 높여 실행 경로를 통일시키고 SIMT (Single Instruction, Multiple Threads) 아키텍처 활용도를 극대화합니다.
        2.  **메모리 접근 패턴 개선**: 정렬된 입자들이 물리 데이터 테이블의 인접 영역에 동시 접근하여 텍스처 캐시 히트율을 극대화합니다.

#### **1. 수정 대상 모듈 및 파일**

| 파일 경로                         | 클래스 / 함수      | 변경 내용                                                    |
| :-------------------------------- | :----------------- | :----------------------------------------------------------- |
| `base/mqi_treatment_session.cpp`  | `run_simulation`   | 단일 커널 호출을 `while` 루프로 변경 및 정렬 로직 추가       |
| `base/mqi_track.hpp`              | `track_t`          | 에너지 비교를 위한 `operator<` 또는 functor 추가             |
| `CMakeLists.txt`                  | -                  | Thrust 라이브러리 링크 설정 및 컴파일러 옵션 추가              |

#### **2. 주요 변수 및 함수 변경 사항**

* **`run_simulation` 함수 내 주요 변경**:
    * 기존의 단일 `transport_particles_patient` 커널 호출을 `while` 루프로 감싸 입자가 모두 소멸될 때까지 반복하도록 구조를 변경합니다.
    * `while` 루프의 시작 지점에 `thrust::sort`를 호출하는 코드를 추가하여 `d_tracks`(입자 배열)를 에너지 기준으로 정렬합니다.
* **신규 변수 및 로직 (`run_simulation` 함수 내)**:
    * `divergence_rate` (float): 워프 분기율을 저장할 변수입니다.
    * `DIVERGENCE_THRESHOLD` (const float): 정렬 수행 여부를 결정할 임계값입니다.
* **Functor 또는 연산자 오버로딩**:
    * `mqi_track.hpp`에 `by_energy`와 같은 functor 구조체나 `track_t` 클래스 내에 `operator<`를 정의하여 Thrust가 입자를 에너지 기준으로 정렬할 수 있도록 합니다.

#### **3. 실행 절차**

1.  **Thrust 라이브러리 설정 및 비교 로직 구현**:
    * `CMakeLists.txt` 파일을 수정하여 Thrust 라이브러리를 링크하고, 필요 시 `nvcc` 플래그에 `--extended-lambda`를 추가합니다.
    * `mqi_track.hpp` 파일에 `track_t` 구조체의 에너지 값을 비교하는 `by_energy` functor를 `__host__ __device__` 지정자와 함께 추가합니다.
2.  **`mqi_treatment_session.cpp` 수정**:
    * `run_simulation` 함수의 내용을 `#include <thrust/sort.h>` 등을 추가하고, 기존 커널 호출 부분을 `while` 루프 구조로 변경합니다.
    * 루프 내에서 워프 분기율을 측정하고, 임계값을 초과할 경우 `thrust::sort(thrust::device, d_tracks, d_tracks + n_particles, by_energy());`를 호출하도록 적응형 정렬 로직을 구현합니다.
    * 루프가 정상적으로 종료될 수 있도록, 커널 실행 후 활성 입자 수를 GPU에서 업데이트하는 로직을 반드시 포함합니다.

#### **4. 영향 분석 및 완화 방안**

* **예상 문제 1: Thrust 람다/Functor 컴파일 오류**
    * **영향**: `thrust::sort`에 전달되는 비교 람다 함수나 functor에 `__host__ __device__` 지정자가 없으면 디바이스 코드에서 호출할 수 없어 컴파일 오류가 발생합니다. 또한, 최신 C++ 람다 기능을 사용할 경우 컴파일러 옵션이 필요할 수 있습니다.
    * **완화 방안**: `mqi_track.hpp`에 정의하는 비교 로직(functor 또는 람다)에 반드시 `__host__ __device__` 지정자를 명시합니다. 컴파일 오류 발생 시, `CMakeLists.txt`의 `nvcc` 컴파일러 플래그에 `--extended-lambda` 옵션을 추가하는 것을 검토합니다.
* **예상 문제 2: 클래스 멤버 부재로 인한 컴파일 오류**
    * **영향**: 정렬 기준이 되는 `track_t` 클래스의 멤버 변수(예: `kinetic_energy`) 이름이 변경되었거나 존재하지 않는 경우, 비교 로직에서 해당 멤버에 접근할 때 컴파일 오류가 발생합니다.
    * **완화 방안**: 코딩 전 `mqi_track.hpp`의 `track_t` 클래스 정의를 정확히 확인하여 `kinetic_energy`와 같은 멤버 변수가 실제로 존재하는지, 혹은 다른 이름으로 사용되는지 점검하고 코드에 정확히 반영합니다.
* **예상 문제 3: 시뮬레이션 로직 변경으로 인한 무한 루프**
    * **영향**: `run_simulation`의 단일 커널 호출이 `while` 루프로 변경되면서, 루프 종료 조건(활성 입자 수)이 정확하게 업데이트되지 않으면 무한 루프에 빠져 시뮬레이션이 끝나지 않을 수 있습니다.
    * **완화 방안**: `while` 루프의 매 반복마다 커널 실행 후 GPU 상의 활성 입자 수를 정확히 카운트하여 호스트로 전달하고, 이 값을 루프 종료 조건으로 사용하는 견고한 메커니즘을 구현합니다.

---

### **Phase 3: `Event-by-Event` 커널 통합 실행 계획서**

이 문서는 MOQUI 개발 계획의 **Phase 3**에 대한 상세 실행 계획을 기술합니다.

* **목표**: Phase 1, 2에서 검증된 최적화 기법을 기반으로 고성능 **`Event-by-Event` 수송 커널**을 개발하고, 사용자가 시뮬레이션 모델을 선택할 수 있도록 기존 시스템에 통합합니다.
* **핵심 기술**:
    1.  **Woodcock 트래킹**: 복셀 경계 계산을 최소화하는 알고리즘.
    2.  **공유 메모리 기반 스택**: 2차 입자 관리를 위한 고속 스택 구현.
    3.  **워프 레벨 프리미티브**: `__shfl_sync()` 등을 활용한 레지스터 간 직접 통신.

#### **1. 수정 대상 모듈 및 파일**

| 파일 경로                                  | 클래스 / 함수                         | 변경 내용                                                |
| :----------------------------------------- | :------------------------------------ | :------------------------------------------------------- |
| `kernel_functions/mqi_transport_event.hpp` | `transport_event_by_event_kernel`     | 신규 `Event-by-Event` 커널 작성                          |
| `base/mqi_treatment_session.hpp`           | `treatment_session`                   | `transport_model` enum 클래스 추가                       |
| `base/mqi_treatment_session.cpp`           | `run_simulation`                      | `transport_model` 설정에 따른 커널 분기 로직 구현        |
| `physics/mqi_physics_data.hpp`             | `physics_data_manager`                | Woodcock 트래킹을 위한 `max_sigma` 변수 및 계산 로직 추가 |

#### **2. 주요 변수 및 함수 변경 사항**

* **신규 커널**:
    * `transport_event_by_event_kernel`: `mqi_transport_event.hpp`에 새롭게 작성될 `Event-by-Event` 방식의 입자 수송 커널입니다.
* **신규 변수**:
    * `max_sigma` (float): Woodcock 트래킹에 사용될 전체 물질의 최대 총 단면적. `physics_data_manager` 클래스에 추가됩니다.
    * `__shared__ track_t secondary_stack[256]`: 2차 입자 관리를 위해 커널 내에 선언될 공유 메모리 기반 스택.
* **`treatment_session` 클래스 변경**:
    * `transport_model` (enum class): `CONDENSED_HISTORY`와 `EVENT_BY_EVENT` 두 가지 모드를 선택할 수 있는 열거형 클래스를 추가합니다.
* **`run_simulation` 함수 변경**:
    * JSON 설정 파일에서 `transport_model` 값을 읽어와, 해당 값에 따라 기존 커널 또는 신규 커널을 호출하도록 `if` 또는 `switch` 분기문을 추가합니다.

#### **3. 실행 절차**

1.  **`max_sigma` 계산 로직 추가**:
    * `physics_data_manager` 클래스에 모든 물질의 총 단면적을 순회하며 최대값(`max_sigma`)을 찾아 저장하는 기능을 추가합니다.
2.  **`mqi_transport_event.hpp` 신규 커널 작성**:
    * Woodcock 트래킹, 공유 메모리 스택, 워프 레벨 프리미티브 등 핵심 기술을 적용하여 `transport_event_by_event_kernel`을 구현합니다.
3.  **`mqi_treatment_session` 통합 작업**:
    * `mqi_treatment_session.hpp`에 `transport_model` enum 클래스를 정의합니다.
    * `mqi_treatment_session.cpp`의 `run_simulation` 함수에서 JSON 설정 값을 읽어, `transport_model` 값에 따라 호출할 커널을 결정하는 분기문을 작성합니다.

#### **4. 영향 분석 및 완화 방안**

* **예상 문제 1: 복잡한 커널 로직으로 인한 잠재적 버그**
    * **영향**: 공유 메모리 사용 시 race condition, Woodcock 트래킹의 물리 모델 오류 등 `Event-by-Event` 커널의 복잡성으로 인해 디버깅이 어려운 버그가 발생할 수 있습니다.
    * **완화 방안**: `cuda-memcheck`와 같은 메모리 디버깅 도구를 사용하여 공유 메모리 접근 오류를 검증합니다. 커널의 각 기능(Woodcock, 스택 관리 등)에 대해 단위 테스트를 수행하고, 물리적으로 예측 가능한 간단한 시나리오(예: 균일한 매질, 단일 에너지)에서 결과가 올바른지 검증하는 절차를 포함합니다.
* **예상 문제 2: 분기 로직 오류**
    * **영향**: `run_simulation` 내에서 `transport_model` 설정에 따라 커널을 선택하는 분기 로직에 오류가 있을 경우, 사용자의 의도와 다른 시뮬레이션 모델이 실행될 수 있습니다.
    * **완화 방안**: 시뮬레이션 시작 시 어떤 수송 모델(`Condensed History` 또는 `Event-by-Event`)이 선택되었는지 명시적인 로그를 출력하도록 합니다. 또한, 각 모델에 대한 회귀 테스트(regression test) 스위트를 구축하여 코드 변경 후에도 두 모델이 모두 독립적으로 정상 동작하는지 확인합니다.
* **예상 문제 3: 함수 오버로딩/템플릿 오류**
    * **영향**: 새로운 커널과 데이터 구조를 추가하면서 기존 유틸리티 함수(예: `mqi::io::save_npz`, `mqi::grid3d::index`)를 잘못된 타입의 인자로 호출하여 컴파일 오류가 발생할 수 있습니다.
    * **완화 방안**: 새로운 커널에서 기존 함수를 호출할 때, 호출부와 선언부를 비교하여 인자의 타입과 개수가 정확히 일치하는지 철저히 확인합니다. 코드 리뷰 단계에서 함수 시그니처의 일관성을 중점적으로 점검합니다.