\## MOQUI 차세대 GPU 수송 엔진: 단계별 실행 계획



`수정계획\_F.md` 문서에 명시된 MOQUI 차세대 GPU 수송 엔진 개발 계획을 바탕으로, 각 단계별 상세 실행 계획을 개별 문서로 명확하게 정리했습니다. 각 계획은 단계별 목표, 핵심 기술, 수정 대상 모듈과 변수, 그리고 구체적인 실행 절차를 포함하여, 개발자가 명확한 가이드라인에 따라 작업을 수행하고 다른 기능에 영향을 미치지 않도록 설계되었습니다.



-----



\### Phase 1: 텍스처 메모리 적용 실행 계획서



이 문서는 MOQUI 개발 계획의 \*\*Phase 1\*\*에 대한 상세 실행 계획을 기술합니다.



&nbsp; \* \*\*목표\*\*: 기존 `Condensed History` 커널의 물리 데이터 접근 방식을 \*\*텍스처 메모리\*\*로 전환하여 즉각적인 성능 향상을 달성하고, 후속 개발 단계의 기술적 기반을 마련합니다.

&nbsp; \* \*\*핵심 기술\*\*: \*\*CUDA 텍스처 메모리 (Texture Memory)\*\*

&nbsp;     \* \*\*원리\*\*: 물리 데이터(상호작용 단면적, 저지능 등)를 일반 전역 메모리(Global Memory)가 아닌 2D 텍스처 메모리에 바인딩(binding)하여 접근합니다.

&nbsp;     \* \*\*기대 효과\*\*:

&nbsp;       1.  \*\*하드웨어 가속 보간 (Hardware-accelerated Interpolation)\*\*: 텍스처 하드웨어를 통해 자동으로 선형 보간을 수행하여, 소프트웨어 방식보다 월등히 빠른 속도를 제공합니다.

&nbsp;       2.  \*\*전용 캐시 활용\*\*: 텍스처 전용 L1 캐시를 사용하여 공간적 지역성(spatial locality)이 있는 데이터 접근 효율을 극대화합니다.



\#### \*\*1. 수정 대상 모듈 및 파일\*\*



| 파일 경로 | 클래스 / 함수 | 변경 내용 |

| :--- | :--- | :--- |

| `physics/mqi\_physics\_data.hpp` | `physics\_data\_manager` | 신규 클래스 선언 |

| `physics/mqi\_physics\_data.cu` | `physics\_data\_manager` | 클래스 구현 (데이터 로드, 텍스처 객체 생성) |

| `kernel\_functions/mqi\_transport.hpp` | `transport\_particles\_patient` | 커널 내 데이터 접근 방식 수정 |



\#### \*\*2. 주요 변수 및 함수 변경 사항\*\*



&nbsp; \* \*\*신규 클래스\*\*:

&nbsp;     \* `mqi::physics\_data\_manager`: 물리 데이터를 로드하고 CUDA 텍스처 객체를 관리하는 클래스를 `mqi\_physics\_data.hpp`에 선언하고 `mqi\_physics\_data.cu`에 구현합니다.

&nbsp; \* \*\*신규 멤버 변수 ( `physics\_data\_manager` 클래스 내)\*\*:

&nbsp;     \* `cudaTextureObject\_t tex\_object;`: 생성된 텍스처 객체를 저장할 변수입니다.

&nbsp;     \* `cudaArray\* cu\_array;`: GPU 메모리에 텍스처 데이터를 저장할 CUDA 배열입니다.

&nbsp; \* \*\*커널 함수 수정\*\*:

&nbsp;     \* `transport\_particles\_patient` 커널 내에서 전역 메모리 배열을 통해 물리 데이터를 읽어오던 로직을 `tex2D()` 함수를 사용하여 텍스처 메모리에서 데이터를 조회하도록 변경합니다.



\#### \*\*3. 실행 절차\*\*



1\.  \*\*`mqi\_physics\_data.hpp` 파일 생성\*\*:

&nbsp;     \* `physics` 디렉터리에 `mqi\_physics\_data.hpp` 파일을 생성하고 `physics\_data\_manager` 클래스를 정의합니다.

&nbsp;     \* 클래스에는 물리 데이터 로딩, CUDA 배열 생성, 텍스처 객체 생성 및 소멸을 위한 멤버 함수를 선언합니다.

2\.  \*\*`mqi\_physics\_data.cu` 파일 생성 및 구현\*\*:

&nbsp;     \* `physics\_data\_manager` 클래스의 멤버 함수를 구현합니다.

&nbsp;     \* \*\*데이터 로딩\*\*: 기존 방식대로 물리 데이터를 호스트 메모리로 로드합니다.

&nbsp;     \* \*\*텍스처 객체 생성\*\*:

&nbsp;         \* `cudaMallocArray()`를 사용하여 `cu\_array`를 할당합니다.

&nbsp;         \* `cudaMemcpyToArray()`를 사용하여 호스트의 물리 데이터를 `cu\_array`로 복사합니다.

&nbsp;         \* `cudaCreateTextureObject()`를 사용하여 `cu\_array`로부터 `tex\_object`를 생성하고, 선형 보간 및 주소 지정 모드 등의 속성을 설정합니다.

3\.  \*\*`mqi\_transport.hpp` 커널 수정\*\*:

&nbsp;     \* `transport\_particles\_patient` 커널의 인자로 `cudaTextureObject\_t` 타입의 텍스처 객체를 받도록 수정합니다.

&nbsp;     \* 커널 내부에서 물리 데이터 테이블에 접근하는 모든 코드를 `tex2D()` 함수 호출로 변경하여 텍스처 메모리를 통해 데이터를 가져오도록 합니다.

4\.  \*\*`mqi\_treatment\_session` 수정\*\*:

&nbsp;     \* 시뮬레이션 초기화 단계에서 `physics\_data\_manager` 객체를 생성하고, 물리 데이터를 로드 및 텍스처 객체로 변환하는 코드를 추가합니다.

&nbsp;     \* `transport\_particles\_patient` 커널 호출 시 생성된 텍스처 객체를 인자로 전달합니다.



-----



\### Phase 2: 적응형 입자 재정렬 실행 계획서



이 문서는 MOQUI 개발 계획의 \*\*Phase 2\*\*에 대한 상세 실행 계획을 기술합니다.



&nbsp; \* \*\*목표\*\*: Phase 1의 결과물에 \*\*적응형 입자 재정렬(Adaptive Particle Sorting)\*\* 알고리즘을 추가하여 스레드 분기(Thread Divergence) 및 비병합 메모리 접근(Non-coalesced Memory Access) 문제를 해결하고 GPU 활용률을 극대화합니다.

&nbsp; \* \*\*핵심 기술\*\*: \*\*Thrust 라이브러리를 이용한 병렬 정렬 (Parallel Sort)\*\*

&nbsp;     \* \*\*원리\*\*: 수송 커널 실행 직전, GPU 상의 전체 입자 배열을 에너지 순(내림차순)으로 정렬하여 한 워프(warp) 내 스레드들이 유사한 에너지의 입자를 처리하게 만듭니다.

&nbsp;     \* \*\*기대 효과\*\*:

&nbsp;       1.  \*\*스레드 분기 최소화\*\*: 유사 에너지 입자들의 물리적 상호작용 확률을 높여 실행 경로를 통일시키고 SIMT (Single Instruction, Multiple Threads) 아키텍처 활용도를 극대화합니다.

&nbsp;       2.  \*\*메모리 접근 패턴 개선\*\*: 정렬된 입자들이 물리 데이터 테이블의 인접 영역에 동시 접근하여 텍스처 캐시 히트율을 극대화합니다.



\#### \*\*1. 수정 대상 모듈 및 파일\*\*



| 파일 경로 | 클래스 / 함수 | 변경 내용 |

| :--- | :--- | :--- |

| `base/mqi\_treatment\_session.cpp` | `run\_simulation` | 단일 커널 호출을 `while` 루프로 변경 및 정렬 로직 추가 |

| `base/mqi\_track.hpp` | `track\_t` | 에너지 비교를 위한 `operator<` 또는 functor 추가 |

| `CMakeLists.txt` | - | Thrust 라이브러리 링크 설정 |



\#### \*\*2. 주요 변수 및 함수 변경 사항\*\*



&nbsp; \* \*\*`run\_simulation` 함수 내 주요 변경\*\*:

&nbsp;     \* 기존의 단일 `transport\_particles\_patient` 커널 호출을 `while` 루프로 감싸 입자가 모두 소멸될 때까지 반복하도록 구조를 변경합니다.

&nbsp;     \* `while` 루프의 시작 지점에 `thrust::sort`를 호출하는 코드를 추가하여 `d\_tracks`(입자 배열)를 에너지 기준으로 정렬합니다.

&nbsp; \* \*\*신규 변수 및 로직 ( `run\_simulation` 함수 내)\*\*:

&nbsp;     \* `divergence\_rate` (float): 워프 분기율을 저장할 변수입니다.

&nbsp;     \* `DIVERGENCE\_THRESHOLD` (const float): 정렬 수행 여부를 결정할 임계값입니다.

&nbsp;     \* \*\*프로파일링 코드\*\*: 워프 분기율(`divergence\_rate`)을 측정하는 간단한 코드를 추가합니다. 이 값은 `if (divergence\_rate > DIVERGENCE\_THRESHOLD)` 조건문에서 정렬 수행 여부를 판단하는 데 사용됩니다.

&nbsp; \* \*\*Functor 또는 연산자 오버로딩\*\*:

&nbsp;     \* `mqi\_track.hpp`에 `by\_energy`와 같은 functor 구조체나 `track\_t` 클래스 내에 `operator<`를 정의하여 Thrust가 입자를 에너지 기준으로 정렬할 수 있도록 합니다.



\#### \*\*3. 실행 절차\*\*



1\.  \*\*Thrust 라이브러리 설정\*\*:

&nbsp;     \* `CMakeLists.txt` 파일을 수정하여 CUDA Toolkit에 포함된 Thrust 라이브러리를 프로젝트에 링크합니다.

2\.  \*\*에너지 비교 로직 구현\*\*:

&nbsp;     \* `mqi\_track.hpp` 파일을 엽니다.

&nbsp;     \* `track\_t` 구조체 또는 클래스에 에너지 값을 비교하는 `by\_energy` functor를 아래와 같이 추가합니다.

&nbsp;       ```cpp

&nbsp;       struct by\_energy {

&nbsp;           \_\_host\_\_ \_\_device\_\_

&nbsp;           bool operator()(const track\_t\& a, const track\_t\& b) {

&nbsp;               return a.kinetic\_energy > b.kinetic\_energy; // 내림차순 정렬

&nbsp;           }

&nbsp;       };

&nbsp;       ```

3\.  \*\*`mqi\_treatment\_session.cpp` 수정\*\*:

&nbsp;     \* `run\_simulation` 함수의 내용을 수정합니다.

&nbsp;     \* `#include <thrust/sort.h>`와 `#include <thrust/execution\_policy.h>`를 추가합니다.

&nbsp;     \* 기존 커널 호출 부분을 아래와 같은 `while` 루프 구조로 변경합니다.

&nbsp;       ```cpp

&nbsp;       while (/\* 활성 입자 수 > 0 \*/) {

&nbsp;           // (선택) 워프 분기율 측정 로직 (cuProfiler\* API 등 활용)

&nbsp;           float divergence\_rate = get\_warp\_divergence\_rate();



&nbsp;           if (divergence\_rate > DIVERGENCE\_THRESHOLD) {

&nbsp;               thrust::sort(thrust::device, d\_tracks, d\_tracks + n\_particles, by\_energy());

&nbsp;           }



&nbsp;           transport\_particles\_patient<<<...>>>(...);



&nbsp;           // GPU에서 활성 입자 수 업데이트

&nbsp;       }

&nbsp;       ```

4\.  \*\*적응형 정렬 로직 구현\*\*:

&nbsp;     \* 워프 분기율을 측정하는 간단한 프로파일링 코드를 추가합니다. NVIDIA의 CUDA Profiling Tools Interface (CUPTI)를 사용하거나, 커널 내에서 `\_\_shfl\_sync`와 같은 워프 내장 함수를 사용하여 분기 발생 여부를 간단히 체크하고 그 결과를 전역 카운터에 원자적으로 더하는 방식으로 구현할 수 있습니다.

&nbsp;     \* 측정된 분기율이 사전에 설정된 임계치(`DIVERGENCE\_THRESHOLD`)를 초과할 경우에만 `thrust::sort`를 호출하도록 조건문을 추가하여 불필요한 정렬 오버헤드를 최소화합니다.



-----



\### Phase 3: `Event-by-Event` 커널 통합 실행 계획서



이 문서는 MOQUI 개발 계획의 \*\*Phase 3\*\*에 대한 상세 실행 계획을 기술합니다.



&nbsp; \* \*\*목표\*\*: Phase 1, 2에서 검증된 최적화 기법을 기반으로 고성능 \*\*`Event-by-Event` 수송 커널\*\*을 개발하고, 사용자가 시뮬레이션 모델을 선택할 수 있도록 기존 시스템에 통합합니다.

&nbsp; \* \*\*핵심 기술\*\*:

&nbsp;   1.  \*\*Woodcock 트래킹\*\*: 복셀 경계 계산을 최소화하는 알고리즘.

&nbsp;   2.  \*\*공유 메모리 기반 스택\*\*: 2차 입자 관리를 위한 고속 스택 구현.

&nbsp;   3.  \*\*워프 레벨 프리미티브\*\*: `\_\_shfl\_sync()` 등을 활용한 레지스터 간 직접 통신.

&nbsp;   4.  \*\*동적 병렬처리\*\*: 커널 내에서 자식 커널을 호출하여 작업 부하 분산.

&nbsp;   5.  \*\*지능형 에너지 컷오프\*\*: 물질에 따라 에너지 컷오프 값을 동적으로 적용.



\#### \*\*1. 수정 대상 모듈 및 파일\*\*



| 파일 경로 | 클래스 / 함수 | 변경 내용 |

| :--- | :--- | :--- |

| `kernel\_functions/mqi\_transport\_event.hpp` | `transport\_event\_by\_event\_kernel` | 신규 `Event-by-Event` 커널 작성 |

| `base/mqi\_treatment\_session.hpp` | `treatment\_session` | `transport\_model` enum 클래스 추가 |

| `base/mqi\_treatment\_session.cpp` | `run\_simulation` | `transport\_model` 설정에 따른 커널 분기 로직 구현 |

| `physics/mqi\_physics\_data.hpp` | `physics\_data\_manager` | Woodcock 트래킹을 위한 `max\_sigma` 변수 추가 |



\#### \*\*2. 주요 변수 및 함수 변경 사항\*\*



&nbsp; \* \*\*신규 커널\*\*:

&nbsp;     \* `transport\_event\_by\_event\_kernel`: `mqi\_transport\_event.hpp`에 새롭게 작성될 `Event-by-Event` 방식의 입자 수송 커널입니다.

&nbsp; \* \*\*신규 변수\*\*:

&nbsp;     \* `max\_sigma` (float): Woodcock 트래킹에 사용될 전체 물질의 최대 총 단면적. `physics\_data\_manager` 클래스에 추가됩니다.

&nbsp;     \* `\_\_shared\_\_ track\_t secondary\_stack\[256]`: 2차 입자 관리를 위해 커널 내에 선언될 공유 메모리 기반 스택.

&nbsp; \* \*\*`treatment\_session` 클래스 변경\*\*:

&nbsp;     \* `transport\_model` (enum class): `CONDENSED\_HISTORY`와 `EVENT\_BY\_EVENT` 두 가지 모드를 선택할 수 있는 열거형 클래스를 `mqi\_treatment\_session.hpp`에 추가합니다.

&nbsp; \* \*\*`run\_simulation` 함수 변경\*\*:

&nbsp;     \* JSON 설정 파일에서 `transport\_model` 값을 읽어와, 해당 값에 따라 기존 `transport\_particles\_patient` 커널 또는 새로 작성된 `transport\_event\_by\_event\_kernel`을 호출하도록 `if` 또는 `switch` 분기문을 추가합니다.



\#### \*\*3. 실행 절차\*\*



1\.  \*\*`max\_sigma` 계산 로직 추가\*\*:

&nbsp;     \* `physics\_data\_manager` 클래스에 모든 물질의 총 단면적(`total\_cross\_section`)을 순회하며 최대값(`max\_sigma`)을 찾아 저장하는 기능을 추가합니다. 이 값은 커널에 전달됩니다.

2\.  \*\*`mqi\_transport\_event.hpp` 신규 커널 작성\*\*:

&nbsp;     \* \*\*Woodcock 트래킹 구현\*\*:

&nbsp;         \* 입자를 `max\_sigma`를 기준으로 가상의 평균자유행로(MFP)만큼 이동시킵니다.

&nbsp;         \* 이동한 위치에서 `실제 단면적 / max\_sigma` 확률로 실제 상호작용 여부를 결정(rejection sampling)합니다.

&nbsp;         \* '가상 상호작용'일 경우 입자는 그대로 직진합니다.

&nbsp;     \* \*\*공유 메모리 스택 구현\*\*:

&nbsp;         \* `\_\_shared\_\_ track\_t secondary\_stack\[256];`와 같이 스레드 블록 내에 공유 메모리 스택을 선언합니다.

&nbsp;         \* 2차 입자 발생 시 이 스택에 `push`하고, 주 입자 처리가 끝나면 스택에서 `pop`하여 처리합니다.

&nbsp;     \* \*\*기타 고급 최적화 적용\*\*:

&nbsp;         \* 필요에 따라 워프 내 데이터 교환을 위해 `\_\_shfl\_sync()`와 같은 워프 레벨 프리미티브를 사용합니다.

&nbsp;         \* 2차 입자가 특정 임계치 이상 발생할 경우, CUDA 동적 병렬처리를 사용하여 자식 커널(`process\_secondaries<<<...>>>`)을 호출하는 로직을 검토 및 구현합니다.

&nbsp;         \* 물질(예: 뼈, 폐)에 따라 다른 에너지 컷오프 값을 적용하는 지능형 컷오프 로직을 구현합니다.

3\.  \*\*`mqi\_treatment\_session` 통합 작업\*\*:

&nbsp;     \* `mqi\_treatment\_session.hpp`에 `transport\_model` enum 클래스를 정의합니다.

&nbsp;     \* `mqi\_treatment\_session.cpp`의 `run\_simulation` 함수를 수정합니다.

&nbsp;     \* 시뮬레이션 시작 전, JSON 설정 파일로부터 `transport\_model` 값을 읽어옵니다.

&nbsp;     \* `run\_simulation` 함수 내 메인 `while` 루프에서, 읽어온 `transport\_model` 값에 따라 `transport\_particles\_patient`를 호출할지, `transport\_event\_by\_event\_kernel`을 호출할지 결정하는 분기문을 작성합니다.



이 계획들을 통해 각 단계별 목표를 명확히 하고, 코드 수정 범위를 한정하여 개발 과정의 안정성과 효율성을 높일 수 있습니다.

