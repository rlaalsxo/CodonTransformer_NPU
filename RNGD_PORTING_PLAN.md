# CodonTransformer RNGD NPU 포팅 구현 계획

## Context

CodonTransformer(BigBirdForMaskedLM 기반 코돈 최적화 모델)를 FuriosaAI RNGD NPU에서 실행 가능하도록 수정한다. `furiosa_llm`의 고수준 API는 BigBird를 지원하지 않으므로, `furiosa.torch`의 `torch.compile(backend="rngd")` 백엔드를 직접 사용한다.

**핵심 근거:**
- CodonTransformer는 추론 시 `original_full` 모드 = 표준 dense attention (sparse 아님)
- 모든 연산이 표준 PyTorch ops (커스텀 C++ 없음)
- FX 트레이싱 가능 (데이터 의존 제어 흐름 없음)
- `eager_fallback=True` 설정으로 미지원 연산은 자동 CPU fallback

---

## 수정 대상 파일

| 파일 | 수정 유형 | 내용 |
|------|----------|------|
| `CodonTransformer/CodonPrediction.py` | **수정 완료** | RNGD 디바이스/컴파일 지원 추가 |
| `infer.py` | **신규 생성 완료** | CLI 추론 스크립트 |
| `requirements_rngd.txt` | **신규 생성 완료** | RNGD 포함 패키지 의존성 |

---

## 구현 상세 (수정 완료)

### 1. furiosa.torch 조건부 import (CodonPrediction.py 상단)

```python
try:
    import furiosa.torch
    FURIOSA_AVAILABLE = True
except ImportError:
    FURIOSA_AVAILABLE = False
```

### 2. RNGD 헬퍼 함수 추가

```python
def _is_rngd_device(device: torch.device) -> bool:
    if not FURIOSA_AVAILABLE:
        return False
    return str(device).startswith("rngd")

def _compile_for_rngd(model: torch.nn.Module) -> torch.nn.Module:
    from furiosa.torch import backend
    from furiosa.native_torch import compiler

    compiler_config = compiler.Config(
        implicit_type_casting=True,
        implicit_type_casting_all_activation=True,
        tactic_hint=compiler.TacticHintConfig.Default,
        max_num_unique_shapes=500,
        tactic_sorting_policy=compiler.TacticSortingPolicy.ByEstimation,
    )

    rngd_backend = backend.backend_with(
        compiler_config=compiler_config,
        eager_fallback=True,
    )

    return torch.compile(model, backend=rngd_backend, dynamic=False)
```

### 3. predict_dna_sequence() 수정

- RNGD일 때 `model.to(device)` 스킵 (CPU에 유지)
- `_compile_for_rngd(model)` 호출
- 입력 텐서 dict comprehension으로 RNGD 디바이스 이동

### 4. load_model() 수정

- RNGD일 때 `.to(device)` 스킵

---

## CLI 사용법 (infer.py)

```bash
# CPU 실행
python infer.py --protein "MKTVRQER..." --organism "Homo sapiens"

# RNGD NPU 실행
python infer.py --protein "MKTVRQER..." --device rngd:0

# 결과 파일 저장
python infer.py --protein "MKTVRQER..." --output results/output.json
python infer.py --protein "MKTVRQER..." --output results/output.fasta

# 비결정적 다중 서열 생성
python infer.py --protein "MKTVRQER..." --no-deterministic --num-sequences 5

# 지원 생물종 목록
python infer.py --list-organisms
```

---

## 패키지 설치

```bash
# 시스템 의존성 + Python 패키지 (RNGD 서버)
sudo apt-get update && sudo apt-get install -y pkg-config libsentencepiece-dev && pip install --upgrade pip && pip install -r requirements_rngd.txt
```

**주의:** `furiosa-native-torch`는 Python 3.11 이상 필요.

---

## 검증 방법

1. **기능 검증**: 동일 입력에 대해 CPU vs RNGD 출력 비교 (deterministic 모드)
2. **커버리지 확인**:
   ```python
   from furiosa.torch.debug import RNGDCoverageTrace
   trace = RNGDCoverageTrace("codon_test")
   with trace:
       result = predict_dna_sequence(...)
   print(trace.statistics())
   ```
3. **에러 없이 동작**: `eager_fallback=True`로 미지원 연산 CPU fallback 확인
4. **furiosa 미설치 환경**: `FURIOSA_AVAILABLE=False`일 때 기존 동작 유지 확인

---

## AntiFold RNGD 호환성 분석

**결론: ❌ 부적합**

- GVP 인코더가 `torch_geometric.nn.MessagePassing` + `torch_scatter` 사용
- FX 트레이싱 불가
- Transformer 부분(16 layers)은 표준 ops지만 GVP와 분리 불가
