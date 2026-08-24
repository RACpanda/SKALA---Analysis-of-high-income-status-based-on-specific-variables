# Adult Income Explorer

SKALA 교육 과정에서 진행한 **Adult Census Income 기반 데이터 분석 및 머신러닝 프로젝트**입니다.

사용자가 직접 변수를 선택해 **연 소득 50,000달러 초과 여부와의 관계를 탐색**하고, 개인의 조건을 입력해 **연 소득 50,000달러 초과 확률을 예측**할 수 있는 Streamlit 웹 애플리케이션입니다.

단순한 통계 결과 출력에 그치지 않고 일반 사용자도 분석 결과를 이해할 수 있도록 입력, 분석, 예측, 결과 설명 과정을 하나의 웹 UX로 구성했습니다.

현재 버전인 **v1.5**에서는 새로운 분석 기능을 추가하기보다, v1.4에서 안정화하고 배포한 서비스를 기준으로 **코드 구조 정리, 중복 제거, dependency 정리, 테스트 강화 및 유지보수성 개선**을 수행했습니다.

---

## 1. 주요 기능

서비스는 크게 세 영역으로 구성됩니다.

### 1.1 소득과의 관계

사용자가 궁금한 변수 하나를 선택해 연 소득 50,000달러 초과 여부와 어떤 관계가 있는지 확인할 수 있습니다.

필요한 경우 다른 변수를 **함께 고려할 항목**으로 선택하여 분석할 수 있습니다.

지원하는 변수 유형은 다음과 같습니다.

- 연속형
- 이진형
- 범주형

분석 결과는 다음 흐름으로 제공합니다.

1. 데이터에서 보이는 관계
2. 다른 조건을 함께 고려한 결과
3. 모델이 계산한 예상 비율
4. 조건이 충족되는 경우 PSM 추가 분석

전문적인 통계 수치를 먼저 보여주기보다 사용자가 이해할 수 있는 해석을 우선 제공하고, p-value 등 상세 통계 정보는 필요한 경우 확인할 수 있도록 구성했습니다.

---

### 1.2 내 소득 예측

사용자가 자신의 조건을 입력하면 학습된 머신러닝 모델이 **연 소득 50,000달러를 초과할 확률**을 계산합니다.

주요 입력 변수는 다음과 같습니다.

- 나이
- 성별
- 인종
- 혼인 상태
- 가구 내 관계
- 출신 국가
- 교육 수준
- 직업
- 고용 형태
- 주당 근무시간
- 투자·자산 이익
- 투자·자산 손실

범주형 값은 사용자가 이해하기 쉽도록 다음과 같이 표시합니다.

- 남성 (Male)
- 민간 기업 (Private)
- 학사 (Bachelors)
- 전문직 (Prof-specialty)

화면에서는 사용자용 한글 표현을 제공하지만 모델에는 학습 데이터에서 사용한 원본 범주값을 전달합니다.

---

### 1.3 문의하기

서비스 이용 중 발생하는 질문, 오류, 불편사항 및 개선 의견을 제출할 수 있습니다.

지원하는 문의 유형은 다음과 같습니다.

- 이용 방법 문의
- 분석 결과 문의
- 예측 결과 문의
- 오류 신고
- 불편 사항
- 기능 개선 요청
- 기타

문의 제출 시 다음 정보가 자동 기록됩니다.

- 문의 ID
- 접수 시각
- 서비스 버전
- 현재 페이지
- 문의 유형
- 문의 내용
- 선택 입력 이메일
- 문의 상태

배포 환경에서는 문의 데이터를 **Supabase PostgreSQL**에 저장하여 애플리케이션 재시작이나 재배포 이후에도 문의 기록이 유지되도록 구성했습니다.

---

# 2. 소득과의 관계 분석

## 2.1 데이터에서 보이는 관계

다른 조건을 별도로 고려하기 전에 관심 변수와 연 소득 50,000달러 초과 여부 사이에서 관찰되는 관계를 확인합니다.

### 연속형 변수

Point-biserial correlation을 이용해 연속형 변수와 이진형 소득 변수 사이의 관계를 확인합니다.

예시:

> 나이 값이 큰 쪽에서 연 소득 5만 달러를 넘는 비율도 전반적으로 높은 방향의 관계가 나타났습니다.

### 이진형 변수

두 그룹에서 연 소득 50,000달러를 넘는 비율을 직접 비교합니다.

### 범주형 변수

각 범주의 고소득 비율을 비교하고 Chi-square 검정을 수행합니다.

사용자 화면에서는 주요 차이를 먼저 보여주며 전문적인 통계 검정 결과는 상세 정보에서 확인할 수 있습니다.

---

## 2.2 다른 조건을 함께 고려한 결과

관심 변수와 소득 사이의 관계를 Logistic Regression 기반으로 다시 확인합니다.

사용자가 함께 고려할 항목을 선택하면 해당 변수들을 모델에 함께 포함하여 관계를 분석합니다.

통제 변수를 선택하지 않은 경우에는 별도의 조정 결과라는 표현 대신 **통계 모델로 다시 확인한 결과**로 구분합니다.

v1.1부터 일반 Logistic Regression이 안정적으로 적합되지 않는 경우 **Binomial GLM fallback**을 사용할 수 있도록 분석 안정성을 강화했습니다.

범주형 관심 변수에서는 개별 범주의 결과뿐 아니라 변수 전체의 관계를 확인하기 위한 **Joint Wald Test**도 지원합니다.

표본이 매우 적거나 결과가 한쪽으로 치우친 범주는 안정적인 추정이 어려울 수 있습니다.

이 경우 내부 통계 경고나 더미 변수명을 그대로 노출하지 않고, 사용자가 이해할 수 있는 안내 문구를 제공합니다.

---

## 2.3 예상 비율

Logistic Regression 결과를 사용자가 쉽게 해석할 수 있도록 관심 변수 값에 따른 예상 고소득 비율을 제공합니다.

통제 변수가 선택된 경우 다른 변수의 관측값은 유지하면서 관심 변수만 동일한 값으로 설정해 평균 예측확률을 계산합니다.

사용자 화면에서는 다음과 같이 구분합니다.

- 통제 변수 있음 → **조건을 고려한 예상 비율**
- 통제 변수 없음 → **모델이 계산한 예상 비율**

---

## 2.4 PSM

이진형 관심 변수와 통제 변수가 선택된 경우 선택적으로 **Propensity Score Matching**을 수행할 수 있습니다.

PSM은 관측된 조건이 비슷한 대상을 매칭해 두 그룹을 추가로 비교하기 위한 분석입니다.

다만 데이터에 포함된 변수만 고려할 수 있으므로 결과를 확정적인 인과효과로 해석하지 않습니다.

매칭 전후 공변량 균형은 Standardized Mean Difference를 이용해 확인합니다.

---

# 3. 예측 모델

예측 모델은 `HistGradientBoostingClassifier`를 기반으로 구성했습니다.

최종 모델 파라미터는 다음과 같습니다.

```python
MODEL_PARAMS = {
    "learning_rate": 0.14447746112718687,
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": 0.45606998421703593,
}
```

v1.2에서 현재 12개 입력 변수를 기준으로 재튜닝을 수행했으나 기존 모델 대비 ROC-AUC 개선 폭이 약 `0.00044`로 매우 작아 기존 파라미터를 유지했습니다.

v1.4의 Python 및 scikit-learn 환경 전환에서도 알고리즘과 하이퍼파라미터를 변경하지 않고 동일 설정으로 모델을 다시 생성하여 기존 성능이 재현되는지 확인했습니다.

---

# 4. 확률 보정

모델이 출력하는 확률의 신뢰성을 높이기 위해 calibration을 적용했습니다.

비교한 방법은 다음과 같습니다.

- Uncalibrated
- Sigmoid
- Isotonic

OOF 검증 결과 Sigmoid calibration이 Brier Score와 ROC-AUC를 유지하면서 Log Loss 측면에서도 안정적인 결과를 보여 최종 방식으로 선택했습니다.

최종 예측 구조는 다음과 같습니다.

```text
HistGradientBoosting
        ↓
Sigmoid Calibration
        ↓
연 소득 50,000달러 초과 확률
```

---

# 5. 분류 Threshold

분류 threshold에 대해서도 OOF 검증을 수행했습니다.

검토한 기준은 다음과 같습니다.

- 기본 threshold 0.50
- F1 최적 threshold
- Balanced Accuracy 최적 threshold
- Precision과 Recall이 가장 가까운 threshold

False Positive 또는 False Negative에 대한 별도의 업무 비용이 정의되지 않았기 때문에 특정 지표만 임의로 최대화하지 않고 기본 threshold `0.50`을 유지했습니다.

---

# 6. 최종 모델 성능

Held-out test set 기준 최종 결과입니다.

| Metric | Result |
| --- | ---: |
| Test rows | 6,508 |
| Accuracy | 0.8791 |
| Precision | 0.7925 |
| Recall | 0.6747 |
| F1 | 0.7289 |
| ROC-AUC | 0.9334 |
| Brier Score | 0.0848 |
| Log Loss | 0.2704 |
| Actual Positive Rate | 0.2409 |
| Mean Predicted Probability | 0.2439 |
| Probability Bias | +0.0029 |

Calibration 이후 평균 예측확률은 실제 positive rate와 매우 가까운 수준으로 나타났습니다.

다만 Precision은 높고 Recall은 상대적으로 낮으므로 calibration이 모든 분류 성능 지표를 동시에 향상시켰다고 해석하지 않습니다.

---

# 7. 예측 결과 설명

## 7.1 내 입력값 살펴보기

각 입력값을 학습 데이터의 대표적인 값으로 하나씩 변경했을 때 예측 확률이 얼마나 달라지는지 확인합니다.

이 기능은 특정 사용자의 현재 입력 조건을 이해하기 위한 **모델 기반 비교 설명**입니다.

각 변수는 하나씩 독립적으로 변경되므로 결과를 서로 더할 수 없으며 인과효과를 의미하지 않습니다.

---

## 7.2 모델이 많이 참고한 정보

전체 테스트 데이터를 기준으로 Permutation Importance를 계산합니다.

각 변수를 섞었을 때 ROC-AUC가 얼마나 감소하는지를 기준으로 모델이 해당 정보를 예측에 얼마나 활용하는지 확인합니다.

이 값은 다음을 의미하지 않습니다.

- 특정 개인의 예측에 미친 영향의 크기
- 실제 소득에 미치는 직접적인 영향
- 인과효과

---

## 7.3 What-if

다른 입력 조건은 유지한 채 하나의 변수만 변경하면서 모델의 예측 확률이 어떻게 달라지는지 확인할 수 있습니다.

예:

- 나이만 바꾸면 예측 확률은 어떻게 달라지는가?
- 교육 수준만 바꾸면 예측 확률은 어떻게 달라지는가?
- 고용 형태만 바꾸면 예측 확률은 어떻게 달라지는가?

What-if 결과는 모델 내부의 예측 변화를 보여주는 기능이며 실제 조건을 변경했을 때 소득이 동일한 방식으로 변한다는 의미는 아닙니다.

---

# 8. 사용자 인터페이스

서비스는 일반 사용자가 분석 전문용어를 몰라도 사용할 수 있도록 구성했습니다.

예를 들어 다음과 같이 표현을 단순화했습니다.

```text
관심 변수 → 궁금한 항목
통제 변수 → 함께 고려할 항목
분석 실행 → 결과 보기
예측 실행 → 내 결과 확인하기
```

범주형 값은 원본 데이터의 의미를 유지하면서 사용자 화면에서는 `한글 (원본값)` 형식으로 제공합니다.

근거 없이 특정 국가나 범주로 단정하기 어려운 원본 값은 임의로 번역하지 않습니다.

---

# 9. 결과 상태 관리

Streamlit widget 값과 실제 분석 결과를 분리하여 관리합니다.

사용자가 입력값을 변경하더라도 기존 결과는 즉시 변경되지 않습니다.

```text
입력 변경
   ↓
기존 결과 유지
   ↓
결과 보기 / 내 결과 확인하기
   ↓
새 결과로 갱신
```

새 개인 예측이 실행된 경우 이전 개인 입력을 기준으로 생성된 What-if 결과는 초기화합니다.

이를 통해 화면의 입력값과 실제 결과 생성 시점이 혼동되지 않도록 구성했습니다.

---

# 10. 오류 처리

사용자 화면과 개발자 로그를 분리했습니다.

```text
사용자 화면
→ 이해할 수 있는 한국어 안내

개발자 로그
→ 실제 예외 및 traceback
```

사용자 화면에는 다음과 같은 내부 정보가 직접 노출되지 않도록 관리합니다.

- Python traceback
- 로컬 파일 경로
- 라이브러리 내부 객체명
- API 응답 전문
- Secret 및 환경변수

반면 개발 로그에서는 `logger.exception()` 등을 통해 실제 원인을 확인할 수 있습니다.

---

# 11. 문의 저장 구조

문의 기능은 UI와 저장 로직을 분리했습니다.

```text
src/ui_inquiry.py
        ↓
src/inquiry.py
        ↓
Supabase Data API
        ↓
PostgreSQL
```

문의 저장은 `supabase-py` SDK를 사용하지 않고 `httpx`를 통한 REST 요청으로 처리합니다.

배포 환경:

```text
Streamlit
    ↓
HTTPS
    ↓
Supabase REST Data API
    ↓
public.user_inquiries
```

로컬 개발 환경에서는 필요할 경우 CSV fallback을 사용할 수 있습니다.

---

# 12. 문의 데이터 보안

Supabase 문의 테이블은 Row Level Security를 이용한 최소 권한 구조를 사용합니다.

기본 권한 방향은 다음과 같습니다.

```text
anon INSERT  O
anon SELECT  X
anon UPDATE  X
anon DELETE  X
```

웹 애플리케이션에는 Publishable key를 사용합니다.

RLS를 우회할 수 있는 고권한 Secret 또는 Service Role 계열 키는 웹 애플리케이션 코드에 사용하지 않습니다.

로컬 Secret은 다음 파일에서 관리합니다.

```text
.streamlit/secrets.toml
```

이 파일은 Git 추적 대상에서 제외합니다.

배포 환경에서는 Streamlit Cloud의 Secrets 기능을 이용합니다.

---

# 13. v1.5 코드 구조 개선

v1.5에서는 기존 기능을 유지하면서 코드베이스의 책임을 명확하게 분리했습니다.

기존에는 `app.py`가 UI, 분석 결과 표시, 예측 화면, 문의 화면 등을 대부분 담당하면서 파일 크기가 크게 증가했습니다.

v1.5에서는 다음 구조로 정리했습니다.

```text
app.py
│
├─ Streamlit 기본 설정
├─ 공통 스타일
├─ 서비스 데이터 캐시
├─ Hero
├─ Navigation
└─ 앱 진입점

src/ui_association.py
│
├─ 관계 분석 입력
├─ 관찰 관계 결과
├─ 조정 결과
├─ 예상 비율
└─ PSM UI

src/ui_prediction.py
│
├─ 예측 입력
├─ 개인 예측 결과
├─ 개인 입력 설명
├─ Global Importance
└─ What-if UI

src/ui_inquiry.py
└─ 문의 Streamlit UI

src/ui_common.py
└─ 공통 사용자 표시 helper

src/inquiry.py
├─ 문의 입력 검증
├─ 문의 record 생성
├─ CSV 저장
└─ Supabase 저장
```

분석 및 모델 계산 자체는 기존 `association.py`, `statistics.py`, `modeling.py` 등에 유지하여 UI와 계산 책임을 분리했습니다.

---

# 14. 사용자 표시 라벨 중앙화

v1.5 이전에는 `app.py`와 `visualization.py`가 각각 변수명 및 범주값 번역 정보를 가지고 있었습니다.

이 구조에서는 한쪽만 수정할 경우 화면과 그래프의 표현이 달라질 수 있습니다.

v1.5에서는 다음 파일로 통합했습니다.

```text
src/labels.py
```

여기에서 다음 정보를 중앙 관리합니다.

- 변수명
- 범주값
- 변수 유형 표시명

`app.py`, Streamlit UI 모듈, Plotly 시각화는 모두 동일한 라벨 정보를 참조합니다.

---

# 15. 시각화 코드 정리

실제 웹 UI에서 사용하지 않는 과거 Plotly 그래프 생성 코드를 제거했습니다.

제거 대상은 사용자 화면에서 더 이상 소비되지 않는 시각화 계층으로 한정했습니다.

통계 계산 자체나 Logistic Regression 결과는 제거하지 않았습니다.

현재 관계 분석에서 실제 사용하는 핵심 시각화는 다음과 같습니다.

- 관심 변수 값별 예상 고소득 비율
- PSM 공변량 균형
- 개인 예측 설명
- Global Permutation Importance
- What-if Simulation

---

# 16. 실행 환경

v1.4 이후 현재 환경은 Python 3.13 계열을 기준으로 관리합니다.

주요 검증 환경:

```text
Python        3.13.15
pandas        3.0.5
NumPy         2.4.6
SciPy         1.18.1
scikit-learn  1.9.0
statsmodels   0.14.6
Matplotlib    3.11.1
Plotly        6.9.0
joblib        1.5.3
Streamlit     1.62.0
```

NumPy는 최신 버전 자체를 무조건 사용하는 대신 현재 joblib 및 전체 테스트와의 호환성을 기준으로 검증한 버전을 사용합니다.

---

# 17. Dependency 관리

production dependency와 개발용 dependency를 분리했습니다.

```text
requirements.txt
→ 서비스 실행에 필요한 패키지

requirements-dev.txt
→ requirements.txt + pytest
```

현재 production에서 사용하지 않는 과거 dependency는 제거했습니다.

예:

- Polars
- Seaborn
- supabase-py
- postgrest

Supabase 연결은 `httpx` 기반 REST 요청으로 처리합니다.

---

# 18. 프로젝트 구조

```text
.
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── ML_MODELING_SUMMARY.md
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── inquiries/
│
├── scripts/
│   ├── audit_v1_5.py
│   │
│   └── experiments/
│       ├── model_calibration_v1_2.py
│       ├── model_threshold_v1_2.py
│       └── model_tuning_v1_2.py
│
├── src/
│   ├── __init__.py
│   ├── association.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── inquiry.py
│   ├── labels.py
│   ├── modeling.py
│   ├── model_visualization.py
│   ├── statistics.py
│   ├── ui_association.py
│   ├── ui_common.py
│   ├── ui_inquiry.py
│   ├── ui_prediction.py
│   └── visualization.py
│
└── tests/
    ├── conftest.py
    ├── test_association.py
    ├── test_config.py
    ├── test_data.py
    ├── test_error_handling.py
    ├── test_inquiry.py
    ├── test_labels.py
    ├── test_model_visualization.py
    ├── test_modeling.py
    ├── test_statistics.py
    ├── test_ui_common.py
    ├── test_ui_structure.py
    └── test_visualization.py
```

과거 모델 튜닝, calibration, threshold 검증 스크립트는 단순 백업이 아니라 현재 모델 설정의 선택 근거를 재현하기 위한 실험 코드이므로 유지합니다.

---

# 19. 설치 방법

Python 3.13 환경을 기준으로 가상환경을 생성합니다.

```bash
python3.13 -m venv .venv-v1.5
source .venv-v1.5/bin/activate
```

production dependency:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

개발 및 테스트 환경:

```bash
python -m pip install -r requirements-dev.txt
```

dependency 상태 확인:

```bash
python -m pip check
```

---

# 20. 실행 방법

Adult Census Income 데이터를 다음 경로에 준비합니다.

```text
data/raw/adult.csv
```

Streamlit 앱을 실행합니다.

```bash
python -m streamlit run app.py
```

가상환경의 Python과 Streamlit을 명확하게 연결하기 위해 `streamlit run app.py`보다 `python -m streamlit run app.py` 방식을 권장합니다.

---

# 21. 문의 저장 설정

로컬 개발에서는 CSV fallback을 사용할 수 있습니다.

```text
outputs/inquiries/user_inquiries.csv
```

배포 환경에서는 Streamlit Secrets에 Supabase 설정을 등록합니다.

```toml
INQUIRY_STORAGE_MODE = "supabase"

SUPABASE_URL = "..."
SUPABASE_KEY = "sb_publishable_..."
```

실제 Secret 값이나 프로젝트별 설정값은 Git 저장소에 포함하지 않습니다.

---

# 22. 테스트

기본 테스트:

```bash
python -m pytest -q
```

warning까지 오류로 처리하는 엄격한 검증:

```bash
python -m pytest -q -W error
```

v1.5에서는 다음 영역을 테스트합니다.

### 데이터

- Adult 데이터 정제
- 입력 컬럼 계약
- 분석 대상 변수

### 연관성 분석

- 연속형
- 이진형
- 범주형
- 통제 변수 없음
- 단일 통제 변수
- 복수 통제 변수
- PSM
- 예외 상황

### 예측

- 모델 학습
- 모델 bundle
- 개인 예측
- 예측 설명
- Global Importance
- 모델 시각화

### 사용자 UI

- 공통 라벨
- 사용자 표시 helper
- UI 모듈 구조
- 내부 오류 미노출
- 서버 로그 기록

### 문의

- 입력 검증
- 이메일 선택 입력
- CSV 저장
- Supabase Data API 요청
- HTTP 오류
- 네트워크 오류
- Secret 누락
- email 미입력 시 DB NULL 처리

---

# 23. v1.5 Final Audit

v1.5에서는 별도의 코드베이스 Audit 스크립트를 추가했습니다.

```bash
python scripts/audit_v1_5.py
```

주요 검사 항목은 다음과 같습니다.

- Python AST 파싱
- 사용하지 않는 import 후보
- 참조되지 않는 production 정의 후보
- 동일 이름 정의
- 중복 상수 후보
- requirements와 실제 import 비교
- 과거 API 및 파일 참조
- 라벨 상수 중복
- 프로젝트 파일 구조
- Git Secret 추적 여부

정적 분석에서 동일한 이름이나 값을 발견하더라도 자동 삭제하지 않습니다.

예를 들어 서로 다른 모듈에서 사용하는 private helper 또는 의미가 다른 `0.1` 기준값은 이름이나 값만 같다는 이유로 공통화하지 않습니다.

---

# 24. 모델 공정성 진단

모델은 전체 성능뿐 아니라 일부 집단별 성능 차이도 진단합니다.

다만 집단별 Recall 차이나 최소 Recall 기준은 **모델을 점검하기 위한 진단 정보**이며, 해당 값만으로 특정 집단에 대한 차별이나 공정성을 확정하지 않습니다.

다음 요소를 함께 고려해야 합니다.

- 집단별 표본 수
- positive rate
- 데이터 분포
- 변수 구성
- 모델 구조
- 평가 지표 선택

공정성 관련 결과는 모델의 한계를 확인하기 위한 보조 진단으로 해석합니다.

---

# 25. 결과 해석 시 주의사항

이 서비스가 제공하는 통계 분석은 **데이터에서 변수들이 함께 나타나는 관계**를 보여줍니다.

다른 조건을 통계적으로 함께 고려하더라도 특정 변수가 소득 차이의 직접적인 원인이라고 단정할 수 없습니다.

PSM 역시 관측된 변수만 조정할 수 있으며 데이터에 포함되지 않은 요인의 차이는 통제할 수 없습니다.

예측 기능은 학습된 머신러닝 모델이 입력 조건을 바탕으로 계산한 확률이며 실제 개인의 소득을 보장하거나 확정하는 결과가 아닙니다.

개인 입력 설명과 What-if 역시 모델의 예측 변화를 확인하기 위한 기능이며 인과관계를 의미하지 않습니다.

---

# 26. Version History

## v1.0 — Foundation

- 변수 선택형 연관성 분석
- 연속형 / 이진형 / 범주형 분석
- Logistic Regression 기반 분석
- 선택적 PSM
- HistGradientBoosting 기반 소득 예측
- 개인 예측 설명
- What-if Simulation
- Permutation Importance
- Plotly + Streamlit 웹 UI

---

## v1.1 — Association Analysis Upgrade

- Logistic Regression 안정성 강화
- Standard Logit 실패 시 Binomial GLM fallback
- 범주형 관심 변수 Overall Joint Wald Test
- Adjusted Probability 추가

---

## v1.2 — Prediction Reliability Upgrade

- 현재 12개 입력 변수 기준 모델 재튜닝 및 검증
- 기존 `MODEL_PARAMS` 유지
- OOF 기반 calibration 비교
- Sigmoid calibration 적용
- threshold 후보 비교
- classification threshold 0.50 유지
- calibrated probability 기반 예측 / 설명 / What-if 통일

---

## v1.3 — GUI / UX Refinement

- 메인 Hero 및 서비스 명칭 개편
- 브라우저 UI 정리
- 페이지 navigation 안정화
- 입력 화면 전문용어 단순화
- 범주형 선택값 `한글 (영어)` 지원
- 관계 분석 결과 UX 재구성
- 예측 결과 UX 재구성
- 결과가 실행 시점에만 갱신되도록 상태 관리
- Odds Ratio 중심 사용자 UI 제거
- 개인 예측 설명 UX 개선
- Plotly 변수명 및 범주값 한글화
- 그래프 스타일 통일
- 결과 섹션 간 여백 및 정보 밀도 조정

---

## v1.4 — Service Stabilization & User Support

- Python 3.13 기반 실행환경 최신화
- 주요 라이브러리 호환성 검증
- NumPy / joblib 호환성 검증
- scikit-learn 최신 환경에서 모델 bundle 재생성
- 기존 모델 성능 재현 확인
- production API 기준 회귀 테스트 재구성
- warning-as-error 기반 호환성 검증
- 사용자 오류 메시지와 개발자 로그 분리
- 희소 범주 분석 경고 UX 개선
- 사용자 문의하기 기능 추가
- Supabase 기반 문의 데이터 영속 저장
- Publishable key + RLS 기반 최소 권한 구성
- Streamlit 웹 배포

---

## v1.5 — Codebase Cleanup & Maintainability

- 공통 변수·범주 라벨을 `src/labels.py`로 중앙화
- 근거가 불명확한 범주값 임의 번역 제거
- 사용하지 않는 과거 시각화 코드 제거
- `APP_VERSION`을 `src/config.py`로 중앙화
- 문의 저장 설정 및 경로 정리
- Supabase REST 저장 테스트 강화
- CSV / Supabase 문의 저장 계약 테스트 추가
- 공통 Streamlit helper를 `src/ui_common.py`로 분리
- 문의 UI를 `src/ui_inquiry.py`로 분리
- 관계 분석 UI를 `src/ui_association.py`로 분리
- 개인 예측 UI를 `src/ui_prediction.py`로 분리
- `app.py`를 앱 진입점 및 navigation 중심으로 단순화
- UI 오류 처리 테스트를 모듈 구조에 맞게 확장
- 사용하지 않는 dependency 제거
- production / development requirements 분리
- 과거 임시 파일, cache 및 stale 문서 정리
- 모델 문서의 오래된 bundle 파일명 수정
- 전체 repository Audit 스크립트 추가
- requirements와 실제 import 불일치 검사
- dead code 및 중복 라벨 검사
- Git Secret 추적 여부 검사
- 기존 통계 분석 및 머신러닝 모델 동작 유지

---

# 27. 현재 상태

**v1.5 Codebase Cleanup & Maintainability 완료**

v1.5에서는 새로운 분석 알고리즘이나 모델을 추가하지 않고 v1.4에서 배포한 서비스의 동작을 유지하면서 내부 구조를 정리했습니다.

최종 구조는 다음 원칙을 따릅니다.

```text
계산 로직
→ association / statistics / modeling

사용자 UI
→ ui_association / ui_prediction / ui_inquiry

공통 표시
→ labels / ui_common

서비스 저장
→ inquiry

앱 진입점
→ app.py
```

모델 알고리즘, 기존 `MODEL_PARAMS`, Sigmoid Calibration, Threshold 0.50 및 핵심 통계 분석 방식은 유지했습니다.

v1.5의 목표는 새로운 기능 추가가 아니라 **현재 서비스를 더 단순하고 검증 가능하며 이후 수정하기 쉬운 코드베이스로 만드는 것**입니다.