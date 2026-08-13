# Adult Income Explorer

Adult Census Income 데이터를 기반으로 사용자가 직접 변수를 선택하여  
**고소득 여부와의 연관성을 분석하고, 개별 조건에 대한 고소득 확률을 예측하는 웹 기반 분석·예측 도구**입니다.

SKALA 과정에서 학습한 데이터 처리, 통계 분석, 머신러닝, 모델 평가, 시각화, 웹 서비스 구현을 하나의 End-to-End 프로젝트로 통합하는 것을 목표로 합니다.

---

## 1. 프로젝트 소개

Adult Census Income 데이터는 개인의 나이, 교육 수준, 직업, 근무시간 등의 특성과 함께 연 소득이 50,000달러를 초과하는지 여부를 제공합니다.

본 프로젝트는 기존의 특정 변수 중심 정적 분석에서 확장하여, 사용자가 직접 분석 조건을 선택할 수 있는 두 가지 기능을 제공합니다.

### 연관성 분석

사용자가

- 관심 변수(Exposure) 1개
- 통제 변수(Control) 0개 이상

을 직접 선택하여 `high_income`과의 관계를 탐색합니다.

분석 결과는 관심 변수 유형에 따라 자동으로 달라집니다.

- 연속형 변수: Point-biserial correlation
- 이진 변수: 집단별 고소득률, Risk Ratio, Odds Ratio, Fisher exact test
- 범주형 변수: 범주별 고소득률, Chi-square test
- 조정 후 분석: Logistic Regression 기반 Adjusted Odds Ratio
- 선택적 추가 분석: Propensity Score Matching(PSM)

### 고소득 예측

사용자가 한 개인의 실제 조건을 입력하면 학습된 머신러닝 모델이

- 연 소득 50,000달러 초과 예측 여부
- 고소득 예측 확률
- 개인 입력 기준 예측 설명
- 전체 모델 Feature Importance
- 특정 입력값 변화에 따른 What-if 예측

을 제공합니다.

연관성 분석과 머신러닝 예측은 서로 다른 질문을 다룹니다.

> 연관성 분석은 "어떤 변수가 고소득 여부와 어떻게 연결되어 있는가?"를 탐색하고,  
> 예측 기능은 "현재 입력 조건에서 모델이 고소득일 확률을 얼마나 예측하는가?"를 다룹니다.

---

## 2. 주요 기능

### 2.1 사용자 선택형 연관성 분석

관심 변수로 사용할 수 있는 변수는 다음과 같습니다.

| 변수 | 유형 |
|---|---|
| age | Continuous |
| workclass | Categorical |
| education | Categorical |
| marital-status | Categorical |
| occupation | Categorical |
| relationship | Categorical |
| race | Categorical |
| sex | Binary |
| capital-gain | Continuous |
| capital-loss | Continuous |
| hours-per-week | Continuous |
| native-country | Categorical |

`fnlwgt`는 Census 표본 가중치이므로 사용자 분석 변수에서 제외했습니다.

`education-num`은 `education`의 숫자형 표현으로 정보가 중복되므로 제외했습니다.

`income`, `high_income`은 결과변수이므로 설명변수로 사용할 수 없습니다.

---

### 2.2 조정 전 연관성 분석

관심 변수 유형에 맞춰 자동으로 분석 방식을 선택합니다.

#### Continuous

예:

- age
- capital-gain
- capital-loss
- hours-per-week

Point-biserial correlation을 이용해 연속형 변수와 `high_income` 사이의 조정 전 연관성을 확인합니다.

시각화에서는 동일한 분석 표본을 분위 구간으로 나누어 구간별 실제 고소득률을 표시합니다.

#### Binary

현재 기본 이진 변수:

- sex

두 집단의

- 표본 수
- 고소득률
- 고소득률 차이
- Risk Ratio
- Odds Ratio
- 95% Confidence Interval
- Fisher exact p-value
- Cohen's h

를 계산합니다.

#### Categorical

예:

- education
- occupation
- workclass
- race
- native-country

각 범주의 고소득률을 비교하고 Chi-square test를 수행합니다.

---

### 2.3 통제 변수 조정

사용자가 선택한 통제 변수는 특정 값으로 고정되는 변수가 아닙니다.

예를 들어

```text
관심 변수: education
통제 변수: age, sex, race
```

를 선택하면 다음 질문을 분석합니다.

> 나이, 성별, 인종을 통계적으로 고려한 뒤에도 교육 수준과 고소득 여부 사이의 연관성이 나타나는가?

조정 분석은 Logistic Regression을 사용하며 관심 변수에 대해

- Adjusted Odds Ratio
- 95% Confidence Interval
- p-value

를 제공합니다.

범주형 변수에서는 하나의 기준 범주를 두고 다른 범주와 비교합니다.

---

### 2.4 Propensity Score Matching

현재 PSM은 다음 조건에서 선택적으로 사용할 수 있습니다.

- 관심 변수가 Binary
- 통제 변수가 1개 이상 존재

사용자가 선택한 통제 변수로 propensity score를 추정한 뒤

- Common Support 적용
- 1:1 Greedy Nearest-Neighbor Matching
- Replacement 미사용
- `0.2 × SD(logit propensity score)` Caliper
- 매칭 전후 Standardized Mean Difference(SMD)
- Matched outcome에 대한 McNemar test

를 수행합니다.

매칭 후 절대 SMD `< 0.1`을 균형 진단 기준으로 사용합니다.

PSM은 관측된 통제변수의 분포 차이를 줄이는 방법이며, 관측되지 않은 교란요인을 제거하지 못합니다.

따라서 결과를 확정적인 인과효과로 해석하지 않습니다.

---

## 3. 머신러닝 예측

### 모델

현재 고소득 예측 모델은

```text
HistGradientBoostingClassifier
```

를 사용합니다.

범주형 변수는 One-Hot Encoding 대신 HistGradientBoosting의 native categorical feature 처리를 사용합니다.

모델 입력 피처는 다음과 같습니다.

```text
age
workclass
education
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
```

모델 학습과 테스트 데이터는 `80:20`으로 분리하며 `high_income`을 기준으로 stratified split을 적용합니다.

---

## 4. 현재 모델 성능

최근 전체 파이프라인 실행 기준:

| Metric | Score |
|---|---:|
| Accuracy | 0.8373 |
| Precision | 0.6130 |
| Recall | 0.8807 |
| F1 | 0.7228 |
| ROC-AUC | 0.9327 |
| Test rows | 6,508 |

현재 모델은 Recall이 상대적으로 높고 Precision은 그보다 낮습니다.

이는 실제 고소득자를 많이 탐지하는 대신 일부 비고소득자를 고소득으로 예측하는 False Positive도 발생한다는 의미입니다.

ROC-AUC는 약 `0.933`으로, 테스트 데이터에서 두 클래스를 확률 순서로 구분하는 능력은 높은 수준으로 나타났습니다.

현재 하이퍼파라미터는 이전 모델 비교 실험에서 얻은 값을 초기 설정으로 사용하고 있습니다. 현재 최종 피처 구성에서 다시 전체 하이퍼파라미터 탐색을 수행한 것은 아니므로 절대적인 최적값이라고 해석하지 않습니다.

---

## 5. 개인 예측 설명

단순한 분류 결과뿐 아니라 각 입력값이 현재 모델 예측과 어떤 관계가 있는지도 확인할 수 있습니다.

현재 설명 방식은 각 feature를 하나씩 학습 데이터의 대표값으로 변경한 뒤 예측 확률의 변화를 계산합니다.

예:

```text
현재 age = 52
학습 데이터 대표 age = 37

기존 예측 확률 = 68%
age만 대표값으로 변경한 예측 확률 = 59%

대표값 대비 예측 확률 차이 = +9%p
```

이 값은 개별 feature를 각각 독립적으로 변경한 결과이므로 서로 더할 수 없습니다.

또한 SHAP value나 인과효과를 의미하지 않습니다.

---

## 6. What-if Simulation

사용자는 다른 모든 조건을 유지한 상태에서 하나의 변수만 변경해 모델 예측이 어떻게 달라지는지 확인할 수 있습니다.

연속형 변수는 학습 데이터의 약 5~95 분위 범위에서 여러 값을 생성합니다.

예:

```text
age만 변화
다른 입력값은 고정

age 25 → 고소득 예측 확률 18%
age 35 → 고소득 예측 확률 31%
age 45 → 고소득 예측 확률 48%
...
```

범주형 변수는 학습 당시 관측된 각 범주별 예측 확률을 비교합니다.

What-if 결과는 모델의 민감도 또는 시나리오 변화에 대한 예측 결과이며, 해당 변수를 실제로 변화시켰을 때 발생하는 인과효과가 아닙니다.

---

## 7. 전체 모델 Feature Importance

전체 테스트셋을 기준으로 Permutation Importance를 계산합니다.

각 변수를 무작위로 섞었을 때 ROC-AUC가 얼마나 감소하는지를 이용해 모델의 전체적인 예측 기여도를 평가합니다.

Feature Importance는

> "이 변수가 고소득의 원인이다"

를 의미하지 않습니다.

여러 피처가 동시에 모델에 포함된 상태에서 해당 정보가 예측 성능에 얼마나 기여하는지를 나타냅니다.

---

## 8. 모델 공정성 진단

모델은 `sex`, `race` 그룹별로

- Recall
- False Negative Rate
- 실제 양성 표본 수

를 별도로 계산합니다.

양성 표본이 30개 이상인 집단만 상대적으로 신뢰 가능한 집단으로 분류합니다.

현재 내부 진단 기준:

```text
Minimum group Recall >= 0.60
Maximum Recall gap <= 0.10
```

최근 실행에서는 `sex`, `race` 모두 Recall gap이 `0.10`을 소폭 초과했습니다.

따라서 현재 모델은 전체 성능이 높더라도 집단별 성능이 완전히 동일하지 않으며, 실제 서비스에서 모델 결과를 해석할 때 이 차이를 함께 고려해야 합니다.

---

## 9. 데이터 정제

원본 데이터:

```text
data/raw/adult.csv
```

공통 정제 과정은 다음과 같습니다.

1. 문자열 앞뒤 공백 제거
2. `"?"`, 빈 문자열을 결측값으로 통일
3. 숫자형 변수 변환
4. 변환할 수 없는 숫자값은 결측 처리
5. 완전 중복 행 제거
6. 논리적 허용 범위를 벗어난 행 제거
7. `income`으로부터 `high_income` 생성

공통 정제 단계에서는 단순히 결측값이 존재한다는 이유로 행 전체를 삭제하지 않습니다.

각 연관성 분석에서는 사용자가 선택한

```text
target + exposure + controls
```

에 필요한 열만 기준으로 Complete-case 표본을 생성합니다.

머신러닝에서는 target이 없는 행만 제거하고 피처 결측값은 모델 Pipeline에서 처리합니다.

---

## 10. 프로젝트 구조

```text
.
├── app.py
├── main.py
│
├── data/
│   └── raw/
│       └── adult.csv
│
├── docs/
│   ├── ANALYSIS_DESIGN.md
│   ├── ML_MODELING_SUMMARY.md
│   └── MODEL_SELECTION_LOG.md
│
├── outputs/
│   ├── figures/
│   │   ├── model_confusion_matrix.png
│   │   ├── model_performance_metrics.png
│   │   └── model_roc_curve.png
│   │
│   ├── models/
│   │   └── income_model_bundle.joblib
│   │
│   └── tables/
│       ├── model_card.json
│       ├── model_fairness_by_group.csv
│       ├── model_feature_importance.csv
│       ├── model_input_schema.json
│       ├── model_metrics.json
│       └── model_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── association.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── modeling.py
│   ├── model_visualization.py
│   ├── statistics.py
│   └── visualization.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 11. 모듈 역할

### `app.py`

Streamlit 웹 UI를 담당합니다.

분석이나 머신러닝 계산을 직접 구현하지 않고 `src`의 검증된 함수들을 호출합니다.

### `src/config.py`

- 공통 경로
- Adult 데이터 컬럼
- 분석 가능 변수
- 변수 타입
- 예측 피처
- Random seed

를 관리합니다.

### `src/data.py`

Adult 데이터를 로딩하고 모든 기능에서 사용할 공통 정제 DataFrame을 생성합니다.

### `src/eda.py`

개발 및 데이터 품질 점검을 위한 범용 EDA를 수행합니다.

### `src/statistics.py`

연관성 분석에서 필요한 범용 통계 계산을 담당합니다.

- Binary group association
- Propensity Score Matching
- SMD balance diagnostics
- McNemar test

### `src/association.py`

사용자의 관심 변수와 통제 변수 선택을 받아 전체 연관성 분석 흐름을 담당합니다.

```text
Request validation
→ Analysis sample
→ Unadjusted association
→ Adjusted Logistic Regression
→ Optional PSM
→ Result object
```

### `src/modeling.py`

- HistGradientBoosting 학습
- 모델 평가
- 모델 bundle 저장
- 사용자 입력 예측
- 개인별 예측 설명
- What-if simulation
- Permutation Importance
- Fairness diagnostics

를 담당합니다.

### `src/visualization.py`

웹 사용자에게 표시되는 Plotly Figure를 생성합니다.

분석을 다시 계산하거나 중간 CSV 파일을 읽지 않고 `association.py`, `modeling.py`가 반환한 결과를 직접 시각화합니다.

### `src/model_visualization.py`

개발자용 모델 평가 시각화를 담당합니다.

- Performance metrics
- ROC curve
- Confusion matrix

를 PNG로 저장합니다.

### `main.py`

개발·검증용 CLI 진입점입니다.

실제 웹 사용자 요청은 `app.py`가 담당합니다.

---

## 12. 설치

### Python

Python 3.11 환경을 기준으로 개발했습니다.

### 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 13. 개발 파이프라인 실행

프로젝트 루트에서:

```bash
python main.py --stage all
```

을 실행합니다.

실행 순서:

```text
EDA
→ 데이터 정제
→ 모델 학습
→ 모델 평가
→ 모델 bundle 저장
→ 모델 진단 시각화
```

개별 단계도 실행할 수 있습니다.

```bash
python main.py --stage data
python main.py --stage eda
python main.py --stage model
python main.py --stage model-viz
```

---

## 14. 웹 애플리케이션 실행

모델 학습이 완료되어

```text
outputs/models/income_model_bundle.joblib
```

이 존재해야 합니다.

그 다음:

```bash
streamlit run app.py
```

을 실행합니다.

브라우저에서 다음 두 기능을 사용할 수 있습니다.

```text
연관성 분석
고소득 예측
```

---

## 15. 검증

핵심 서비스 기능에 대해 Smoke Test를 수행했습니다.

```text
[PASS] continuous association
[PASS] categorical association
[PASS] binary association + PSM
[PASS] individual prediction
[PASS] what-if
[PASS] global feature importance

ALL SMOKE TESTS PASSED
```

즉 다음 주요 연결을 실제 실행으로 확인했습니다.

```text
data
→ association
→ statistics
→ visualization
```

및

```text
model bundle
→ modeling
→ prediction
→ explanation
→ what-if
→ visualization
```

---

## 16. 주요 출력 파일

### 배포용 모델

```text
outputs/models/income_model_bundle.joblib
```

포함 내용:

- 학습된 sklearn Pipeline
- Prediction input schema
- Model card
- Global permutation importance

### 개발용 모델 평가

```text
outputs/tables/model_metrics.json
outputs/tables/model_predictions.csv
outputs/tables/model_feature_importance.csv
outputs/tables/model_fairness_by_group.csv
outputs/tables/model_input_schema.json
outputs/tables/model_card.json
```

### 개발용 시각화

```text
outputs/figures/model_performance_metrics.png
outputs/figures/model_roc_curve.png
outputs/figures/model_confusion_matrix.png
```

사용자별 연관성 분석 및 예측 그래프는 파일로 저장하지 않고 웹 요청 시 동적으로 생성합니다.

---

## 17. 해석상의 주의사항

### 연관성은 인과관계가 아닙니다

Adult Census Income은 관찰 데이터입니다.

Logistic Regression 또는 PSM으로 관측된 변수를 통제하더라도 데이터에 존재하지 않는

- 개인 능력
- 가정환경
- 지역 특성
- 교육비
- 네트워크
- 기타 사회경제적 요인

등은 통제할 수 없습니다.

따라서 결과는 조건부 연관성으로 해석해야 하며 확정적인 인과효과를 의미하지 않습니다.

### 예측도 인과관계가 아닙니다

머신러닝 모델이 특정 변수에 높은 Feature Importance를 부여하거나 What-if 결과에서 예측 확률이 크게 변한다고 해서 해당 변수가 실제 소득 변화를 일으킨다고 결론 내릴 수 없습니다.

### 데이터의 한계

Adult Census Income 데이터는 과거 미국 Census 기반 데이터입니다.

따라서 결과를 현재의 특정 국가, 사회, 노동시장 또는 개인에게 그대로 일반화해서는 안 됩니다.

민감 변수인 `sex`, `race` 관련 결과 역시 집단의 본질적 능력 차이로 해석해서는 안 됩니다.

---

## 18. 프로젝트 핵심 원칙

본 프로젝트는 다음 원칙을 기준으로 구현했습니다.

1. 연관성 분석과 예측을 분리한다.
2. 관심 변수와 통제 변수의 역할을 명확히 구분한다.
3. 통제 변수는 값을 고정하는 것이 아니라 통계적으로 조정한다.
4. 사용자별 분석 결과는 정적 파일에 저장하지 않는다.
5. 통계 결과와 그래프는 동일한 분석 표본을 사용한다.
6. 모델 설명과 Feature Importance를 인과효과로 표현하지 않는다.
7. PSM 결과 역시 확정적인 인과효과로 표현하지 않는다.
8. 개발용 모델 진단과 사용자용 시각화를 분리한다.
9. 오류가 있는 분석 결과를 억지로 생성하지 않는다.
10. 웹 UI는 분석 로직을 중복 구현하지 않고 기존 모듈을 호출한다.

---

## 19. 향후 개선

- 다양한 Logistic Regression 추정 실패 상황에 대한 안정적인 fallback
- 희소 범주에 대한 사용자 안내 개선
- 현재 최종 피처 구성 기준 하이퍼파라미터 재탐색
- Prediction threshold 비교 및 조정
- 집단별 모델 성능 개선
- 자동 테스트 범위 확대
- Streamlit UI/UX 개선
- 배포 환경 구성
- 분석 결과 설명 자동 생성

---

## 20. 사용 데이터

**Adult Census Income**

목표 변수:

```text
high_income
```

정의:

```text
income > 50K  → 1
income <= 50K → 0
```

본 프로젝트는 실제 소득액을 예측하는 회귀 프로젝트가 아니라  
**연 소득 50,000달러 초과 여부를 분석하고 예측하는 이진 분류 프로젝트**입니다.