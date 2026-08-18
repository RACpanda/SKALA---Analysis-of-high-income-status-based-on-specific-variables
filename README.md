# Adult Income Explorer

Adult Census Income 데이터를 기반으로 사용자가 직접 변수를 선택하여  
**고소득 여부와의 연관성을 분석하고, 개별 조건에 대한 고소득 확률을 예측하는 웹 기반 분석·예측 도구**입니다.

SKALA 과정에서 학습한 데이터 처리, 통계 분석, 머신러닝, 모델 평가, 시각화, 웹 서비스 구현을 하나의 End-to-End 프로젝트로 통합하는 것을 목표로 합니다.

---

# 1. 프로젝트 소개

Adult Census Income 데이터는 개인의 나이, 교육 수준, 직업, 근무시간 등의 특성과 함께 연 소득이 50,000달러를 초과하는지 여부를 제공합니다.

본 프로젝트는 크게 두 가지 기능으로 구성됩니다.

### 연관성 분석

사용자가 직접

- 관심 변수(Exposure) 1개
- 통제 변수(Control) 0개 이상

을 선택하여 `high_income`과의 관계를 분석합니다.

질문의 예시는 다음과 같습니다.

```text
관심 변수: education
통제 변수: age, sex
```

분석 질문:

> 나이와 성별을 통계적으로 고려한 뒤에도 교육 수준과 고소득 여부 사이의 연관성이 나타나는가?

### 고소득 예측

사용자가 한 개인의 실제 조건을 입력하면 학습된 머신러닝 모델이

- 연 소득 50,000달러 초과 여부
- 고소득 예측 확률
- 개인 입력 기준 예측 설명
- 전체 모델 Feature Importance
- 특정 입력값 변화에 따른 What-if 결과

를 제공합니다.

연관성 분석과 머신러닝 예측은 서로 다른 질문을 다룹니다.

> **연관성 분석**  
> 어떤 변수가 고소득 여부와 어떻게 연결되어 있는가?

> **머신러닝 예측**  
> 현재 입력 조건에서 모델이 고소득일 확률을 얼마나 예측하는가?

---

# 2. Version History

## Version 1.0

Version 1.0에서는 Adult Income Explorer의 기본 분석·예측 서비스를 구현했습니다.

### 연관성 분석

- 사용자 관심 변수 선택
- 사용자 통제 변수 선택
- 연속형 / 이진형 / 범주형 변수별 조정 전 분석
- Logistic Regression 기반 조정 후 분석
- Adjusted Odds Ratio
- 선택적 Propensity Score Matching
- Plotly 기반 동적 시각화

### 머신러닝 예측

- HistGradientBoosting 기반 고소득 예측
- 개별 고소득 예측 확률
- 개인 입력 기준 예측 설명
- What-if Simulation
- Global Permutation Importance
- `sex`, `race` 기준 공정성 진단

### 웹 서비스

- Streamlit 기반 사용자 인터페이스
- 연관성 분석 / 고소득 예측 기능 분리
- 사용자 요청 시 동적으로 분석 및 시각화

---

## Version 1.1

Version 1.1에서는 새로운 머신러닝 모델을 추가하거나 예측 모델을 교체하지 않았습니다.

대신 **Version 1.0의 연관성 분석 엔진을 더 안정적이고 해석하기 쉽게 강화했습니다.**

### 주요 변경사항

| 기능 | Version 1.0 | Version 1.1 |
|---|---|---|
| 조정 모형 | Standard Logistic Regression | Logit + 필요 시 Binomial GLM fallback |
| 범주형 변수 분석 | 범주별 Adjusted OR | Overall Wald Test + 범주별 Adjusted OR |
| 조정 결과 해석 | Odds Ratio 중심 | Odds Ratio + Adjusted Probability |
| 추정 실패 처리 | 분석 실패 가능 | 안정적 fallback 및 진단 강화 |
| 머신러닝 예측 모델 | HistGradientBoosting | 변경 없음 |
| What-if / Feature Importance | 지원 | 변경 없음 |

Version 1.1의 핵심은 다음 세 가지입니다.

1. **Logistic Regression 안정성 강화**
2. **범주형 관심 변수 Overall Wald Test**
3. **Adjusted Probability**

---

# 3. 연관성 분석

## 3.1 분석 가능 변수

관심 변수와 통제 변수로 사용할 수 있는 변수는 다음과 같습니다.

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

다음 변수는 사용자 선택 변수에서 제외했습니다.

### `fnlwgt`

Census 표본 가중치이므로 일반적인 관심 변수로 사용하지 않습니다.

### `education-num`

`education`의 숫자형 표현으로 정보가 중복되므로 제외했습니다.

### `income`, `high_income`

결과 변수이므로 설명 변수로 사용할 수 없습니다.

---

# 4. 조정 전 연관성 분석

관심 변수의 유형에 따라 분석 방법을 자동으로 선택합니다.

## Continuous

대상 예:

- age
- capital-gain
- capital-loss
- hours-per-week

Point-biserial correlation을 이용하여 연속형 변수와 `high_income` 사이의 조정 전 연관성을 확인합니다.

시각화에서는 동일한 분석 표본을 분위 구간으로 나누어 구간별 실제 고소득률을 보여줍니다.

---

## Binary

현재 기본 이진 관심 변수:

```text
sex
```

두 집단에 대해 다음을 계산합니다.

- 표본 수
- 고소득률
- 고소득률 차이
- Risk Ratio
- Odds Ratio
- 95% Confidence Interval
- Fisher exact p-value
- Cohen's h

---

## Categorical

대상 예:

- education
- occupation
- workclass
- race
- native-country

각 범주의 실제 고소득률을 계산하고 Chi-square test를 수행합니다.

---

# 5. 통제 변수 조정

사용자가 선택한 통제 변수는 특정 값으로 고정되는 것이 아닙니다.

예:

```text
관심 변수: education
통제 변수: age, sex, race
```

이 분석은 다음 질문을 의미합니다.

> 나이, 성별, 인종을 통계적으로 고려한 뒤에도 교육 수준과 고소득 여부 사이의 연관성이 나타나는가?

조정 후 분석은 기본적으로 이항 Logistic Regression을 이용합니다.

주요 결과:

- Adjusted Odds Ratio
- 95% Confidence Interval
- p-value
- Adjusted Probability
- 범주형 관심 변수 Overall Wald Test

이 결과는 선택한 통제 변수를 고려한 **조건부 연관성**이며 인과효과를 의미하지 않습니다.

---

# 6. Version 1.1 — Logistic Regression 안정성 강화

## 문제

범주가 많은 변수나 희소한 범주가 포함된 경우 다음과 같은 문제가 발생할 수 있습니다.

- Logistic Regression 수렴 실패
- 완전분리 또는 준완전분리
- Singular / near-singular matrix
- 매우 큰 회귀계수 또는 Odds Ratio
- 불안정한 신뢰구간

Version 1.0에서는 이러한 조합에서 분석이 실패할 수 있었습니다.

## Version 1.1 해결 구조

Version 1.1에서는 다음 순서로 조정 모형을 적합합니다.

```text
Standard Logistic Regression
        │
        ├── 정상 수렴
        │       ↓
        │    결과 사용
        │
        └── 실패 또는 불안정
                ↓
        Binomial GLM
        IRLS + pseudo-inverse
                │
                ├── 정상 수렴
                │       ↓
                │    결과 사용
                │
                └── 여전히 불안정
                        ↓
                  분석 실패 안내
```

GLM fallback은 분석 문제 자체를 바꾸는 것이 아닙니다.

현재 프로젝트에서는 동일한 **Binomial + Logit** 구조를 다른 적합 방식으로 계산하여 수치적 안정성을 높이는 목적으로 사용합니다.

결과에는 실제 적합 방식도 기록합니다.

```text
standard_logit
binomial_glm_pinv
```

다만 완전분리처럼 데이터 자체에서 회귀계수를 안정적으로 식별할 수 없는 경우에는 GLM fallback으로도 해결되지 않을 수 있습니다.

이 경우 불안정한 결과를 억지로 제공하지 않습니다.

---

# 7. Version 1.1 — Overall Wald Test

범주형 관심 변수는 하나의 기준 범주와 여러 비기준 범주로 변환됩니다.

예를 들어 `education`을 분석하면 다음과 같은 개별 회귀계수가 만들어질 수 있습니다.

```text
Bachelors vs reference
Masters vs reference
Doctorate vs reference
...
```

Version 1.0에서는 각 범주의 Adjusted Odds Ratio와 p-value를 개별적으로 확인했습니다.

하지만 이것만으로는 다음 질문에 바로 답하기 어렵습니다.

> 교육 수준이라는 변수 전체가 통제 변수 조정 후에도 고소득 여부와 연관되어 있는가?

Version 1.1에서는 이를 위해 **Overall Wald Test**를 추가했습니다.

## 귀무가설

```text
H0:

범주형 관심 변수의
모든 비기준 범주 회귀계수 = 0
```

예:

```text
β_Bachelors = 0
β_Masters   = 0
β_Doctorate = 0
...
```

Overall p-value가 충분히 작으면 선택한 통제 변수를 고려한 뒤에도 해당 범주형 변수 전체가 고소득 여부와 통계적으로 연관되어 있다는 근거로 해석합니다.

결과 화면에서는 다음 정보를 제공합니다.

- Wald χ²
- Degrees of Freedom
- Overall p-value

### 주의

Overall Test가 유의하다고 해서 모든 범주가 서로 유의하게 다르다는 의미는 아닙니다.

따라서 범주형 변수는 다음 순서로 해석합니다.

```text
Overall Wald Test
변수 전체의 연관성 확인
        ↓
Adjusted Odds Ratio
각 범주와 기준 범주의 세부 차이 확인
```

---

# 8. Version 1.1 — Adjusted Probability

Adjusted Odds Ratio는 중요한 통계량이지만 일반 사용자가 실제 차이의 크기를 직관적으로 이해하기 어려울 수 있습니다.

따라서 Version 1.1에서는 **Adjusted Probability**를 추가했습니다.

## 계산 방법

Average Marginal Prediction 방식으로 계산합니다.

```text
실제 분석 표본 유지
        ↓
각 사람의 통제 변수 값 유지
        ↓
관심 변수만 특정 값 또는 범주로 변경
        ↓
각 사람의 고소득 확률 계산
        ↓
전체 확률 평균
```

예를 들어:

```text
관심 변수: education
통제 변수: age, sex
```

라면 실제 분석 표본의 `age`, `sex` 값은 그대로 둡니다.

대신 모든 사람의 `education`만 특정 교육 수준으로 바꾸어 예측 확률을 계산합니다.

예:

```text
Bachelors    34.8%
Masters      47.1%
Doctorate    58.6%
```

숫자는 분석 결과에 따라 달라집니다.

## 세 결과의 차이

```text
조정 전 고소득률
→ 실제 관측 데이터에서 계산한 비율

Adjusted Odds Ratio
→ Logistic Regression / GLM의 조건부 연관성

Adjusted Probability
→ 동일 조정 모형에서 계산한 평균 고소득 예측확률
```

Adjusted Probability는 실제 집단 고소득률과 동일한 값이 아니며 인과효과도 의미하지 않습니다.

또한 머신러닝 예측 기능에서 제공하는 **한 개인의 HistGradientBoosting 예측 확률과도 다른 값**입니다.

---

# 9. Propensity Score Matching

현재 PSM은 다음 조건에서 선택적으로 사용할 수 있습니다.

- 관심 변수가 Binary
- 통제 변수가 1개 이상 존재

사용자가 선택한 통제 변수로 propensity score를 추정한 뒤 다음 절차를 수행합니다.

- Common Support 적용
- 1:1 Greedy Nearest-Neighbor Matching
- Replacement 미사용
- `0.2 × SD(logit propensity score)` Caliper
- 매칭 전후 Standardized Mean Difference
- Matched outcome McNemar test

매칭 후 절대 SMD `< 0.1`을 균형 진단 기준으로 사용합니다.

PSM은 관측된 통제 변수의 분포 차이를 줄이는 방법입니다.

관측되지 않은 교란요인은 제거할 수 없으므로 결과를 확정적인 인과효과로 해석하지 않습니다.

---

# 10. 연관성 분석 결과 구조

Version 1.1의 브라우저 결과는 다음 흐름으로 구성됩니다.

```text
01 · UNADJUSTED

관심 변수 유형별 조정 전 분석
- Binary: 집단별 고소득률
- Continuous: Point-biserial correlation
- Categorical: 범주별 고소득률 + Chi-square


02 · ADJUSTED

통제 변수 조정 후 분석
- Standard Logistic Regression
- 필요 시 Binomial GLM fallback
- 범주형 변수 Overall Wald Test
- Adjusted Odds Ratio
- 95% Confidence Interval
- p-value


03 · ADJUSTED PROBABILITY

Average Marginal Prediction 기반
조정 고소득 확률


04 · PROPENSITY SCORE MATCHING

Binary exposure에서
사용자가 요청한 경우 추가 수행
```

---

# 11. 머신러닝 예측

Version 1.1에서도 머신러닝 예측 모델은 Version 1.0과 동일합니다.

## 모델

```text
HistGradientBoostingClassifier
```

범주형 변수는 One-Hot Encoding 대신 HistGradientBoosting의 native categorical feature 처리를 사용합니다.

입력 피처:

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

모델 학습과 테스트 데이터는 `80:20`으로 분리하며 `high_income` 기준 stratified split을 적용합니다.

---

# 12. 현재 머신러닝 모델 성능

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

ROC-AUC는 약 `0.933`으로 나타났습니다.

현재 하이퍼파라미터는 이전 모델 비교 실험에서 얻은 값을 초기 설정으로 사용하고 있습니다.

현재 최종 피처 구성에서 전체 하이퍼파라미터 탐색을 다시 수행한 것은 아니므로 절대적인 최적값으로 해석하지 않습니다.

---

# 13. 개인 예측 설명

단순한 분류 결과뿐 아니라 각 입력값이 현재 예측과 어떤 관계가 있는지도 확인합니다.

각 feature의 현재 값을 학습 데이터의 대표값으로 하나씩 변경하고 예측 확률 변화를 계산합니다.

예:

```text
현재 age = 52
학습 데이터 대표 age = 37

기존 예측 확률 = 68%
age만 대표값으로 변경 = 59%

확률 차이 = +9%p
```

각 feature를 독립적으로 변경한 결과이므로 변수별 차이를 서로 더할 수 없습니다.

또한 SHAP value나 인과효과를 의미하지 않습니다.

---

# 14. What-if Simulation

사용자는 다른 모든 입력 조건을 유지하고 하나의 변수만 변경하여 모델 예측 확률이 어떻게 달라지는지 확인할 수 있습니다.

연속형 변수는 학습 데이터의 약 5~95 분위 범위에서 여러 값을 생성합니다.

예:

```text
age 25 → 18%
age 35 → 31%
age 45 → 48%
...
```

범주형 변수는 학습 당시 관측된 범주별 예측 확률을 비교합니다.

What-if 결과는 모델의 입력 민감도 또는 시나리오 변화에 대한 예측이며 인과효과가 아닙니다.

---

# 15. Global Feature Importance

전체 테스트셋을 기준으로 Permutation Importance를 계산합니다.

각 변수를 무작위로 섞었을 때 ROC-AUC가 얼마나 감소하는지를 이용하여 모델의 전체적인 예측 기여도를 평가합니다.

Feature Importance는

> 이 변수가 고소득의 원인이다.

를 의미하지 않습니다.

여러 피처가 동시에 모델에 포함된 상태에서 해당 정보가 예측에 얼마나 기여하는지를 나타냅니다.

---

# 16. 모델 공정성 진단

모델은 `sex`, `race` 그룹별로 다음을 계산합니다.

- Recall
- False Negative Rate
- 실제 양성 표본 수

양성 표본이 30개 이상인 집단을 중심으로 성능 차이를 확인합니다.

현재 내부 진단 기준:

```text
Minimum group Recall >= 0.60
Maximum Recall gap <= 0.10
```

최근 실행에서는 `sex`, `race` 모두 Recall gap이 `0.10`을 소폭 초과했습니다.

따라서 전체 모델 성능뿐 아니라 집단별 성능 차이도 함께 고려해야 합니다.

---

# 17. 데이터 정제

원본 데이터:

```text
data/raw/adult.csv
```

공통 정제 과정:

1. 문자열 앞뒤 공백 제거
2. `"?"`, 빈 문자열을 결측값으로 통일
3. 숫자형 변수 변환
4. 변환 실패 숫자값 결측 처리
5. 완전 중복 행 제거
6. 논리적 허용 범위를 벗어난 행 제거
7. `income`으로부터 `high_income` 생성

공통 정제 단계에서는 결측값이 있다는 이유만으로 모든 행을 제거하지 않습니다.

각 연관성 분석에서는 선택된

```text
target + exposure + controls
```

에 필요한 열만 기준으로 Complete-case 표본을 구성합니다.

머신러닝에서는 target이 없는 행만 제거하고 피처 결측값은 모델 Pipeline에서 처리합니다.

---

# 18. 프로젝트 구조

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

# 19. 모듈 역할

## `app.py`

Streamlit 웹 UI를 담당합니다.

분석 또는 머신러닝 계산을 직접 구현하지 않고 `src` 모듈에서 생성된 결과를 화면에 표시합니다.

---

## `src/config.py`

- 공통 경로
- Adult 데이터 컬럼
- 분석 가능 변수
- 변수 타입
- 예측 피처
- Random seed

를 관리합니다.

---

## `src/data.py`

Adult 데이터를 로딩하고 모든 기능에서 사용할 공통 정제 DataFrame을 생성합니다.

---

## `src/eda.py`

개발 및 데이터 품질 점검을 위한 범용 EDA를 수행합니다.

---

## `src/statistics.py`

연관성 분석에서 재사용되는 통계 계산을 담당합니다.

- Binary group association
- Propensity Score Matching
- SMD balance diagnostics
- McNemar test

---

## `src/association.py`

사용자의 관심 변수와 통제 변수 선택을 받아 전체 연관성 분석 흐름을 담당합니다.

### Version 1.1 기준 흐름

```text
Request validation
        ↓
Analysis sample
        ↓
Unadjusted association
        ↓
Standard Logistic Regression
        ↓
필요 시 Binomial GLM fallback
        ↓
Categorical Overall Wald Test
        ↓
Adjusted Odds Ratio
        ↓
Adjusted Probability
        ↓
Optional PSM
        ↓
Result object
```

---

## `src/modeling.py`

- HistGradientBoosting 학습
- 모델 평가
- 모델 bundle 저장
- 사용자 입력 예측
- 개인별 예측 설명
- What-if Simulation
- Permutation Importance
- Fairness diagnostics

를 담당합니다.

Version 1.1에서는 머신러닝 모델링 구조를 변경하지 않았습니다.

---

## `src/visualization.py`

웹 사용자용 Plotly Figure를 생성합니다.

주요 시각화:

- 조정 전 고소득률
- 연속형 관심 변수 구간별 고소득률
- Adjusted Odds Ratio 및 95% CI
- Adjusted Probability
- PSM balance
- 개인 고소득 예측 확률
- 개인별 예측 설명
- What-if Simulation
- Global Permutation Importance

통계량을 다시 계산하지 않고 `association.py`, `modeling.py`에서 반환된 결과를 시각화합니다.

---

## `src/model_visualization.py`

개발자용 모델 평가 시각화를 담당합니다.

- Performance metrics
- ROC curve
- Confusion matrix

를 PNG로 저장합니다.

---

## `main.py`

개발 및 검증용 CLI 진입점입니다.

웹 사용자 요청은 `app.py`가 담당합니다.

---

# 20. 설치

Python 3.11 환경을 기준으로 개발했습니다.

```bash
pip install -r requirements.txt
```

---

# 21. 개발 파이프라인 실행

프로젝트 루트에서:

```bash
python main.py --stage all
```

실행 순서:

```text
EDA
→ 데이터 정제
→ 모델 학습
→ 모델 평가
→ 모델 bundle 저장
→ 모델 진단 시각화
```

개별 단계:

```bash
python main.py --stage data
python main.py --stage eda
python main.py --stage model
python main.py --stage model-viz
```

---

# 22. 웹 애플리케이션 실행

모델 학습이 완료되어 다음 파일이 존재해야 합니다.

```text
outputs/models/income_model_bundle.joblib
```

실행:

```bash
streamlit run app.py
```

브라우저에서는 두 기능을 사용할 수 있습니다.

```text
연관성 분석
고소득 예측
```

---

# 23. 검증

## Version 1.0 Smoke Test

```text
[PASS] continuous association
[PASS] categorical association
[PASS] binary association + PSM
[PASS] individual prediction
[PASS] what-if
[PASS] global feature importance

ALL SMOKE TESTS PASSED
```

검증된 주요 연결:

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

## Version 1.1 추가 검증

### Adjusted Probability

```text
결과 객체 생성
→ 모든 probability가 0~1 범위인지 확인
→ Plotly 그래프 생성 확인
```

### Overall Wald Test

```text
Categorical exposure
→ Overall Test 생성

Continuous / Binary exposure
→ Overall Test 미생성

Wald statistic
Degrees of Freedom
p-value
→ 정상 범위 확인
```

### Logistic Regression

```text
Standard Logit
→ 정상 수렴 확인
```

### GLM Fallback

테스트 환경에서 Standard Logit을 의도적으로 실패시킨 뒤 fallback 경로를 확인합니다.

```text
Standard Logit 강제 실패
        ↓
Binomial GLM
        ↓
fit_method = binomial_glm_pinv
        ↓
converged = True
```

---

# 24. 주요 출력 파일

## 배포용 모델

```text
outputs/models/income_model_bundle.joblib
```

포함 내용:

- 학습된 sklearn Pipeline
- Prediction input schema
- Model card
- Global permutation importance

---

## 개발용 모델 평가

```text
outputs/tables/model_metrics.json
outputs/tables/model_predictions.csv
outputs/tables/model_feature_importance.csv
outputs/tables/model_fairness_by_group.csv
outputs/tables/model_input_schema.json
outputs/tables/model_card.json
```

---

## 개발용 시각화

```text
outputs/figures/model_performance_metrics.png
outputs/figures/model_roc_curve.png
outputs/figures/model_confusion_matrix.png
```

사용자별 연관성 분석 및 예측 그래프는 정적 파일로 저장하지 않고 웹 요청 시 동적으로 생성합니다.

---

# 25. 해석상의 주의사항

## 연관성은 인과관계가 아닙니다

Adult Census Income은 관찰 데이터입니다.

Logistic Regression, GLM 또는 PSM으로 관측된 변수를 통제하더라도 데이터에 포함되지 않은 요인은 통제할 수 없습니다.

예:

- 개인 능력
- 가정환경
- 지역 특성
- 교육비
- 네트워크
- 기타 사회경제적 요인

따라서 결과는 조건부 연관성으로 해석해야 합니다.

---

## Adjusted Probability도 인과효과가 아닙니다

Adjusted Probability는 조정 모형으로 계산한 평균 예측확률입니다.

특정 변수의 값을 실제로 변화시키면 소득이 해당 확률만큼 변화한다는 의미가 아닙니다.

---

## 머신러닝 예측도 인과관계가 아닙니다

높은 Feature Importance나 What-if 결과가 해당 변수가 실제 소득 변화를 일으킨다는 의미는 아닙니다.

---

## 데이터의 한계

Adult Census Income 데이터는 과거 미국 Census 기반 데이터입니다.

결과를 현재 특정 국가, 사회, 노동시장 또는 개인에게 그대로 일반화해서는 안 됩니다.

`sex`, `race` 관련 결과도 집단의 본질적인 능력 차이로 해석해서는 안 됩니다.

---

# 26. 프로젝트 핵심 원칙

1. 연관성 분석과 머신러닝 예측을 분리한다.
2. 관심 변수와 통제 변수의 역할을 명확히 구분한다.
3. 통제 변수는 특정 값으로 고정하지 않고 통계적으로 조정한다.
4. 조정 전 결과와 조정 후 결과를 구분한다.
5. 통계 결과와 시각화는 동일한 분석 표본을 사용한다.
6. Logistic Regression 및 GLM 결과를 인과효과로 표현하지 않는다.
7. PSM 결과 역시 확정적인 인과효과로 표현하지 않는다.
8. 범주형 관심 변수는 전체 효과와 개별 범주 효과를 구분한다.
9. Adjusted Probability와 실제 관측 고소득률을 구분한다.
10. Adjusted Probability와 머신러닝 개인 예측 확률을 구분한다.
11. 일반 Logistic Regression이 불안정하면 검증된 fallback을 사용한다.
12. fallback으로도 안정적인 추정이 불가능하면 결과를 억지로 생성하지 않는다.
13. 모델 설명과 Feature Importance를 인과효과로 표현하지 않는다.
14. 개발용 모델 진단과 사용자용 시각화를 분리한다.
15. 웹 UI는 분석 로직을 중복 구현하지 않는다.

---

# 27. 향후 개선

Version 1.1 이후에는 새로운 분석법을 계속 추가하기보다 현재 기능의 검증과 서비스 완성도를 높이는 것을 우선합니다.

- 핵심 서비스 자동 테스트 확대
- 다양한 변수 조합에 대한 분석 안정성 검증
- 사용자 친화적인 오류 메시지 개선
- 결과 화면 UI/UX 고도화
- 현재 최종 피처 구성 기준 하이퍼파라미터 재탐색
- Prediction threshold 비교 및 조정
- 집단별 모델 성능 및 공정성 진단 개선
- 분석 결과 다운로드 기능 검토
- 실제 배포 환경 구성

---

# 28. 사용 데이터

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

본 프로젝트는 실제 연소득 금액을 예측하는 회귀 프로젝트가 아닙니다.

**연 소득 50,000달러 초과 여부를 분석하고 예측하는 이진 결과 분석·분류 프로젝트입니다.**