# Adult Income Explorer

Adult Census Income 데이터를 기반으로 사용자가 직접 변수를 선택하여  
**고소득 여부와의 연관성을 분석하고, 개별 조건에 대한 고소득 확률을 예측하는 웹 기반 분석·예측 도구**입니다.

SKALA 과정에서 학습한 데이터 처리, 통계 분석, 머신러닝, 모델 평가, 시각화, 웹 서비스 구현을 하나의 End-to-End 프로젝트로 통합하는 것을 목표로 합니다.

---

# 1. 프로젝트 소개

Adult Census Income 데이터는 개인의 나이, 교육 수준, 직업, 근무시간 등의 특성과 함께 연 소득이 50,000달러를 초과하는지 여부를 제공합니다.

본 프로젝트는 크게 두 가지 기능으로 구성됩니다.

## 연관성 분석

사용자가 직접

- 관심 변수(Exposure) 1개
- 통제 변수(Control) 0개 이상

을 선택하여 `high_income`과의 관계를 분석합니다.

예:

```text
관심 변수: education
통제 변수: age, sex
```

분석 질문:

> 나이와 성별을 통계적으로 고려한 뒤에도 교육 수준과 고소득 여부 사이의 연관성이 나타나는가?

## 고소득 예측

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

## Version 1.0 — Service Foundation

분석·예측 서비스의 기본 구조를 구현했습니다.

### 연관성 분석

- 사용자 관심 변수 선택
- 사용자 통제 변수 선택
- Continuous / Binary / Categorical 변수별 조정 전 분석
- Logistic Regression 기반 조정 후 분석
- Adjusted Odds Ratio
- 선택적 Propensity Score Matching
- Plotly 기반 동적 시각화

### 머신러닝 예측

- HistGradientBoosting 기반 고소득 예측
- 개인 고소득 예측 확률
- 개인 입력 기준 예측 설명
- What-if Simulation
- Global Permutation Importance
- `sex`, `race` 기준 공정성 진단

### 웹 서비스

- Streamlit 기반 UI
- 연관성 분석 / 고소득 예측 기능 분리
- 사용자 요청에 따른 동적 분석 및 시각화

---

## Version 1.1 — Association Analysis Upgrade

Version 1.1에서는 머신러닝 모델을 변경하지 않고  
**연관성 분석의 안정성과 해석력을 강화했습니다.**

주요 변경사항:

- Standard Logistic Regression 실패 시 Binomial GLM fallback
- 범주형 관심 변수 Overall Wald Test
- Average Marginal Prediction 기반 Adjusted Probability
- 회귀모형 수렴 및 추정 실패 진단 강화

### 핵심 변화

```text
v1.0
Adjusted Odds Ratio 중심

        ↓

v1.1
Adjusted Odds Ratio
+
Overall Wald Test
+
Adjusted Probability
+
GLM fallback
```

---

## Version 1.2 — Prediction Reliability Upgrade

Version 1.2에서는 새로운 머신러닝 알고리즘을 추가하기보다  
**현재 예측 모델의 신뢰성과 확률 해석을 강화했습니다.**

주요 작업:

1. 현재 최종 피처 기준 모델 재튜닝
2. Probability Calibration 검증
3. Classification Threshold 재검토

### 최종 결정

| 항목 | 결정 |
|---|---|
| 기본 모델 | HistGradientBoostingClassifier 유지 |
| Hyperparameter | 기존 설정 유지 |
| Probability Calibration | Sigmoid |
| Classification Threshold | 0.50 |
| 예측 확률 | Calibrated Probability 사용 |

즉 현재 최종 예측 구조는 다음과 같습니다.

```text
사용자 입력
    ↓
공통 전처리
    ↓
HistGradientBoostingClassifier
    ↓
Raw Probability
    ↓
Sigmoid Calibration
    ↓
Calibrated Probability
    ↓
Threshold 0.50
    ↓
>50K / <=50K
```

---

# 3. Version 1.2 모델 재튜닝

현재 서비스에서 실제 사용하는 최종 예측 피처를 기준으로  
HistGradientBoosting의 하이퍼파라미터를 다시 탐색했습니다.

## 방법

- 현재 production 데이터 처리 방식 사용
- 동일한 학습 데이터 사용
- 5-fold Stratified Cross Validation
- RandomizedSearchCV
- ROC-AUC 기준 후보 선정
- 40개 parameter combination 탐색

## 결과

| 모델 | CV ROC-AUC Mean | CV Std |
|---|---:|---:|
| 기존 모델 | 0.925832 | 0.003864 |
| 최적 튜닝 후보 | 0.926272 | 0.004279 |

차이:

```text
+0.000440
```

최적 후보:

```text
learning_rate        0.1385808858
max_depth            5
max_iter             236
max_leaf_nodes       63
min_samples_leaf     24
l2_regularization    1.6978276485
```

개선폭이 매우 작고 CV 변동성은 오히려 소폭 증가했습니다.

따라서 **현재 기존 MODEL_PARAMS를 유지**했습니다.

현재 설정:

```python
MODEL_PARAMS = {
    "learning_rate": 0.14447746112718687,
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": 0.45606998421703593,
}
```

재튜닝의 목적은 반드시 새로운 파라미터를 채택하는 것이 아니라,  
현재 설정이 최종 서비스 피처에서도 충분히 적절한지 다시 확인하는 것이었습니다.

---

# 4. Version 1.2 Probability Calibration

웹 서비스에서는 사용자에게 다음과 같이 확률값 자체를 제공합니다.

```text
고소득 예측 확률
63.4%
```

ROC-AUC가 높다고 해서 이러한 확률값 자체가 실제 발생률과 잘 일치한다는 의미는 아닙니다.

따라서 다음 세 가지를 비교했습니다.

```text
Uncalibrated
Sigmoid Calibration
Isotonic Calibration
```

## Training OOF Calibration 결과

| Method | ROC-AUC | Brier Score ↓ | Log Loss ↓ | Mean Probability Bias |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.925759 | 0.111171 | 0.337592 | +0.104995 |
| Sigmoid | **0.926222** | 0.089987 | **0.282839** | **+0.000025** |
| Isotonic | 0.925840 | **0.089973** | 0.286128 | +0.000028 |

기존 모델은:

```text
실제 고소득률       24.09%
평균 예측확률       34.59%
```

로 평균적으로 약 `+10.50%p` 높은 확률을 출력했습니다.

Sigmoid Calibration 적용 후:

```text
실제 고소득률       24.09%
평균 예측확률       24.09%
```

수준으로 평균적인 확률 편향이 크게 감소했습니다.

Isotonic의 Brier Score가 아주 조금 더 낮았지만 차이는 매우 작았고,  
Sigmoid가 Log Loss와 ROC-AUC에서 더 좋은 결과를 보였습니다.

따라서 최종 모델에는 **Sigmoid Calibration**을 적용했습니다.

---

# 5. Version 1.2 Classification Threshold

Calibration 적용 후 예측확률을 기준으로 classification threshold를 다시 검토했습니다.

탐색 범위:

```text
0.05 ~ 0.95
```

## 주요 후보

| 기준 | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 기본값 | **0.50** | 0.7781 | 0.6520 | 0.7095 |
| Best F1 | 0.39 | 0.6964 | 0.7603 | **0.7269** |
| Best Balanced Accuracy | 0.24 | 0.5922 | **0.8763** | 0.7068 |
| Precision ≈ Recall | 0.43 | 0.7288 | 0.7221 | 0.7254 |

`0.39`에서는 F1이 증가하지만 Precision이 감소하고 Recall이 증가합니다.

이는 단순한 성능 개선이 아니라 다음과 같은 정책적 선택을 의미합니다.

```text
낮은 threshold
→ 고소득자를 더 많이 탐지
→ Recall 증가
→ False Positive 증가

높은 threshold
→ 고소득 판정을 더 보수적으로 수행
→ Precision 증가
→ False Negative 증가
```

현재 프로젝트에는 False Positive와 False Negative 중 어느 쪽의 비용이 더 큰지에 대한 업무 목적이 정의되어 있지 않습니다.

따라서 보정된 확률 자체를 직관적으로 해석할 수 있도록  
**Classification Threshold는 0.50을 유지**했습니다.

---

# 6. 현재 최종 머신러닝 모델

## 기본 모델

```text
HistGradientBoostingClassifier
```

범주형 변수는 One-Hot Encoding 대신 HistGradientBoosting의 native categorical feature 처리를 사용합니다.

## Probability Calibration

```text
Sigmoid Calibration
3-fold Cross Validation
```

## Classification Threshold

```text
0.50
```

## 입력 피처

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

총 12개 피처를 사용합니다.

학습 / 테스트 데이터는 `80:20`으로 분리하며 `high_income`을 기준으로 stratified split을 적용합니다.

---

# 7. 현재 최종 모델 성능

Version 1.2 최종 모델 기준:

| Metric | Score |
|---|---:|
| Accuracy | **0.8791** |
| Precision | **0.7925** |
| Recall | 0.6747 |
| F1 | **0.7289** |
| ROC-AUC | **0.9334** |
| Brier Score | **0.0848** |
| Log Loss | **0.2704** |
| Test rows | 6,508 |

확률 수준:

```text
실제 고소득률            24.0934%
평균 예측 고소득 확률    24.3876%
평균 확률 편향            +0.2942%p
```

## Version 1.1 대비

| Metric | v1.1 | v1.2 |
|---|---:|---:|
| Accuracy | 0.8373 | **0.8791** |
| Precision | 0.6130 | **0.7925** |
| Recall | **0.8807** | 0.6747 |
| F1 | 0.7228 | **0.7289** |
| ROC-AUC | 0.9327 | **0.9334** |

Calibration 이후 `0.50` threshold를 사용하면서 예측 판정 특성이 달라졌습니다.

Version 1.1은 상대적으로 많은 표본을 고소득으로 분류하여 Recall이 높고 Precision이 낮았습니다.

Version 1.2는 고소득 판정이 더 보수적으로 바뀌면서 Precision이 증가하고 Recall이 감소했습니다.

따라서 모든 분류 지표가 일방적으로 개선됐다고 해석하지 않습니다.

Version 1.2의 핵심 개선은:

> **기존 ROC-AUC 수준을 유지하면서 사용자에게 제공하는 예측확률의 calibration을 크게 개선한 것**

입니다.

---

# 8. 연관성 분석

## 8.1 분석 가능 변수

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

제외 변수:

### `fnlwgt`

Census 표본 가중치이므로 일반적인 사용자 관심 변수에서 제외합니다.

### `education-num`

`education`의 숫자형 표현으로 정보가 중복되므로 제외합니다.

### `income`, `high_income`

결과 변수이므로 설명변수로 사용할 수 없습니다.

---

# 9. 조정 전 연관성 분석

관심 변수 유형에 따라 자동으로 분석 방법을 선택합니다.

## Continuous

예:

```text
age
capital-gain
capital-loss
hours-per-week
```

Point-biserial correlation을 이용합니다.

시각화에서는 같은 분석 표본을 분위 구간으로 나누어 실제 고소득률을 표시합니다.

## Binary

현재 기본 이진 변수:

```text
sex
```

계산 결과:

- 표본 수
- 고소득률
- 고소득률 차이
- Risk Ratio
- Odds Ratio
- 95% Confidence Interval
- Fisher exact p-value
- Cohen's h

## Categorical

예:

```text
education
occupation
workclass
race
native-country
```

각 범주의 고소득률을 비교하고 Chi-square test를 수행합니다.

---

# 10. 통제 변수 조정

통제 변수는 특정 값으로 고정되는 변수가 아닙니다.

예:

```text
관심 변수: education
통제 변수: age, sex, race
```

분석 질문:

> 나이, 성별, 인종을 통계적으로 고려한 뒤에도 교육 수준과 고소득 여부 사이의 연관성이 나타나는가?

조정 후 분석에서는 다음 결과를 제공합니다.

- Adjusted Odds Ratio
- 95% Confidence Interval
- p-value
- Overall Wald Test
- Adjusted Probability

이 결과는 선택한 통제 변수를 고려한 **조건부 연관성**이며 인과효과를 의미하지 않습니다.

---

# 11. Logistic Regression / GLM Fallback

기본 조정 모형은 Logistic Regression입니다.

일부 변수 조합에서는 희소 범주, 완전분리 또는 설명변수 구조 때문에 추정이 불안정해질 수 있습니다.

Version 1.1부터 다음 구조를 사용합니다.

```text
Standard Logistic Regression
        ↓
정상 수렴
        ↓
결과 사용

실패 또는 불안정
        ↓
Binomial GLM
IRLS + pseudo-inverse
        ↓
정상 수렴
        ↓
결과 사용

여전히 불안정
        ↓
분석 실패 안내
```

GLM fallback은 분석 질문을 변경하는 것이 아니라 동일한 Binomial-Logit 구조를 다른 적합 방식으로 계산하는 것입니다.

안정적으로 추정할 수 없는 경우 결과를 억지로 생성하지 않습니다.

---

# 12. Overall Wald Test

범주형 관심 변수에서는 각 비기준 범주의 회귀계수가 동시에 0인지 검정합니다.

귀무가설:

```text
H0:
범주형 관심 변수의 모든 비기준 범주 회귀계수 = 0
```

Overall p-value가 충분히 작다면 선택한 통제 변수를 고려한 뒤에도 해당 관심 변수 전체와 고소득 여부 사이에 통계적 연관성이 있다는 근거로 해석합니다.

Overall Test가 유의하다고 해서 모든 범주가 서로 유의하게 다르다는 의미는 아닙니다.

따라서:

```text
Overall Wald Test
        ↓
변수 전체 연관성 확인
        ↓
Adjusted OR / CI / p-value
        ↓
개별 범주 차이 확인
```

순서로 해석합니다.

---

# 13. Adjusted Probability

Adjusted Probability는 Average Marginal Prediction 방식으로 계산합니다.

```text
실제 분석 표본 유지
        ↓
통제 변수의 실제 관측값 유지
        ↓
관심 변수만 특정 값으로 변경
        ↓
각 사람의 고소득 예측확률 계산
        ↓
전체 평균 계산
```

예:

```text
관심 변수: education
통제 변수: age, sex
```

각 사람의 실제 `age`, `sex`는 유지하고 `education`만 특정 범주로 변경합니다.

이 결과는:

- 실제 관측 고소득률과 다르며
- 머신러닝 개인 예측 확률과도 다르고
- 인과효과도 아닙니다.

---

# 14. Propensity Score Matching

PSM은 다음 조건에서 선택적으로 수행합니다.

- 관심 변수가 Binary
- 통제 변수가 1개 이상

방법:

- Common Support
- 1:1 Greedy Nearest-Neighbor Matching
- Replacement 미사용
- `0.2 × SD(logit propensity score)` Caliper
- Standardized Mean Difference
- McNemar test

매칭 후 절대 SMD `< 0.1`을 균형 진단 기준으로 사용합니다.

PSM 역시 관측되지 않은 교란요인을 제거할 수 없으므로 확정적인 인과효과로 해석하지 않습니다.

---

# 15. 개인 예측 설명

각 feature의 현재 값을 학습 데이터의 대표값으로 하나씩 변경해 예측확률 변화를 계산합니다.

예:

```text
현재 age = 52
대표 age = 37

현재 예측확률 = 68%
age만 대표값으로 변경 = 59%

차이 = +9%p
```

각 feature를 독립적으로 변경한 결과이므로 영향값을 서로 더할 수 없습니다.

또한 SHAP value나 인과효과를 의미하지 않습니다.

Version 1.2부터 이러한 계산에서도 **Sigmoid 보정된 최종 예측확률**을 사용합니다.

---

# 16. What-if Simulation

다른 모든 조건을 유지하고 하나의 feature만 변경합니다.

예:

```text
age 25 → 18%
age 35 → 31%
age 45 → 48%
```

연속형 변수는 학습 데이터의 약 5~95 분위 범위에서 여러 값을 생성합니다.

범주형 변수는 학습 당시 관측된 범주를 사용합니다.

What-if 결과는 모델의 입력 민감도와 시나리오별 예측을 보여주는 것이며 인과효과가 아닙니다.

Version 1.2부터 What-if 역시 **보정된 예측확률**을 사용합니다.

---

# 17. Global Feature Importance

전체 테스트셋을 기준으로 Permutation Importance를 계산합니다.

각 변수를 무작위로 섞었을 때 ROC-AUC가 얼마나 감소하는지 이용합니다.

Feature Importance는:

> 해당 변수가 고소득의 원인이다.

라는 의미가 아닙니다.

여러 변수가 모델에 동시에 포함된 상태에서 해당 정보가 예측 성능에 얼마나 기여하는지를 나타냅니다.

---

# 18. 모델 공정성 진단

`sex`, `race` 그룹별로 다음을 계산합니다.

- Recall
- False Negative Rate
- 실제 양성 표본 수

양성 표본이 30개 이상인 집단을 중심으로 결과를 해석합니다.

현재 내부 기준:

```text
Minimum group Recall >= 0.60
Maximum Recall gap <= 0.10
```

공정성 진단 결과는 전체 모델 성능과 별도로 확인합니다.

---

# 19. 데이터 정제

원본 데이터:

```text
data/raw/adult.csv
```

공통 정제:

1. 문자열 공백 제거
2. `"?"`, 빈 문자열을 결측값으로 통일
3. 숫자형 변수 변환
4. 변환 실패값 결측 처리
5. 완전 중복 행 제거
6. 논리적 허용 범위 밖 행 제거
7. `income`으로 `high_income` 생성

연관성 분석에서는:

```text
target + exposure + controls
```

에 필요한 열을 기준으로 Complete-case 분석 표본을 만듭니다.

머신러닝에서는 target이 없는 행만 제거하고 피처 결측값은 Pipeline에서 처리합니다.

---

# 20. 프로젝트 구조

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
│       ├── model_predictions.csv
│       ├── model_tuning_v1_2.json
│       ├── model_calibration_v1_2.json
│       └── model_threshold_v1_2.json
│
├── scripts/
│   └── experiments/
│       ├── model_tuning_v1_2.py
│       ├── model_calibration_v1_2.py
│       └── model_threshold_v1_2.py
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

`scripts/experiments`는 서비스 실행에 필요한 production 코드가 아니라 모델 선택 과정을 재현하기 위한 실험 코드입니다.

---

# 21. 모듈 역할

## `app.py`

Streamlit 기반 웹 UI를 담당합니다.

분석 또는 머신러닝 계산을 직접 구현하지 않고 `src`의 함수를 호출합니다.

## `src/config.py`

- 공통 경로
- Adult 데이터 컬럼
- 변수 타입
- 분석 가능 변수
- 예측 피처
- Random seed

를 관리합니다.

## `src/data.py`

Adult 데이터를 로딩하고 공통 정제 DataFrame을 생성합니다.

## `src/eda.py`

개발 및 데이터 품질 점검용 EDA를 수행합니다.

## `src/statistics.py`

- Binary group association
- Propensity Score Matching
- SMD balance diagnostics
- McNemar test

를 담당합니다.

## `src/association.py`

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
Overall Wald Test
        ↓
Adjusted Odds Ratio
        ↓
Adjusted Probability
        ↓
Optional PSM
```

을 담당합니다.

## `src/modeling.py`

- HistGradientBoosting 학습
- Sigmoid Probability Calibration
- Classification threshold 적용
- 모델 평가
- 모델 bundle 저장
- 사용자 입력 예측
- 개인별 예측 설명
- What-if Simulation
- Permutation Importance
- Fairness diagnostics

를 담당합니다.

## `src/visualization.py`

사용자용 Plotly Figure를 생성합니다.

분석 결과를 다시 계산하지 않고 `association.py`, `modeling.py` 결과를 직접 시각화합니다.

## `src/model_visualization.py`

개발자용 모델 평가 시각화를 담당합니다.

- Performance Metrics
- ROC Curve
- Confusion Matrix

를 PNG로 저장합니다.

## `main.py`

개발·검증용 CLI 진입점입니다.

실제 웹 사용자는 `app.py`를 통해 서비스를 사용합니다.

---

# 22. 설치

Python 3.11 환경을 기준으로 개발했습니다.

```bash
pip install -r requirements.txt
```

---

# 23. 개발 파이프라인 실행

전체 실행:

```bash
python main.py --stage all
```

개별 실행:

```bash
python main.py --stage data
python main.py --stage eda
python main.py --stage model
python main.py --stage model-viz
```

---

# 24. 웹 애플리케이션 실행

먼저 배포용 모델이 존재해야 합니다.

```text
outputs/models/income_model_bundle.joblib
```

실행:

```bash
streamlit run app.py
```

브라우저에서:

```text
연관성 분석
고소득 예측
```

두 기능을 사용할 수 있습니다.

---

# 25. 검증

## 기본 Smoke Test

```text
[PASS] continuous association
[PASS] categorical association
[PASS] binary association + PSM
[PASS] individual prediction
[PASS] what-if
[PASS] global feature importance
```

## Version 1.1

추가 검증:

- Logistic → GLM fallback
- Overall Wald Test
- Adjusted Probability

## Version 1.2

추가 검증:

```text
Model Retuning
→ 기존 파라미터와 튜닝 후보 비교

Probability Calibration
→ Uncalibrated / Sigmoid / Isotonic 비교

Classification Threshold
→ 0.05 ~ 0.95 탐색

Final Evaluation
→ calibrated probability
→ classification metrics
→ Brier Score
→ Log Loss
```

---

# 26. 주요 출력 파일

## 배포 모델

```text
outputs/models/income_model_bundle.joblib
```

포함 내용:

- 최종 calibrated estimator
- Prediction input schema
- Classification threshold
- Calibration method
- Model card
- Global permutation importance

## 개발용 모델 평가

```text
outputs/tables/model_metrics.json
outputs/tables/model_predictions.csv
outputs/tables/model_feature_importance.csv
outputs/tables/model_fairness_by_group.csv
outputs/tables/model_input_schema.json
outputs/tables/model_card.json
```

## Version 1.2 실험 결과

```text
model_tuning_v1_2.*
model_calibration_v1_2.*
model_threshold_v1_2.*
```

모델 선택 과정과 판단 근거를 재현하기 위한 결과입니다.

---

# 27. 해석상의 주의사항

## 연관성은 인과관계가 아닙니다

Adult Census Income은 관찰 데이터입니다.

관측된 변수를 통제하더라도 데이터에 없는 교란요인은 통제할 수 없습니다.

따라서 Logistic Regression, GLM, PSM 결과는 확정적인 인과효과로 해석하지 않습니다.

## Adjusted Probability도 인과효과가 아닙니다

Adjusted Probability는 조정된 통계모형의 평균 예측확률입니다.

## 머신러닝 예측도 인과관계가 아닙니다

Feature Importance, 개인별 설명, What-if 결과는 모두 모델의 예측 구조를 설명하는 것이며 실제 인과효과를 의미하지 않습니다.

## Calibrated Probability도 실제 미래를 보장하지 않습니다

Probability Calibration은 예측확률과 관측 비율의 일치도를 개선합니다.

그러나 개별 사용자의 실제 소득 발생 확률을 보장하거나 미래의 소득을 확정적으로 예측하는 것은 아닙니다.

## 데이터의 한계

Adult Census Income은 과거 미국 Census 기반 데이터입니다.

현재 특정 국가, 노동시장 또는 개인에게 그대로 일반화할 수 없습니다.

`sex`, `race` 관련 결과 역시 집단의 본질적인 능력 차이로 해석해서는 안 됩니다.

---

# 28. 프로젝트 핵심 원칙

1. 연관성 분석과 머신러닝 예측을 분리한다.
2. 관심 변수와 통제 변수의 역할을 구분한다.
3. 통제 변수는 특정 값으로 고정하지 않고 통계적으로 조정한다.
4. 조정 전 결과와 조정 후 결과를 구분한다.
5. 통계 결과와 시각화는 동일한 분석 표본을 사용한다.
6. Logistic Regression 및 GLM 결과를 인과효과로 표현하지 않는다.
7. PSM 결과 역시 확정적인 인과효과로 표현하지 않는다.
8. 범주형 관심 변수는 전체 효과와 개별 범주 효과를 구분한다.
9. Adjusted Probability와 관측 고소득률을 구분한다.
10. Adjusted Probability와 머신러닝 개인 예측 확률을 구분한다.
11. 불안정한 Logistic Regression 결과를 억지로 생성하지 않는다.
12. 모델 재튜닝 결과가 실질적으로 개선되지 않으면 기존 모델을 유지한다.
13. 예측확률의 순위 성능과 확률 calibration을 별도로 검증한다.
14. 명확한 업무 비용 기준 없이 classification threshold를 임의 최적화하지 않는다.
15. Feature Importance와 What-if를 인과효과로 표현하지 않는다.
16. 개발용 모델 진단과 사용자용 시각화를 분리한다.
17. 웹 UI에서 통계 및 머신러닝 로직을 중복 구현하지 않는다.

---

# 29. 향후 계획

## Version 1.3 — GUI / UX Refinement

분석 및 예측 기능이 충분히 안정화된 이후 사용자 인터페이스를 정리합니다.

주요 계획:

- 전체 정보 구조 정렬
- 연관성 분석 결과 섹션 정리
- 예측 결과 섹션 정리
- 카드 / 여백 / 버튼 스타일 통일
- Plotly 그래프 스타일 통일
- 변수명 표현 방식 통일
- 오류 및 경고 메시지 정리
- 반응형 화면 점검

## 이후

- 자동 테스트 범위 확대
- 문서 최종 정리
- 배포 환경 구성
- 서비스 안정성 검증

---

# 30. 사용 데이터

**Adult Census Income**

Target:

```text
high_income
```

정의:

```text
income > 50K  → 1
income <= 50K → 0
```

본 프로젝트는 실제 연소득 금액을 예측하는 회귀 문제가 아닙니다.

**연 소득 50,000달러 초과 여부의 연관성을 분석하고, 해당 여부를 예측하는 이진 결과 분석·분류 프로젝트입니다.**