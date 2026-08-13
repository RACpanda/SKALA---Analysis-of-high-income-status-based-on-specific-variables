from src.data import load_and_clean

from src.association import (
    AnalysisRequest,
    analyze_association,
)

from src.modeling import (
    get_global_feature_importance,
    get_prediction_input_schema,
    predict_income_input,
    simulate_income_what_if,
)

from src.visualization import (
    create_association_visualizations,
    plot_global_feature_importance,
    plot_prediction_probability,
    plot_prediction_explanation,
    plot_what_if_simulation,
)


df = load_and_clean()


# ============================================================
# 1. Continuous association
# ============================================================

age_request = AnalysisRequest(
    exposure="age",
    controls=(
        "sex",
        "education",
    ),
)

age_result = analyze_association(
    df,
    age_request,
)

age_figures = (
    create_association_visualizations(
        age_result
    )
)

print(
    "[PASS] continuous association"
)


# ============================================================
# 2. Categorical association
# ============================================================

education_request = AnalysisRequest(
    exposure="education",
    controls=(
        "age",
        "sex",
    ),
)

education_result = (
    analyze_association(
        df,
        education_request,
    )
)

education_figures = (
    create_association_visualizations(
        education_result
    )
)

print(
    "[PASS] categorical association"
)


# ============================================================
# 3. Binary association + PSM
# ============================================================

sex_request = AnalysisRequest(
    exposure="sex",
    controls=(
        "age",
        "education",
        "hours-per-week",
    ),
    include_psm=True,
)

sex_result = (
    analyze_association(
        df,
        sex_request,
    )
)

sex_figures = (
    create_association_visualizations(
        sex_result
    )
)

print(
    "[PASS] binary association + PSM"
)


# ============================================================
# 4. Prediction input
# ============================================================

schema = (
    get_prediction_input_schema()
)

user_input = {
    feature: info[
        "reference_value"
    ]
    for feature, info
    in schema[
        "features"
    ].items()
}

prediction_result = (
    predict_income_input(
        user_input
    )
)

prediction_figure = (
    plot_prediction_probability(
        prediction_result
    )
)

explanation_figure = (
    plot_prediction_explanation(
        prediction_result
    )
)

print(
    "[PASS] individual prediction"
)


# ============================================================
# 5. What-if
# ============================================================

what_if = (
    simulate_income_what_if(
        user_input,
        feature="age",
    )
)

what_if_figure = (
    plot_what_if_simulation(
        what_if
    )
)

print(
    "[PASS] what-if"
)


# ============================================================
# 6. Global feature importance
# ============================================================

importance = (
    get_global_feature_importance()
)

importance_figure = (
    plot_global_feature_importance(
        importance
    )
)

print(
    "[PASS] global feature importance"
)


print(
    "\nALL SMOKE TESTS PASSED"
)