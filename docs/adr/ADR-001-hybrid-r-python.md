# ADR-001: Hybrid R extract / Python model

**Context.** fitzRoy is the best-maintained AFL data package and has no equal in Python. The Melbourne
data-science market overwhelmingly asks for Python, and SHAP, XGBoost and Streamlit deploy most easily there.

**Decision.** R only extracts data (and renders one analysis report that reads processed tables). Everything
else is Python. The halves talk through parquet files, not `reticulate`.

**Consequences.** + Simple, robust deploys; either half can be replaced independently; CI needs no R.
− Two toolchains (renv + pip). The Quarto report is the only place R touches analysis, deliberately kept out
of the prediction path.
