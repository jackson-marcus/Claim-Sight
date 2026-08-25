"""Root conftest: stub lightgbm with sklearn-backed estimators.

On Windows dev/CI environments where lib_lightgbm.dll is unavailable,
we replace lightgbm classes with sklearn GradientBoosting equivalents
that are picklable and actually learn from data.
"""

import sys
import types


def _try_real_lightgbm():
    try:
        import lightgbm  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


if not _try_real_lightgbm():
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    class LGBMClassifier(GradientBoostingClassifier):
        def __init__(self, n_estimators=100, verbose=0, random_state=None, **kw):
            super().__init__(n_estimators=n_estimators, random_state=random_state, verbose=0)
            self._lgbm_verbose = verbose

    class LGBMRegressor(GradientBoostingRegressor):
        def __init__(
            self, n_estimators=100, verbose=0, random_state=None, objective=None, alpha=0.5, **kw
        ):
            if objective == "quantile":
                super().__init__(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    loss="quantile",
                    alpha=alpha,
                    verbose=0,
                )
            else:
                super().__init__(n_estimators=n_estimators, random_state=random_state, verbose=0)
            self.objective = objective
            self.alpha = alpha
            self._lgbm_verbose = verbose

    class LGBMRanker(GradientBoostingRegressor):
        def __init__(self, n_estimators=100, verbose=0, random_state=None, **kw):
            super().__init__(n_estimators=n_estimators, random_state=random_state, verbose=0)
            self._lgbm_verbose = verbose

    lgb = types.ModuleType("lightgbm")
    lgb.LGBMClassifier = LGBMClassifier
    lgb.LGBMRegressor = LGBMRegressor
    lgb.LGBMRanker = LGBMRanker
    LGBMClassifier.__module__ = "lightgbm"
    LGBMRegressor.__module__ = "lightgbm"
    LGBMRanker.__module__ = "lightgbm"
    sys.modules["lightgbm"] = lgb
