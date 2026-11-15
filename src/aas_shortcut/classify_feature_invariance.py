"""Classifying features by what they do with objective values."""

from __future__ import annotations

from aas_shortcut.calculate_deep_ela_features import is_deep_ela_feature

# Feature groups defined only by comparisons of objective values, hence invariant
# under any strictly increasing transformation by construction.
COMPARISON_BASED_GROUPS: tuple[str, ...] = (
    "disp.",
    "ela_level.",
    "nbc.",
)

# Individual features, rather than whole groups, that are comparison-based.
COMPARISON_BASED_FEATURES: tuple[str, ...] = (
    "pca.expl_var.cov_x",
    "pca.expl_var.cor_x",
    "pca.expl_var_PC1.cov_x",
    "pca.expl_var_PC1.cor_x",
    "fitness_distance.distance_mean",
    "fitness_distance.distance_std",
    # At eps = 0 the information-content symbols are the signs of consecutive
    # objective differences, which a strictly increasing transformation preserves.
    # The other `ic.*` features threshold |dy / dx| against a positive eps.
    "ic.m0",
)


def is_structurally_invariant(feature: str) -> bool:
    """Whether a feature is comparison-based by definition."""
    if is_deep_ela_feature(feature):
        return False
    if feature == "nbc.nb_fitness.cor":
        return False
    if feature in COMPARISON_BASED_FEATURES:
        return True
    return any(feature.startswith(prefix) for prefix in COMPARISON_BASED_GROUPS)
