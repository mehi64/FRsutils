# OWA Weighting Strategies

This document provides a comprehensive overview of common **Ordered Weighted Averaging (OWA)** weight strategies used in fuzzy rough set models. These strategies are essential in computing **fuzzy infimum** and **fuzzy supremum** approximations.

---

## 1. Overview of OWA Strategies

OWA strategies produce normalized weight vectors of length `n` based on predefined mathematical rules. They are used in fuzzy logic models to **aggregate** values with positional importance, commonly for approximating lower and upper bounds.

Each strategy defines two key functions:

- **Lower weights**: For fuzzy **infimum**.
- **Upper weights**: For fuzzy **supremum**.

---

## 2. OWA Strategy Table

| Name         | Formula Example (Lower)                         | Parameters        | Aliases         |
|--------------|--------------------------------------------------|-------------------|-----------------|
| **Linear**   | w_i = 2i / (n(n+1))                              | None              | linear          |
| **Exponential** | w_i ∝ base^i                                 | base > 1          | exponential, exp|
| **Harmonic** | w_i ∝ 1 / i                                      | None              | harmonic, harm  |
| **Logarithmic** | w_i ∝ log(i + 1)                             | None              | logarithmic, log|

All weights are normalized so that sum(w_i) = 1.

---

## 3. Notes

- Each strategy defines both `lower_weights(n)` and `upper_weights(n)` methods.
- A unified `.weights(n, descending=True)` API is available on all strategies.
- These strategies are **pluggable** and registered under `OWAWeightStrategy` using the Registry pattern.

---

## 4. References

1. Yager, R. R. (1988). "On ordered weighted averaging aggregation operators in multicriteria decisionmaking". *IEEE Transactions on Systems, Man, and Cybernetics*, 18(1), 183–190.
2. Torra, V. (1997). "The weighted OWA operator". *International Journal of Intelligent Systems*, 12(2), 153–166.
3. Wikipedia: [Ordered Weighted Averaging Operators](https://en.wikipedia.org/wiki/Ordered_weighted_averaging_operator)

---