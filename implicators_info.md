# 🧠 Fuzzy Implicators: Overview and Properties

This document provides a list of commonly used fuzzy implicators, their formulae, valid parameter ranges, and references.

## 📘 What Are Fuzzy Implicators?

In **fuzzy logic**, an **implicator** is a function  

**I: [0,1]² → [0,1]**

used to generalize classical logical implication for fuzzy sets. Implicators are widely used in fuzzy inference systems, fuzzy control, and approximate reasoning.

---

## ✅ Properties of Fuzzy Implicators

| Property                   | Symbol / Definition |
|---------------------------|----------------------|
| **I1. Boundary Conditions** | `I(0, 0) = I(1, 1) = 1, I(1, 0) = 0` |
| **I2. Monotonicity**       | `b₁ ≤ b₂ => I(a, b₁) ≤ I(a, b₂)`<br>`a₁ ≤ a₂ => I(a₁, b) ≥ I(a₂, b)` |
| **I3. Exchange Principle** | `I(a, b) = I(1 - b, 1 - a)` |
| **I4. Identity**           | `I(a, 1) = 1` |
| **I5. Contrapositive Symmetry** | `I(a, b) = I(1 - b, 1 - a)` |
| **I6. Classical Implication** | `a ≤ b => I(a, b) = 1` |


> ✅ Most practical fuzzy systems expect I1, I2, I4, and I6 at minimum.

---


## Implicators' equation

<!--Gaines -->
<img src="images/implicators/eq_imp_gaines.JPG" alt="eq_imp_gaines" width="600"/>

------

<!--Gödel -->
<img src="images/implicators/eq_imp_goedel.JPG" alt="eq_imp_goedel" width="600"/>

------

<!--Kleene–Dienes-->
<img src="images/implicators/eq_imp_kd.JPG" alt="eq_imp_kd" width="600"/>

------

<!--Reichenbach-->
<img src="images/implicators/eq_imp_reichenbach.JPG" alt="eq_imp_Reichenbach" width="600"/>

-----

<!--Lukasiewicz-->
<img src="images/implicators/eq_imp_luk.JPG" alt="eq_imp_luk" width="600"/>

-----

<!--Yager-->
<img src="images/implicators/eq_imp_yager.JPG" alt="eq_imp_yager" width="600"/>

-----

<!--Weber-->
<img src="images/implicators/eq_imp_weber.JPG" alt="eq_imp_weber" width="600"/>

-----

<!--Frank-->
<img src="images/implicators/eq_imp_frank.JPG" alt="eq_imp_frank" width="600"/>

-----

<!--Sugeno–Weber-->
<img src="images/implicators/eq_imp_sugeno_weber.JPG" alt="eq_imp_sugeno_weber" width="600"/>

-----

## 📊 Implicator Properties Table

| Implicator             | I1 | I2 | I3 | I4 | I5 | I6 |
|------------------------|:--:|:--:|:--:|:--:|:--:|:--:|
| **Gaines**             | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Gödel**              | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Kleene-Dienes**      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Reichenbach**        | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Lukasiewicz**        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Yager (p = 2)**      | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Weber**              | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Frank (s = 2)**      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Sugeno–Weber (λ = 0)** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |

---

## 📎 Notes

- ✅ = Property is satisfied  
- ❌ = Property is not satisfied  
- These results are generally true for all values of parameters (where applicable), assuming valid ranges:
  - Frank: $s > 0, s \ne 1$
  - Yager: $p > 0$
  - Sugeno–Weber: $\lambda \geq -1$

------


## References (must be checked)

|Implicator	|Reference|
|------|------|
|Gaines|Gaines, B. R. (1978). Fuzzy and probability uncertainty logics. Information and Control, 38(2).|
|Gödel|Gödel, K. (1932). Zum intuitionistischen Aussagenkalkül. (Defined in t-norm form later.)|
|Kleene-Dienes|Kleene, S. C. (1952). Introduction to Metamathematics (linked with Dienes' psychological logic).|
|Reichenbach|Reichenbach, H. (1944). Philosophic Foundations of Quantum Mechanics|
|Lukasiewicz|Łukasiewicz, J. (1920). On Three-Valued Logic|
|Yager|Yager, R. R. (1980). On a general class of fuzzy connectives. Fuzzy Sets and Systems|
|Weber|Weber, S. (1983). A general concept of fuzzy connectives, negations, and implications. FSS|
|Frank|Frank, M. J. (1979). On the simultaneous associativity of F(x, y) and x+y−F(x, y). Aequationes Math|
|Sugeno–Weber|Sugeno, M. & Weber, M. (1986). A new approach to fuzzy reasoning. IJGS|