# Fuzzy-Rough utilities (Under development)

A basic Python library needed for fuzzy rough set calculations e.g.:

- lower approximation
- upper approximation
- positive region

## Algorithgms and containings

- Implicators
  - Gaines
  - Goedel
  - Kleene–Dienes
  - Reichenbach
  - Lukasiewicz
- T-norms
  - min tnorm
  - product tnorm
- OWAFRS (Ordered Weighted Average Fuzzy-Rough Sets) 
- VQRS (Vaguely Quantified Rough Sets)
- ITFRS (Implicator/T-norm Fuzzy-Rough Sets)

## Notes
- All functions expect to get normalized scalar of normalized numpy arrays.
- Make sure the input dataset is normalized. This library expects all inputs to all functions are in range [0,1]
- This library will use all features of data instances to calculate the fuzzzy-rough measures.


## Some technical information to remember
### tnorms
- works on 1D vectors (for aggregating the values to a scalar in similarity calculations)
- works on nxnx2 maps for fast calculations
- min tnorm and product tnorm will act the same if one of the input parameters takes the binary values; So, no matter which one you use, they provide the same reults
- implicators work on scalar but can be vectorized with np.vectorize()

## How to Cite

If you use this library in your research, please cite it as follows:

**APA** (adjust to your preferred style):  
> Mehran Amiri. (*2025r*). *FRutils* (Version 0.0.1) [Computer software]. https://github.com/mehi64/FRutils

**BibTeX** (for LaTeX users):
```bibtex
@software{Mehran_Amiri_FRutils_2025,
  author = {Mehran_Amiri},
  title = {FRutils},
  url = {https://github.com/mehi64/FRutils},
  version = {0.0.1},
  year = {2025}
}

#TODO:
- Add tests for tnorms with non-binary masks