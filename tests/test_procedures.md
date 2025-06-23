# Test procedures

This document provides information on test procedures of FRsutils framework. 

## 1. Pytest

We use pytest to test FRsutils framework.

---

## 2. General information about testing

  -Class RegistryFactoryMixin is not tested alone. It is tested with all classes inheriting that.
  - When testing, for some functions, the logger logs into test_logs.csv. To make sure all are corretc, you can check the outputs.
  - Classes used as base, as well as mixins are not tested separately. they got tested within derived classes.
  
## 3. Testing core classes (summary of tests performed)
  - TNorms: See tests/unittest_summary/unittest_tnorms_summary.xlsx
  - Implicators: See   tests/unittest_summary/unittest_implicators_summary.xlsx


## 4. Testing Fuzzy Quantifiers
  - Correctness of outputs
    - Mainly by visualization of different values
    - using unit tests

