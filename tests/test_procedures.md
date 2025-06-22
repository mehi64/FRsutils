# Test procedures

This document provides information on test procedures of FRsutils framework. 

## 1. Pytest

We use pytest to test FRsutils framework.

---

## 2. General information about testing

  -Class RegistryFactoryMixin is not tested alone. It is tested with all classes inheriting that.
  
## 3. Testing Implicators
# 🧪 Unit Test Coverage: test_implicators.py

<table style="width:100%">
  <thead>
    <tr>
      <th>Test Function</th>
      <th>What It Tests</th>
      <th>Targeted Components</th>
      <th>How It Works</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>test_scalar_call_valid</code></td>
      <td>Scalar input support for implicators</td>
      <td><code>Implicator.__call__</code>, <code>_compute_scalar</code></td>
      <td>Calls implicator with <code>a=0.73</code>, <code>b=0.18</code>, checks output ∈ [0,1]</td>
    </tr>
    <tr>
      <td><code>test_implicator_call_vector_output</code></td>
      <td>Vectorized output correctness</td>
      <td><code>Implicator.__call__</code>, <code>_compute_scalar</code></td>
      <td>Uses test vectors from <code>get_implicator_scalar_testsets()</code> and compares results</td>
    </tr>
    <tr>
      <td><code>test_implicator_exhaustive_grid_no_exception</code></td>
      <td>Stability over [0,1] × [0,1]</td>
      <td><code>Implicator.__call__</code></td>
      <td>Evaluates implicator on full 1001×1001 meshgrid and confirms no exceptions occur</td>
    </tr>
    <tr>
      <td><code>test_equivalence_of_constructor_create_fromdict_with_synthetic_data</code></td>
      <td>Consistency across manual, factory, and serialized creation</td>
      <td><code>create</code>, <code>from_dict</code>, <code>to_dict</code>, <code>__call__</code></td>
      <td>Creates three instances and applies them to test vectors to confirm identical outputs</td>
    </tr>
    <tr>
      <td><code>test_equivalence_of_constructor_create_fromdict_with_random_data</code></td>
      <td>Same as above, using random input</td>
      <td>Same as above</td>
      <td>Applies implicator to 1,000,000 random values in [0,1], checks output equivalence</td>
    </tr>
    <tr>
      <td><code>test_implicator_instances_are_distinct</code></td>
      <td>Ensures unique object identity</td>
      <td>Constructor, <code>create</code>, <code>from_dict</code></td>
      <td>Confirms the three created objects have different <code>id()</code> memory addresses</td>
    </tr>
    <tr>
      <td><code>test_create_and_to_dict_from_dict_roundtrip</code></td>
      <td>Serialization structure validation</td>
      <td><code>create</code>, <code>to_dict</code>, <code>from_dict</code></td>
      <td>Checks if keys <code>type</code>, <code>name</code>, <code>params</code> exist and object is reconstructable</td>
    </tr>
    <tr>
      <td><code>test_describe_params_detailed_keys</code></td>
      <td>Parameter reflection</td>
      <td><code>describe_params_detailed</code>, <code>_get_params</code></td>
      <td>Checks that detailed description contains all values from <code>_get_params()</code></td>
    </tr>
    <tr>
      <td><code>test_registry_get_class_and_name</code></td>
      <td>Registry and alias correctness</td>
      <td><code>get_class</code>, <code>get_registered_name</code></td>
      <td>Retrieves class and registered name for each implicator from the registry</td>
    </tr>
    <tr>
      <td><code>test_help</code></td>
      <td>Class documentation availability</td>
      <td><code>help()</code></td>
      <td>Calls <code>help()</code> to get class-level docstring and checks it is non-empty</td>
    </tr>
  </tbody>
</table>

