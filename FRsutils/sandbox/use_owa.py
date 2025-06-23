from FRsutils.core.owa_weights import OWAWeightStrategy
import numpy as np

# # Linear weights (default)
# owa = OWAWeightStrategy.create("linear")
# print(owa.lower_weights(5))  # ascending
# print(owa.upper_weights(5))  # descending

# exponential weights
owa = OWAWeightStrategy.create("exp")
weights = owa.weights(n=10, descending=False)  # same as upper_weights
print(weights)

print(np.sum(weights))

# # Harmonic weights
# owa = OWAWeightStrategy.create("harmonic")
# weights = owa.weights(n=7, descending=True)  # same as upper_weights
# print(weights)
