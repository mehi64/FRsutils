from FRsutils.core.owa_weights import OWAWeightStrategy

# Linear weights (default)
owa = OWAWeightStrategy.create("linear")
print(owa.lower_weights(5))  # ascending
print(owa.upper_weights(5))  # descending

# Harmonic weights
owa = OWAWeightStrategy.create("harmonic")
weights = owa.weights(n=7, descending=True)  # same as upper_weights
print(weights)
