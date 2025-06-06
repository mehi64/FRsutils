import FRsutils.core.tnorms as tn
import numpy as np

tnorm1 = tn.TNorm.create('minimum')
tnorm2 = tn.TNorm.create('prod')
tnorm3 = tn.TNorm.create('yager', strict=False, p=2, f=0.8)

arr1 = np.array([0.7, 0.4])
arr2 = np.array([0.9, 0.5])

print(tnorm1(arr1, arr2))      # min
print(tnorm2(arr1, arr2))      # product
print(tnorm3(arr1, arr2))      # yager with p=3.0

print(tn.TNorm.list_available())