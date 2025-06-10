import FRsutils.core.similarities as sim
import FRsutils.core.tnorms as tn
import tests.synthetic_data_store as sds
import numpy as np

tnrm = tn.TNorm.create('min')
sim1 = sim.Similarity.create('gaussian', strict=False, sigma=0.67)
params = sim1.describe_params_detailed()
hlp =sim1.help()
print(hlp)
nme = sim1.name
dic1 = sim1.to_dict()
similarity_instance = sim.Similarity.from_dict(dic1)

ds = sds.get_similarity_testing_testsets()

X = ds[0]['X']
vals = ds[0]['expected']

sim_mat = sim.calculate_similarity_matrix(
    X,
    sim1,
    tnrm)

print(sim_mat)


