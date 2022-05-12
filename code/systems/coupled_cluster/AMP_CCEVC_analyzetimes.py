import sys
sys.path.append("../libraries")
from func_lib import *
file="energy_data/BeH2_time.bin"
#file="energy_data/HF_time.bin"

import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
times=data["times"]
niter=data["niter"]
projection_errors=data["projection_errors"]
xval=data["xval"]
virtvals=data["virtvals"]
#virtvals=[1,0.8,0.6,0.4,0.2]
for i in range(len(virtvals)):
    time_mean=np.mean(times[i])
    proj_max=np.max(np.abs(projection_errors[i]))
    niter_mean=np.mean(niter[i])
    time_std=np.std(times[i])
    #proj_std=np.std(projection_errors[i])
    niter_std=np.std(niter[i])
    print(virtvals[i])
    print("Time: %f \pm %f"%(time_mean,time_std))
    print("niter: %f \pm %f"%(niter_mean,niter_std))
    print("proj: %f"%(proj_max))#,proj_std))
    #print(projection_errors[i])
