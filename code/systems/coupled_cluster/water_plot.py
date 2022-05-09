import sys
sys.path.append("../libraries")

np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.optimize import minimize, root,newton
file="energy_data/water_631G*.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)

CCSD_vals=data["CCSD"]
WF_vals=np.array(data["WF"])
AMP_vals=data["AMP"]
AMP_20=data["AMP20"]
AMP_50=data["AMP50"]
number_repeats=len(CCSD_vals)
number_samples=len(WF_vals[0])
WF_errors=[[] for i in range(0,number_repeats)]
AMP_errors=[[] for i in range(0,number_repeats)]
AMP_50_errors=[[] for i in range(0,number_repeats)]
AMP_20_errors=[[] for i in range(0,number_repeats)]
for n in range(number_repeats):
    for i in range(number_samples):
        WF_errors[n].append(np.abs(WF_vals[n][i]-CCSD_vals[n]))
        AMP_errors[n].append(np.abs(AMP_vals[n][i]-CCSD_vals[n]))
        AMP_50_errors[n].append(np.abs(AMP_50[n][i]-CCSD_vals[n]))
        AMP_20_errors[n].append(np.abs(AMP_20[n][i]-CCSD_vals[n]))
nanval=100
error_means=np.zeros((number_samples,4))
for i in range(number_samples):
    for n in range(number_repeats):
        pass
        #WF_errors[n][i]=np.nan_to_num(WF_errors[n][i],nan=nanval)
        #AMP_errors[n][i]=np.nan_to_num(AMP_errors[n][i],nan=nanval)
        #AMP_50_errors[n][i]=np.nan_to_num(AMP_50_errors[n][i],nan=nanval)
        #AMP_20_errors[n][i]=np.nan_to_num(AMP_20_errors[n][i],nan=nanval)
WF_errors=np.array(WF_errors)
AMP_errors=np.array(AMP_errors)
AMP_50_errors=np.array(AMP_50_errors)
AMP_20_errors=np.array(AMP_20_errors)
from scipy.stats import mstats

types=["WF","AMP100","AMP50","AMP20"]
errors_collected=[WF_errors,AMP_errors,AMP_50_errors,AMP_20_errors]
means=[]
stds=[]
medians=[]
quantile_75=[]
quantile_25=[]
quantile_90=[]
for error in errors_collected:
    medians.append(np.nanquantile(error,q=0.5,axis=(0,2)))
    quantile_25.append(np.nanquantile(error,q=0.25,axis=(0,2)))
    quantile_75.append(np.nanquantile(error,q=0.75,axis=(0,2)))
    quantile_90.append(np.nanquantile(error,q=0.90,axis=(0,2)))
    means.append(np.nanmean(error,axis=(0,2)))
    stds.append(np.nanstd(error,axis=(0,2)))
xticks=[1,3,5,7,9,11,13,15,17]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
axes[0][0].set_ylabel("Dev. from CCSD (Hartree)")
axes[1][0].set_ylabel("Dev. from CCSD (Hartree)")
axes[1][0].set_xlabel("Number of sample points")
axes[1][1].set_xlabel("Number of sample points")
axes[1][1].set_xticks(xticks)
xaxis=np.arange(1,number_samples+1)
titles=["WF-CCEVC","AMP-CCEVC",r"AMP-CCEVC ($p_v=50\%$)",r"AMP-CCEVC ($p_v=20\%$)"]
for i in range(2):
    for j in range(2):
        axes[i][j].set_title(titles[2*i+j])
        axes[i][j].fill_between(xaxis,2*1e-5,1.6*1e-3,color="tab:green",alpha=0.3,label="chemical acc.")
        axes[i][j].plot(xaxis,means[2*i+j],color="tab:blue",label="mean error")
        axes[i][j].plot(xaxis,medians[2*i+j],color="tab:red",label="median error")
        axes[i][j].plot(xaxis,quantile_90[2*i+j],color="tab:orange",label=r"$90\%$ quantile")
        axes[i][j].fill_between(xaxis,quantile_25[2*i+j],quantile_75[2*i+j],color="tab:red",alpha=0.5,label="IQR")

        axes[i][j].set_yscale("log")
handles, labels = axes[-1][-1].get_legend_handles_labels()

fig.legend(handles, labels, bbox_to_anchor=(0.88,0.87),loc="center",handletextpad=0.3,labelspacing = 0.1)
fig.tight_layout()
fig.subplots_adjust(right=0.85)
plt.savefig("resultsandplots/water.pdf")
plt.show()
