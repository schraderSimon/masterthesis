from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid


basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(1,1)
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
n=30
x=4*np.random.rand(n,2)+2 #n random numbers between 2 and 6 for x and y directions
sample_geom_new=[]
for i in range(n):
    if (x[i,0]+x[i,1]) <=9 or (x[i,0]+x[i,1])>= 11.5:
        sample_geom_new.append([x[i,0],x[i,1]])
sample_geom=np.concatenate((sample_geom_new,[[5.5,6],[6,5.5],[6,6]]))
print(len(sample_geom))
span=np.linspace(2,6,9)

geom_alphas=[]
for x in span:
    for y in span:
        geom_alphas.append((x,y))
x, y = np.meshgrid(span,span)

#E_FCI=FCI_energy_curve(geom_alphas,basis,molecule,unit="Bohr")
E_FCI=np.load("energy_data/BeH2_2to6_FULLCI.npy")
energy_dict={}
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")

evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC()
WF=np.array(E_WF).reshape((9,9))
FCI=np.array(E_FCI).reshape((9,9))
WFerr=np.abs(WF-FCI)*1000
print(WFerr)
E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.5)
energy_dict["num_samples"]=len(sample_geom)
energy_dict["CCSD"]=CCSD_energy_curve(molecule,geom_alphas,basis)
energy_dict["FCI"]=E_FCI
energy_dict["AMP"]=E_AMP_full
energy_dict["WF"]=E_WF
energy_dict["AMPred"]=E_AMP_red
energy_dict["x"]=x
energy_dict["y"]=y
energy_dict["samples"]=sample_geom
import pickle
file="energy_data/BeH2_2d_plot_NOGUPTRI.bin"

with open(file,"wb") as f:
    pickle.dump(energy_dict,f)

print("Grid")
print(np.meshgrid(span,span))
print("Error in CCSD")
print(E_CCSDx-E_FCI)
CCSDerr=np.abs(E_CCSDx-E_FCI)*1000
WFerr=np.abs(E_approx-E_FCI)*1000
AMPerr=np.abs(energy_simen_full-E_FCI)*1000
AMPranderr=np.abs(energy_simen_reduced-E_FCI)*1000
print("Error in WF-EVC")
print(E_approx-E_FCI)
print("Error in amp-evc")
print(energy_simen_full-E_FCI)
z_min=0
z_max=np.amax((CCSDerr,WFerr,AMPerr,AMPranderr))
fig,grid = plt.subplots(2,2)
im0=grid[0,0].pcolormesh(x, y, CCSDerr, cmap='PuRd',shading='auto',vmin=z_min,vmax=z_max)
grid[0,0].set_title("CCSD error")
grid[0,0].set_xlabel("distance H2-Be (Bohr)")
grid[0,0].set_ylabel("distance H1-Be (Bohr)")
grid[0,0].scatter(sample_geom[:,0],sample_geom[:,1])
plt.legend()

grid[0,1].pcolormesh(x, y, WFerr, cmap='PuRd',shading='auto',vmin=z_min,vmax=z_max)
im1=grid[0,1].set_title("WF-EVC error")
grid[0,1].scatter(sample_geom[:,0],sample_geom[:,1])
grid[0,1].set_xlabel("distance H2-Be (Bohr)")
grid[0,1].set_ylabel("distance H1-Be (Bohr)")
plt.legend()

im2=grid[1,0].pcolormesh(x, y, AMPerr, cmap='PuRd',shading='auto',vmin=z_min,vmax=z_max)
grid[1,0].set_title("AMP error")
grid[1,0].scatter(sample_geom[:,0],sample_geom[:,1])
grid[1,0].set_xlabel("distance H2-Be (Bohr)")
grid[1,0].set_ylabel("distance H1-Be (Bohr)")

im2=grid[1,1].pcolormesh(x, y, AMPranderr, cmap='PuRd',shading='auto',vmin=z_min,vmax=z_max)
grid[1,1].set_title("Randomized AMP error")
grid[1,1].scatter(sample_geom[:,0],sample_geom[:,1])
grid[1,1].set_xlabel("distance H2-Be (Bohr)")
grid[1,1].set_ylabel("distance H1-Be (Bohr)")

plt.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
colorbar=fig.colorbar(im2, cax=cbar_ax,label='Error (mHartree)')
plt.savefig("BeH2_stretch_Procrustes_1.pdf")
plt.show()
