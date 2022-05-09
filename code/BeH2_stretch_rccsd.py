from rccsd_gs import *
basis = 'STO-3G'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule =lambda arr: "Be 0.0 0.0 0.0; H 0.0 0.0 %f; H 0.0 0.0 -%f"%(arr,arr)
molecule=lambda x:  "N 0 0 0; N 0 0 %f"%x
refx=[2]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(1.5,3.0,10)
#sample_geom1=[2.5,3.0,6.0]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=np.linspace(1.5,6.0,20)
geom_alphas=[[x] for x in geom_alphas1]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
print(t1s[0].shape)
print(t2s[0].shape)
print(l1s[0].shape)
print(l2s[0].shape)
energy_simen=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,type="procrustes")

E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
plt.plot(geom_alphas1,E_CCSDx,label="CCSD")
plt.plot(geom_alphas1,E_approx,label="CCSD WF")
plt.plot(geom_alphas1,energy_simen,label="CCSD AMP")
plt.legend()
plt.show()
