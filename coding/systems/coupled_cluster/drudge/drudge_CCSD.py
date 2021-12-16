from pyspark import SparkContext
ctx=SparkContext("local[*]","ccsd")
from dummy_spark import SparkContext
ctx=SparkContext()

from sympy import *
from drudge import *

dr=PartHoleDrudge(ctx) #Particle-hole formalism from drudge
dr.full_simplify=False #Hamiltonian not fully simplified
p=dr.names

c=p.c_ #Annihilation operator
c_dag=p.c_dag #Creation operator
a,b=p.V_dumms[:2]
i,j=p.O_dumms[:2]

t=IndexedBase("t") #A general, indexed tensor, might have two, might have four...
clusters=dr.einst(t[a,i]*c_dag[a]*c[i]+t[a,b,i,j]*c_dag[a]*c_dag[b]*c[j]*c[i]/4) #Define the cluster operator

clusters.display()

dr.set_dbbar_base(t,2) #Antiymmetry of the t_{abij} elements

curr=dr.ham #The Hamiltonian of the Particle-Hole system (I guess it's the default many-body Hamiltonian)
h_bar=dr.ham
print(h_bar)
#Create the similarity tranformed Hamiltonian
for order in range(0,4):
    curr=(curr | clusters).simplify()/(order+1)
    curr.cache()
    h_bar+= curr
h_bar.repartition(cache=True)
print(h_bar)

energy_equation=h_bar.eval_fermi_vev().simplify #Calculate the fermi vaccuum: <vac|Hbar|vac>

t1_projector=c_dag[i]*c[a] #apply on the left this little bitch <3

t1_eqn=(t1_projector*h_bar).eval_fermi_vev().simplify
print(t1_eqn)

t2_projector=c_dag[i]*c_dag[j]*c[b]*c[a]
t2_eqn=(t2_projector*h_bar).eval_fermi_vev().simplify
print(t2_eqn)

from gristmill import *

working_eqn=[dr.define(Symbol("e"),en_eqn),dr.define(t[a,i],t1_eqn),dr.define(t[a,b,i,j],t2_eqn)]
