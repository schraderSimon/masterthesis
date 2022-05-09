from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, symbols, Symbol
from drudge import RestrictedPartHoleDrudge, Stopwatch


conf = SparkConf().setAppName("rccsd")
ctx = SparkContext(conf=conf)
dr = RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

p = dr.names
e_ = p.e_

a, b, c, d = p.V_dumms[:4]
i, j, k, l = p.O_dumms[:4]

X1 = e_[a, i]
X2 = e_[a, i] * e_[b, j]
Y1 = e_[k,c]
Y2= e_[k,c]*e_[l,d]
stopwatch = Stopwatch()

"""The T* operator"""
t = IndexedBase("t")
rhs = IndexedBase("rhs")
cluster = dr.einst( #The cluster operator
    t[a, i] * e_[a, i] + Rational(1, 2) * t[a, b, i, j] * e_[a, i] * e_[b, j]
)
dr.set_n_body_base(t, 2) #Its symmetry (not antisymmetric, but fucked)
cluster = cluster.simplify()
cluster.cache()
"""The T(alpha_i) operator"""

#### Similarity transform of the Hamiltonian
curr_X1 = (dr.ham | X1).simplify()
sim_H_X1 = (dr.ham | X1).simplify() #Single excitation transform: e^-T[H,t_single]e^T
curr_X2 = (dr.ham | X2).simplify()
sim_H_X2 = (dr.ham | X2).simplify() #Double excitation transform: e^-T[H,t_double]e^T
for order in range(0,4):
    curr_X1 = (curr_X1 | cluster).simplify() * Rational(1, order + 1)
    curr_X2 = (curr_X2 | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator1 order {}".format(order + 1), curr_X1)
    stopwatch.tock("Commutator2 order {}".format(order + 1), curr_X2)
    sim_H_X1 += curr_X1
    sim_H_X2 += curr_X2
sim_H_X1 = sim_H_X1.simplify()
sim_H_X1.repartition(cache=True)
stopwatch.tock("[H,X1]-bar assembly", sim_H_X1)

sim_H_X2 = sim_H_X2.simplify()
sim_H_X2.repartition(cache=True)
stopwatch.tock("[H,X2]-bar assembly", sim_H_X2)


derivative_eq_X1_Y1=dr.define(rhs[k,c,i,a],(Y1*sim_H_X1).eval_fermi_vev().simplify().repartition(cache=True))
stopwatch.tock("derivative_eq_X1_Y1 assembly", derivative_eq_X1_Y1)

derivative_eq_X2_Y1=dr.define(rhs[k,c,i,j,a,b],(Y1*sim_H_X2).eval_fermi_vev().simplify().repartition(cache=True))
stopwatch.tock("derivative_eq_X2_Y1 assembly", derivative_eq_X2_Y1)

derivative_eq_X1_Y2=dr.define(rhs[k,l,c,d,i,a],(Y2*sim_H_X1).eval_fermi_vev().simplify().repartition(cache=True))
stopwatch.tock("derivative_eq_X1_Y2 assembly", derivative_eq_X1_Y2)

derivative_eq_X2_Y2=dr.define(rhs[k,l,c,d,i,j,a,b],(Y2*sim_H_X2).eval_fermi_vev().simplify().repartition(cache=True))
stopwatch.tock("derivative_eq_X2_Y2 assembly", derivative_eq_X2_Y2)


with dr.report("RCCSD_Jacobian.html", "RCCSD Jacobian") as rep:
    rep.add(content=derivative_eq_X1_Y1, description="Commutator expectation value <single|e^-T[H,single]e^T|HF> as 'i,a,j,b'")
    rep.add(content=derivative_eq_X2_Y1, description="Commutator expectation value <single|e^-T[H,double]e^T|HF> as 'i,a,j,k,b,c'")
    rep.add(content=derivative_eq_X2_Y1, description="Commutator expectation value <double|e^-T[H,single]e^T|HF> as 'i,j,a,b,k,c'")
    rep.add(content=derivative_eq_X2_Y2, description="Commutator expectation value <double|e^-T[H,double]e^T|HF> as 'i,j,a,b,k,l,c,d'")

from gristmill import *

printer = EinsumPrinter()
we1=[derivative_eq_X1_Y1]
we2=[derivative_eq_X2_Y1]
we3=[derivative_eq_X1_Y2]
we4=[derivative_eq_X2_Y2]
with open("jacobian_X1Y1.txt", "w") as outfile:
    outfile.write(printer.doprint(we1))
with open("jacobian_X2Y1.txt.txt", "w") as outfile:
    outfile.write(printer.doprint(we2))
with open("jacobian_X1Y2.txt", "w") as outfile:
    outfile.write(printer.doprint(we3))
with open("jacobian_X2Y2.txt.txt", "w") as outfile:
    outfile.write(printer.doprint(we4))
eval_seq_1 = optimize(
    we1, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
eval_seq_2 = optimize(
    we2, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
eval_seq_3 = optimize(
    we3, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
eval_seq_4 = optimize(
    we4, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
with open("jacobian_opt_X1Y1.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_1))
with open("jacobian_opt_X2Y1.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_2))
with open("jacobian_opt_X1Y2.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_3))
with open("jacobian_opt_X2Y2.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_4))
