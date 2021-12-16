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
Y1 = e_[i,a]
Y2= e_[i,a]*e_[j,b]
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
tiT=IndexedBase("tiT")
rhs = IndexedBase("rhs")
Tidagger = dr.einst( #The cluster operator
    tiT[i,a] * e_[i, a] + Rational(1, 2) * tiT[i,j,a,b] * e_[i,a] * e_[j,b]
)
dr.set_n_body_base(tiT, 2) #Its symmetry (not antisymmetric, but fucked)
Tidagger = Tidagger.simplify()
Tidagger.cache()
"""The T(alpha_j) operator"""
tj=IndexedBase("tj")
rhs = IndexedBase("rhs")
Tj = dr.einst( #The cluster operator
    tj[a, i] * e_[a, i] + Rational(1, 2) * tj[a, b, i, j] * e_[a, i] * e_[b, j]
)
dr.set_n_body_base(tj, 2) #Its symmetry (not antisymmetric, but fucked)
Tj = Tj.simplify()
Tj.cache()

#### Similarity transform of the Hamiltonian
curr_X1 = (dr.ham | X1).simplify()
sim_H_X1 = (dr.ham | X1).simplify()
curr_X2 = (dr.ham | X2).simplify()
sim_H_X2 = (dr.ham | X2).simplify()
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


derivative_eq_X1=dr.define(rhs[i,a],(Tidagger*sim_H_X1).eval_fermi_vev().simplify())
derivative_eq_X2=dr.define(rhs[i,j,a,b],(Tidagger*sim_H_X2).eval_fermi_vev().simplify())

with dr.report("Approach 2.html", "Approach 2") as rep:
    rep.add(content=derivative_eq_X1, description="Commutator expectation value (single excitations)")
    rep.add(content=derivative_eq_X2, description="Commutator expectation value (double excitations)")

from gristmill import *

printer = EinsumPrinter()
working_equation1=[derivative_eq_X1]
working_equation2=[derivative_eq_X2]
with open("single_expval.txt", "w") as outfile:
    outfile.write(printer.doprint(working_equation1))
with open("double_expval.txt", "w") as outfile:
    outfile.write(printer.doprint(working_equation2))

eval_seq_1 = optimize(
    working_equation1, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
eval_seq_2 = optimize(
    working_equation2, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)

with open("single_expval_optimized.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_1))
with open("double_expval_optimized.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq_2))



ti=IndexedBase("t_r") #The excitation operator (left side, technically we use ti_dagger)
ti=IndexedBase("t_l") #The left hand excitation commutator
left_right
