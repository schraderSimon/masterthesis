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
ti=IndexedBase("ti")
rhs = IndexedBase("rhs")
Tidagger = dr.einst( #The cluster operator
    ti[a,i] * e_[i, a] + Rational(1, 2) * ti[a,b,i,j] * e_[i,a] * e_[j,b]
)
dr.set_n_body_base(ti, 2) #Its symmetry (not antisymmetric, but fucked)
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
curr = (dr.ham | Tj).simplify()
sim_H = (dr.ham | Tj).simplify()
for order in range(0,4):
    curr = (curr | cluster).simplify() * Rational(1, order + 1)
    stopwatch.tock("Commutator order {}".format(order + 1), curr)
    sim_H += curr
sim_H.simplify()
derivative_eq=(Tidagger*sim_H).eval_fermi_vev().simplify()
print(derivative_eq)
with dr.report("Approach 2.html", "Approach 2") as rep:
    rep.add(content=derivative_eq, description="Approach 2")
from gristmill import *

printer = EinsumPrinter()
working_equation=[dr.define(Symbol('l'), derivative_eq)]
eval_seq = optimize(
    working_equation, substs={p.nv: 5000, p.no: 1000},
    contr_strat=ContrStrat.EXHAUST
)
with open("solution.txt", "w") as outfile:
    outfile.write(printer.doprint(eval_seq))
