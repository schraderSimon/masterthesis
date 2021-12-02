def energy_equation(t1,t2,spinints_MO_fys):
    ECCSD = 0
    ECCSD=ECCSD+0.25*np.einsum("iajb,abij->",L,td,optimize=True)
    ECCSD=ECCSD+0.5*np.einsum("iajb,ai,bj->",L[:Nelec,:Nelec,Nelec:dim,Nelec:dim],ts,ts,optimize=True)
def t1_projection(t1,t2):
    pass
def t2_projection(t1,t2):
    pass
def T1_transform(t1):
    pass
