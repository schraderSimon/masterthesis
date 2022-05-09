import numpy as np
import matplotlib.pyplot as plt
dim=3
X=np.random.rand(5,5)
X=(X+X.T)/2
num=10

num_simul=100
means=np.zeros(num)
stds=np.zeros(num)
x=np.linspace(1,num,num)
for noise_damping in range(1,num+1):
    eigenvalues=np.zeros(num_simul)
    for i in range(0,num_simul):
        noise=np.random.uniform(-0.5/noise_damping, 0.5/noise_damping, size=X.shape)
        noise=(noise+noise.T)/noise_damping
        mat=X+noise
        val,vec=np.linalg.eigh(mat)
        eigenvalues[i]=val[0]
    means[noise_damping-1]=np.mean(eigenvalues)
    stds[noise_damping-1]=np.var(eigenvalues)
plt.plot(x,means)
plt.fill_between(x, means - stds, means + stds,
                 color='gray', alpha=0.2)
print(stds)
plt.show()
