#RHF
def twobody_energy_old(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
    MO_eri=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left,HF_coefficients_right,HF_coefficients_left,HF_coefficients_right))
    energy_2e=0
    number_electronshalf=self.number_electronshalf
    large_S=np.zeros((self.number_electronshalf*2,self.number_electronshalf*2))
    large_S[:self.number_electronshalf,:self.number_electronshalf]=determinant_matrix.copy()
    large_S[self.number_electronshalf:,self.number_electronshalf:]=determinant_matrix.copy()
    for k in range(number_electronshalf*2):
        for l in range(k+1,number_electronshalf*2):
            largeS_2e=large_S.copy() #Do I really have to do this 16 times...?
            largeS_2e[:,k]=0
            largeS_2e[:,l]=0
            for a in range(number_electronshalf*2):
                for b in range(a+1,number_electronshalf*2):
                    largeS_2e[a,k]=1
                    largeS_2e[b,l]=1
                    largeS_2e[a-1,k]=0
                    largeS_2e[b-1,l]=0
                    eri_of_interestl=0
                    gright=0
                    nh=number_electronshalf
                    if(k<number_electronshalf and l<number_electronshalf and a < number_electronshalf and b< number_electronshalf):
                        eri_of_interestl=MO_eri[a,k,b,l]
                    elif(k>=number_electronshalf and l>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf):
                        eri_of_interestl=MO_eri[a-number_electronshalf,k-number_electronshalf,b-number_electronshalf,l-number_electronshalf]
                    elif(k<number_electronshalf and l>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):
                        eri_of_interestl=MO_eri[a,k,b-number_electronshalf,l-number_electronshalf]
                    elif(k>=number_electronshalf and l<number_electronshalf and a >= number_electronshalf and b< number_electronshalf):
                        eri_of_interestl=MO_eri[a-number_electronshalf,k-number_electronshalf,b,l]
                    if (a<nh and l< nh):
                        if (b<nh and k< nh):
                            gright=MO_eri[a,l,b,k]
                        elif(b>=nh and k>= nh):
                            gright=MO_eri[a,l,b-nh,k-nh]
                    elif(a>=nh and l>=nh):
                        if (b<nh and k< nh):
                            gright=MO_eri[a-nh,l-nh,b,k]
                        elif(b>=nh and k>= nh):
                            gright=MO_eri[a-nh,l-nh,b-nh,k-nh]
                    eri_of_interest=eri_of_interestl-gright
                    if(abs(eri_of_interest)>1e-10):
                        energy_2e+=np.linalg.det(largeS_2e)*eri_of_interest
    return energy_2e

#UHF
    def twobody_energy_old(self,energy_basis_2e,HF_coefficients_left,HF_coefficients_right,determinant_matrix):
        eri_MO_aabb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        eri_MO_bbaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_aaaa=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[0],HF_coefficients_right[0],HF_coefficients_left[0],HF_coefficients_right[0]),aosym="s1")
        eri_MO_bbbb=ao2mo.get_mo_eri(energy_basis_2e,(HF_coefficients_left[1],HF_coefficients_right[1],HF_coefficients_left[1],HF_coefficients_right[1]),aosym="s1")
        energy_2e=0
        determinant_matrix_alpha=determinant_matrix[0]
        determinant_matrix_beta=determinant_matrix[1]
        number_electronshalf=self.number_electronshalf
        large_S=np.zeros((number_electronshalf*2,number_electronshalf*2))
        large_S[:number_electronshalf,:number_electronshalf]=determinant_matrix_alpha.copy()
        large_S[number_electronshalf:,number_electronshalf:]=determinant_matrix_beta.copy()
        for k in range(number_electronshalf*2):
            for l in range(k+1,number_electronshalf*2):
                largeS_2e=large_S.copy()
                largeS_2e[:,k]=0
                largeS_2e[:,l]=0
                for a in range(number_electronshalf*2):
                    for b in range(number_electronshalf*2):
                        largeS_2e[a,k]=1
                        largeS_2e[b,l]=1
                        largeS_2e[a-1,k]=0
                        largeS_2e[b-1,l]=0
                        if(k<number_electronshalf and l<number_electronshalf and a < number_electronshalf and b< number_electronshalf): #alpha, alpha
                            eri_of_interest=eri_MO_aaaa[a,k,b,l]
                        elif(k>=number_electronshalf and l>=number_electronshalf and a >= number_electronshalf and b>= number_electronshalf): #beta, beta
                            eri_of_interest=eri_MO_bbbb[a-number_electronshalf,k-number_electronshalf,b-number_electronshalf,l-number_electronshalf]
                        elif(k<number_electronshalf and l>=number_electronshalf and a < number_electronshalf and b>= number_electronshalf):#alpha, beta
                            eri_of_interest=eri_MO_aabb[a,k,b-number_electronshalf,l-number_electronshalf]
                        elif(k>=number_electronshalf and l<number_electronshalf and a >= number_electronshalf and b< number_electronshalf): #beta,alpha and b>= number_electronshalf):#alpha, beta
                                eri_of_interest=eri_MO_bbaa[a-number_electronshalf,k-number_electronshalf,b,l]
                        else:
                            continue
                        if(abs(eri_of_interest)>=1e-10):
                            energy_2e+=np.linalg.det(largeS_2e)*eri_of_interest
        return energy_2e
def get_antisymm_element_old(MO_eri,n,na=None,nb=None):

    nh=int(n/2)
    if(na is None or nb is None):
        na=nh
        nb=nh
    G_mat=np.zeros((int(n*(n-1)/2),int(n*(n-1)/2)),dtype=np.float64)
    for j in range(n):
        for i in range(j):
            for l in range(n):
                for k in range(l):
                    gleft=0

                    if (i<na and k< na):
                        if (j<na and l< na):
                            gleft=MO_eri[i,k,j,l]
                        elif(j>=na and l>= na):
                            gleft=MO_eri[i,k,j-na,l-na]
                    elif(i>=na and k>= na):
                        if (j<na and l< na):
                            gleft=MO_eri[i-na,k-na,j,l]
                        elif(j>=na and l>= na):
                            gleft=MO_eri[i-na,k-na,j-na,l-na]
                    gright=0
                    if (i<na and l< na):
                        if (j<na and k< na):
                            gright=MO_eri[i,l,j,k]
                        elif(j>=na and k>=na):
                            gright=MO_eri[i,l,j-na,k-na]
                    elif(i>=na and l>=na):
                        if (j<na and k< na):
                            gright=MO_eri[i-na,l-na,j,k]
                        elif(j>=na and k>=na):
                            gright=MO_eri[i-na,l-na,j-na,k-na]
                    G_mat[int((j)*(j-1)/2+i),int((l)*(l-1)/2+k)]=gleft-gright
    return G_mat
