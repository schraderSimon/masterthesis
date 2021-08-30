import numpy as np

class modelgenerator():
    def __init__(self,matrix_size):
        self.matrix_size=matrix_size
    def model1A(self):
        size=self.matrix_size
        M0=np.zeros((size,size))
        M1=np.zeros((size,size))
        for i in range (size):
            M0[i,i]=i+1
            M1[i,i]=i+1
        for i in range(size-2):
            M1[i,i+2]=1
            M1[i+2,i]=1
        def matrix(x):
            return M0+x*M1
        return matrix
    def model1B(self):
        size=self.matrix_size
        M0=np.zeros((size,size))
        M1=np.zeros((size,size))
        for i in range (size):
            M0[i,i]=2*(i+1)
            M1[i,i]=(i+1)
        for i in range(size-2):
            M1[i,i+2]=1
            M1[i+2,i]=1
        def matrix(x):
            return M0+x*M1
        return matrix
    def model1C(self):
        size=self.matrix_size
        M0=np.zeros((size,size))
        M1=np.zeros((size,size))
        for i in range (size):
            M0[i,i]=100*(i+1)
            M1[i,i]=-75*(i+1)
        M1[0,0]=-100
        M1[1,1]=-200
        for i in range(size-1):
            M1[i,i+1]=1
            M1[i+1,i]=1
        def matrix(x):
            return M0+x*M1
        return matrix
    def model3(self):
        size=self.matrix_size
        M0=np.zeros((size,size))
        M1=np.zeros((size,size))
        for i in range (size):
            M0[i,i]=100*(i+1)
            M1[i,i]=50*(i+1)
        M1[0,0]=40
        M1[1,1]=-80
        M1[2,2]=-180
        M1[3,3]=-260
        M1[4,4]=-320
        M1[5,5]=-335
        for i in range(size-1):
            M1[i,i+1]=2
            M1[i+1,i]=2
            if i<(size-2):
                M1[i,i+2]=5
                M1[i+2,i]=5
            if i<(size-3):
                M1[i,i+3]=5
                M1[i+3,i]=5
        def matrix(x):
            return M0+x*M1
        return matrix
