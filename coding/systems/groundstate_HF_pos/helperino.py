infile=open("infiletest.txt").readlines()
vals=[]
for k,line in enumerate(infile):
    if k%2==1:
        continue
    else:
        vals.append(float(line.split()[-1]))
print(vals)
