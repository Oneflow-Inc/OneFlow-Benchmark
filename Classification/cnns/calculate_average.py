import os

filename = "AutoPara1-7-Normalization2.txt"
f = open(filename)
line = f.readline()
count = 0
sum = 0
values=[]
while line:
    a = line.split(' ')
    for i in range(len(a)):
        if a[i]=='samples/s:':
            x = float(a[i+1][:-1])
            values.append(x)
            print(x)
            count += 1
            sum += x
    line = f.readline()
avg = sum/count

print("count: ", count)
print("sum: ", sum)
print("avg", avg)

vvalues = []
vsum=0
vcount=0
for j in values:
    if 0.7*avg <= j <= 1.3*avg:
        vvalues.append(j)
        vcount += 1
        vsum += j
vavg = vsum / vcount

print("vcount: ", vcount)
print("vsum: ", vsum)
print("vavg: ", vavg)
