import os

filename1 = "Autopara1-7-Normalization.dot"
filename2 = "Autopara1-7-Normalization2.dot"
auto_File = open(filename1)
data_File = open(filename2)
line_auto = auto_File.readline()
line_data = data_File.readline()
count = 0
values=[]
result=[]

while line_data and count < 1600:
    line_auto = auto_File.readline()
    line_data = data_File.readline()
    a = line_auto.split(' ')
    b = line_data.split(' ')
    for i in range(min(len(a),len(b))):
        if a[i] in {'[S]0', '[S]1', '[S]2', '[B]','[P]'}:
            if a[i] != b[i]:
                values.append(count)
                result.append([a,b])
                break
    count += 1

for i in range(len(values)):
    print(values[i])
    print('old: ', result[i][0])
    print('new: ', result[i][1])
