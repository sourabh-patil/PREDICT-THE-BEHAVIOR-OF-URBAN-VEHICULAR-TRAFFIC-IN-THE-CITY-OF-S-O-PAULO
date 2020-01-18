############## program for closed form method or normal equation method #############
#importing libraries
import csv
import numpy as np
import matplotlib.pyplot as plt

#for mean square error
def mse(t, y):
    n = t - y

    sq = [x * x for x in n]

    avg = sum(sq) / int(len(sq))
    rms = np.sqrt(avg)
    return rms

#reading csv file as list
with open ('/Users/Sourabh/Desktop/ML/trafficdata.csv','r') as csvFile:
    trfc_data = list(csv.reader(csvFile,delimiter=','))

mat_X = np.ones((135,18))
mat_Y = np.ones(135)

#loading data in matrix X
for i in range(len(trfc_data)-1):
    for j in range(len(trfc_data[0])-1):
        mat_X[i][j+1] = float(trfc_data[i+1][j])

#loading output in matrix
for i in range(len(trfc_data)-1):
    mat_Y[i]= float(trfc_data[i+1][len(trfc_data[0])-1])

#getting lenghts of 80 and 20% data
tdl = int(((len(trfc_data)-1)*80)/100)
vdl = int(((len(trfc_data)-1)*20)/100)

mat_X_train = mat_X[0:tdl:1]
mat_X_validate = mat_X[tdl:tdl+vdl:1]

mat_Y_train = mat_Y[0:tdl:1]
mat_Y_validate = mat_Y[tdl:tdl+vdl:1]

hypo = np.ones(18)
idmat = np.identity(18)
idmat[0] = 0
lamb = 0.01

M_1 = np.linalg.inv((np.matmul((np.transpose(mat_X_train)),mat_X_train)) + (lamb*idmat))
M_2 = np.matmul(np.transpose(mat_X_train),mat_Y_train)

hypo = np.matmul(M_1,M_2)

validate = np.matmul(hypo,np.transpose(mat_X_validate))

comp_matrix = [mat_Y_validate,validate]

with open ('/Users/Sourabh/Desktop/ML/comp_matrix.csv','w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(comp_matrix)
writeFile.close()
csvFile.close()

x = np.arange(0,vdl,1)

y1 = mat_Y_validate
plt.plot(x,y1,label='real data')

y2 = validate
plt.plot(x,y2,label='validation data')

plt.title('Closed form method')
plt.xlabel('training example')
plt.ylabel('output value')
plt.legend()
plt.show()

o = mse(y1, y2)

print('The mean square error is '+ str(o))






