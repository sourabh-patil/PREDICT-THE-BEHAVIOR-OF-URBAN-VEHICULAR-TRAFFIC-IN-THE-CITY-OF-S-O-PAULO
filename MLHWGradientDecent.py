############### program for gradient decent using vectorization #############

#import required libraries
import csv
import numpy as np
import matplotlib.pyplot as plt

#read data from csv file as list
with open('/Users/Sourabh/Desktop/ML/trafficdata.csv', 'r') as csvFile:
    trfc_data = list(csv.reader(csvFile, delimiter=','))

#getting length for 80 and 20% of the data for further use
tdl = int(((len(trfc_data) - 1) * 80) / 100)
vdl = int(((len(trfc_data) - 1) * 20) / 100)

#defining matrices X and Y for further use
mat_X = np.ones((135, 18))
mat_Y = np.zeros(135)

#loading data from list to matrices X and Y
for i in range(len(trfc_data) - 1):
    for j in range(len(trfc_data[0]) - 1):
        mat_X[i][j + 1] = float(trfc_data[i + 1][j])

for i in range(len(trfc_data) - 1):
    mat_Y[i] = float(trfc_data[i + 1][len(trfc_data[0]) - 1])

#segregating 80 and 20% data in training and validating matrices
mat_X_train = mat_X[0:tdl:1]
mat_X_validate = mat_X[tdl:tdl + vdl:1]

mat_Y_train = mat_Y[0:tdl:1]
mat_Y_validate = mat_Y[tdl:tdl + vdl:1]

#defining matrix for hypothesis initially
beta_hypo = np.ones(18)

#defining diffent variables
lambda_val = 0.01
learn_rate = 0.005
m = len(mat_X_train)


#for returning the differential of cost function with regularization term
def delta_cost_fun(Y_mat, X_mat, beta_hypo,j_val):

    m = len(X_mat)

    diff_cost = 0

    for i in range(m):
        diff_cost = diff_cost + (((hypo(beta_hypo, X_mat[i])) - Y_mat[i]) * X_mat[i][j_val])

    return diff_cost

#for matrix multiplication
def hypo(beta_hypo_2, X_val):
    return np.matmul(beta_hypo_2, X_val)

#for mean square error
def mse(t, y):
    n = t - y

    sq = [x * x for x in n]

    avg = sum(sq) / int(len(sq))
    rms = np.sqrt(avg)
    return rms

#for loop to execute number of iteration
for i in range(1500):
    temp1 = (np.matmul(beta_hypo, np.transpose(mat_X_train)) - mat_Y_train)
    temp2 = (np.matmul(np.transpose(mat_X_train), temp1))
    beta_hypo = (1 - (learn_rate * lambda_val / m)) * beta_hypo - (learn_rate / m) * temp2

#for plotting perpose
x = ([range(len(mat_Y_validate))])
Index = np.reshape(x, (len(mat_Y_validate),))

mat_Y_predicted = (np.matmul(beta_hypo, np.transpose(mat_X_validate)))
y = np.reshape(mat_Y_predicted, (len(mat_Y_predicted),))

#plotting the values
plt.plot(Index, y, label='Predicted')
plt.plot(Index, mat_Y_validate, label='Actual')
plt.xlabel('training example')
plt.ylabel('Output Value')
plt.title('Gradient Decent Method')
plt.legend()
plt.show()
print('The predicted values according to our hypo are===='+ str(beta_hypo))

o = mse(y, mat_Y_validate)

print('The mean square error is '+ str(o))