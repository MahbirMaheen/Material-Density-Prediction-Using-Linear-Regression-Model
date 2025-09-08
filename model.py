import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import copy, math
from sklearn.preprocessing import StandardScaler

#Load dataset
df = pd.read_csv(r'C:/Users/Mahbir Ahmed Maheen/Desktop/Materials Project/Mechanical_Properties.csv')

#Drop rows where Ro is missing 
df = df.dropna(subset=['Ro'])

#Filling missing values with median
df['E'] = df['E'].fillna(df['E'].median())

#Convert relevant columns to numeric and coercing errors to NaN
cols_to_convert = ['E', 'Bhn', 'pH', 'HV', 'Su']
for col in cols_to_convert: 
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

#Transforming categorical feature to numeric value 
df = pd.get_dummies(df, columns = ['Material', 'Heat treatment'], drop_first = True)

#scaling numeric feature for better regression performance
scaler = StandardScaler()
features = ['E', 'Bhn', 'pH', 'HV', 'Su']
df[features] = scaler.fit_transform(df[features])

#features 
x_feature = df[features]
y_output = df['Ro']



#Definign weight parameters
b_init = 0
w_init = np.zeros(x_feature.shape[1])

#Predicting using vectorization 
def predict(x,w,b):
    p = np.dot(x,w)+b
    return p


#computing cost function 
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w)+b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return cost

#Converting Pandas data frame to numpy array
x_np = x_feature.to_numpy() if hasattr(x_feature, 't_numpy') else np.array(x_feature)
y_np = y_output.to_numpy() if hasattr(y_output, 'to_numpy') else np.array(y_output)

# replacing NaN with zeros 
x_np = np.nan_to_num(x_np)  
y_np = np.nan_to_num(y_np)



#compute gradient function 
def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0  

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db



#compute gradient descent function
def gradient_descent(x, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = w_init.copy()
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))

        if i % max(1, math.ceil(num_iters / 10)) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history

# Initialize parameters and run
initial_w = np.zeros_like(w_init)
initial_b = 0.

iterations = 4000
alpha = 1e-3  # learning rate 

w_final, b_final, J_hist = gradient_descent(x_np, y_np, initial_w, initial_b,
                                             compute_cost, compute_gradient,
                                             alpha, iterations)

#print(f"b,w found by gradient descent: {b_final:0.2f},{w_final}") # it can show the optimum value of weight parameters


#Printing each prediction along with actual value 
m, _ = x_np.shape
for i in range(m):
    print(f"prediction: {np.dot(x_np[i], w_final) + b_final:0.2f}, target value: {y_np[i]}")

"""
#This plot will show the cost as a function of learning rate - alpha, can visually observe the optimum value of alpha 
plt.plot(J_hist)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function History During Gradient Descent')
plt.grid(True)
plt.show()

"""