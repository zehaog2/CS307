import numpy as np
import math
def rmse(y_true,y_pred):
    return math.sqrt(sum((y_true[i]-y_pred[i])**2 for i in range(len(y_true)))/len(y_true))
def mae(y_true,y_pred):
    return sum(abs(y_true[i]-y_pred[i])for i in range(len(y_true)))/len(y_true)
def mape(y_true,y_pred):
    return sum(abs(y_true[i]-y_pred[i])/abs(y_true[i])for i in range(len(y_true)))/len(y_true)
def r2(y_true,y_pred):
    top = sum((y_true[i]-y_pred[i])**2 for i in range(len(y_true)))
    bot = sum((y_true[i]-np.mean(y_true))**2 for i in range(len(y_true)))
    return 1 - top/bot
def max_error(y_true,y_pred):
    return max(abs(y_true[i]-y_pred[i])for i in range(len(y_true)))
y_true = np.array([1, 2, 3])
y_pred = np.array([2, 3, 8])
print(rmse(y_true,y_pred))
print(mae(y_true,y_pred))
print(mape(y_true,y_pred))
print(r2(y_true,y_pred))
print(max_error(y_true,y_pred))
