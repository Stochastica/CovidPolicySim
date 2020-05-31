import numpy as N
import numpy.linalg as NL

def _safe_log(y):
    with N.errstate(divide='ignore'):
        return N.where(y == 0, -1, N.log(y))

def log_mean_square_loss(y_true, y_pred):
    with N.errstate(divide='ignore'):
        y_true = N.where(y_true == 0, -1, N.log(y_true))
        y_pred = N.where(y_pred == 0, -1, N.log(y_pred))
    v = (y_true - y_pred).flatten()
    return N.mean(v * v)

def log_max_loss(y_true, y_pred):
    with N.errstate(divide='ignore'):
        y_true = N.where(y_true == 0, -1, N.log(y_true))
        y_pred = N.where(y_pred == 0, -1, N.log(y_pred))
    return N.max(N.abs(y_true - y_pred))
    
def relative_loss(y_true, y_pred):
    return N.mean(N.abs((y_pred - y_true) / N.maximum(y_true, 10)))

def get_log_minkowski_loss(p):
    
    def f(y_true, y_pred):
        with N.errstate(divide='ignore'):
            y_true = N.where(y_true == 0, -1, N.log(y_true))
            y_pred = N.where(y_pred == 0, -1, N.log(y_pred))
        v = (y_true - y_pred).flatten()
        return NL.norm(v, ord=p)
    
    return f

def get_minkowski_loss(p):
    
    def f(y_true, y_pred):
        return NL.norm((y_true-y_pred).flatten(), ord=p)
    
    return f

def get_combined_loss(loss_func):
    
    def f(y_true, y_pred):
        loss1 = loss_func(y_true, y_pred)
        loss2 = loss_func(y_true[:,-1], y_pred[:,-1])
        loss3 = loss_func(y_true.sum(axis=0), y_pred.sum(axis=0))
        return (loss1 + loss2 + loss3) / 3
    
    return f

def combined_loss(y_true, y_pred):
    
    loss1 = log_mean_square_loss(y_true, y_pred)
    loss2 = log_mean_square_loss(y_true[:,-1], y_pred[:,-1])
    return (loss1 + loss2) / 2