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

def get_combined_loss(f1,f2=None, f3=None):
    if f2 is None:
        f2 = f1
    if f3 is None:
        f3 = f2
    def f(y_true, y_pred):
        loss1 = f1(y_true, y_pred)
        loss2 = f2(y_true[:,-1], y_pred[:,-1])
        loss3 = f3(y_true.sum(axis=0), y_pred.sum(axis=0))
        return (loss1 + loss2 + loss3)/3
    
    return f

def combined_loss(y_true, y_pred):
    
    loss1 = log_mean_square_loss(y_true, y_pred)
    loss2 = log_mean_square_loss(y_true[:,-1], y_pred[:,-1])
    return (loss1 + loss2) / 2


def bitonic_loss(y_true, y_pred):
    with N.errstate(divide='ignore'):
        y_true = N.where(y_true == 0, -1, N.log(y_true))
        y_pred = N.where(y_pred == 0, -1, N.log(y_pred))
    diff = y_pred - y_true
    diff = N.where(diff >= 0, diff, -2 * diff)
    
    return NL.norm(diff.flatten(), ord=2)