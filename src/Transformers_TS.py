from sklearn import base, utils
import pandas as pd
import numpy as np
import copy

from sklearn.metrics import confusion_matrix


class FourierComponents(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, periods):
        self.periods = periods
        self.cols = None
        self.day0 = None
    
    def fit(self, X, y=None):
        self.day0 = X.index[0]
        return self
    
    def transform(self, X):
        t = (X.index - self.day0).days.values
        
        self.cols = [ 'sin_1/' + str(period)[:3] + 'd' if f==np.sin else 'cos_1/' + str(period)[:3] + 'd' 
                     for period in self.periods for f in [np.sin, np.cos] ]
        
        res_arr = [ f(2.*np.pi*t/period) for period in self.periods for f in [np.sin, np.cos] ]
        
        df = pd.DataFrame( data = np.vstack(res_arr).T, columns = self.cols, index = X.index)
        
        X = pd.concat([X, df], axis=1)
        
        return X
    
    
class DayOfWeek_Trf(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        for i in range(7):
            X['Day_' + str(i)] = np.array(list(map( lambda x: 1 if x.weekday()==i else 0, X.index)))
                    
        return X


class Add_LinDrift(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        self.day0 = X.index[0]
        return self
    
    def transform(self, X):
        X['LinDrift'] = (X.index-self.day0).days.values
        return X


class AddPastValuesAsFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, n_past_days=1, cut_first=False):
        self.n_past_days = n_past_days
        self.cut_first = cut_first
        self.prev_n_counts = None
        
    def fit(self, X, y=None):
        '''
        these values will be used only in the seq_trainsform
        '''
        self.prev_n_counts = X['counts'].values[-self.n_past_days:][::-1]
        return self
    
    def transform(self, X, y=None):
        for i in range(1, self.n_past_days+1):
            X['prev_' + str(i)] = X['counts'].shift(i).fillna(method='bfill')
            
        if self.cut_first == True:
            X = X[self.n_past_days:]
        
        return X
    
    def seq_transform(self, X, y=None):
        '''
        used in seq learning
        use the values from self.prev_n_counts to generate the new features
        ''' 
        for i in range(1, self.n_past_days+1):
            X['prev_' + str(i)] = self.prev_n_counts[i-1]
        return X
   
    def seq_update(self, X, y):
        '''
        applied after calculating Y_pred
        '''
        self.prev_n_counts = np.concatenate((y, self.prev_n_counts[:-1]), axis=0)
        return self
        


class Cut_First_NRows(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, n_rows=0):
        self.n_rows = n_rows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.n_rows:]

    def seq_transform(self, X, y=None):
        '''
        used in seq learning
        in this case we do not have to cut anything
        ''' 
        return X



class GetRollingMean(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, window=3):
        self.window = window
        self.prev_n_counts = None
        
    def fit(self, X, y=None):
        '''
        these values will be used only in the seq_transform
        '''
        self.prev_n_counts = X['counts'].values[-self.window:][::-1]
        return self
    
    def transform(self, X, y=None):
        X['roll_mean'] = X['counts'].rolling(window=self.window).mean().shift(1).fillna(method='bfill')
        return X

    def seq_transform(self, X, y=None):
        '''
        used in seq learning
        in this case we do not have to cut anything
        '''
        X['roll_mean'] = np.sum(self.prev_n_counts)/float(len(self.prev_n_counts))
        return X

    def seq_update(self, X, y):
        '''
        applied after calculating Y_pred
        '''
        self.prev_n_counts = np.concatenate((y, self.prev_n_counts[:-1]), axis=0)
        return self
    

# TODO how to update prev_ewm
class GetExpWeightMean(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, alpha=3):
        self.alpha = alpha
        self.prev_ewm = None
        self.prev_counts = None
        
    def fit(self, X, y=None):
        '''
        these values will be used only in the seq_transform
        '''
        self.prev_ewm    = X['counts'].ewm(alpha=self.alpha, adjust=False).mean().shift(1).fillna(method='bfill').values[-1]
        self.prev_counts = X['counts'].values[-1]
        return self
    
    def transform(self, X, y=None):
        X['ew_mean'] = X['counts'].ewm(alpha=self.alpha, adjust=False).mean().shift(1).fillna(method='bfill')
        return X

    def seq_transform(self, X, y=None):
        X['ew_mean'] = self.alpha*self.prev_counts + (1-self.alpha)*self.prev_ewm
        return X

    def seq_update(self, X, y):
        '''
        applied after calculating Y_pred
        '''
        self.prev_ewm    = X['ew_mean'].values[-1]
        self.prev_counts = y
        return self
    
    
    
class GetFeatureNames(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self):
        self.cols = None
        
    def fit(self, X, y=None):
        self.cols = X.columns.values
        return self
        
    def transform(self, X, y=None):
        return X
    
    
class my_DropFeatures(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop(self.relevant_cols, axis=1) 
        return X
    
    def seq_transform(self, X):
        return X



class my_AppCopy(base.BaseEstimator, base.TransformerMixin):
    '''
    copy the relevant columns
    turn the columns with type=int to columns with type=float
    '''
    def __init__(self, relevant_cols):
        self.relevant_cols = relevant_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X[self.relevant_cols].copy(deep=True)            
        return X
    
    def seq_transform(self, X):
        X = X.copy(deep=True)
        return X
