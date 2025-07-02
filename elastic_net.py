# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:42:28 2018

@author: Zhipeng
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:37:23 2018

@author: Zhipeng
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:41:39 2018

@author: Zhipeng
"""

import glmnet_python
from glmnet import glmnet
from cvglmnet import cvglmnet
from cvglmnetPredict import cvglmnetPredict
from cvglmnetPlot import cvglmnetPlot
from cvglmnetCoef import cvglmnetCoef
import pandas as pd
import numpy as np
import sys,time,os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
import scipy.stats
from sklearn.base import BaseEstimator,TransformerMixin, clone
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.utils import shuffle
import scipy
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
np.seterr(divide='ignore', invalid='ignore')
class Winsorizer(BaseEstimator, TransformerMixin):
    """Transforms each feature by clipping from below at the pth quantile
        and from above by the (1-p)th quantile.
     Parameters
    ----------
    quantile : float
        The quantile to clip to.
     copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.
     Attributes
    ----------
    quantile : float
        The quantile to clip to.
     data_lb_ : pandas Series, shape (n_features,)
        Per-feature lower bound to clip to.
     data_ub_ : pandas Series, shape (n_features,)
        Per-feature upper bound to clip to.
    """
    def __init__(self, quantile=0.05, copy=True):
        self.quantile = quantile
        self.copy = copy
    def _reset(self):
        """Reset internal data-dependent state of the transformer, if
        necessary. __init__ parameters are not touched.
        """
        if hasattr(self, 'data_lb_'):
            del self.data_lb_
            del self.data_ub_
    def fit(self, X, y=None):
        """Compute the pth and (1-p)th quantiles of each feature to be used
        later for clipping.
         Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to determine clip upper and lower bounds.
         y : Ignored
        """
        X = check_array(X, copy=self.copy, warn_on_dtype=True, estimator=self,
                        dtype=FLOAT_DTYPES)
        self._reset()
        self.data_lb_ = np.percentile(X, 100 * self.quantile, axis=0)
        self.data_ub_ = np.percentile(X, 100 * (1 - self.quantile), axis=0)
        return self
    def transform(self, X):
        """Clips the feature DataFrame X.
         Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform.
        """
        check_is_fitted(self, ['data_lb_', 'data_ub_'])
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        X = np.clip(X, self.data_lb_, self.data_ub_)
        return X


class glmnet_wrapper(BaseEstimator):
    '''
    Grid-search for best alpha and lambda value for ElasticNet 
    with glmnet_python. The main fucntion used is cvglmnet. This 
    is kind of wrapper to put this function in a sklearn fashion,
    which gives more compatibility with sklearn.      
    =======================
    INPUT
    alphas: alpha list to use
    nfold: number of fold to find best lambda per alpha. c.f. cvglmnet
    quantile: persentile to clip for winsorizer (100*quantile = percentail)
    ptype: loss function to use (c.f. cvglmnet)
    not2preprocess: features that no need to be normalized and winsorized
    family: univariate or mulitvariate prediction
    
    OUTPUT ARGUMENTS:
    A dict() is returned with tunning information.
    
    '''
    def __init__(self,alphas=np.linspace(0.1,1,10),nfold=10,quantile=0.01,
                 ptype='mse',family='gaussian',not2preprocess=None):
        self.alphas=alphas
        self.nfold=nfold
        self.quantile=quantile
        self.ptype=ptype
        self.family='gaussian'
        self.not2preprocess=not2preprocess
        self.estimators=[]
        self.best_cvm=[]
        self.best_alpha=[]
        self.best_lambda=[]
        self.sd=StandardScaler()
        self.win=Winsorizer(quantile=self.quantile)   
    def fit(self, X, y):
        # only normalize and winsorize selected features
        # but fit using all features
        if self.not2preprocess is None:
            X2use=X.copy()
            X2win=self.sd.fit_transform(X2use)
            self.win.fit(X2win)
            X_win=self.win.transform(X2win)
            X_norm=X_win.copy()
        else:
            X2exclude=X[:,self.not2preprocess]
            X2use=np.delete(X,self.not2preprocess,axis=1)
            X2win=self.sd.fit_transform(X2use)
            self.win.fit(X2win)
            X_win=self.win.transform(X2win)
            X_norm=np.hstack([X_win,X2exclude])
        
        # covert cv to foldid, keep foldid consistent when comparing between alpha
        cv = KFold(n_splits=self.nfold,shuffle=True)
        foldid2use=y[:,0].copy()
        foldid=-1
        for train_index, test_index in cv.split(X):
            foldid+=1
            foldid2use[test_index]=foldid
        foldid2use=foldid2use.astype(int)
        #glmnet cvglmnet
        if y.shape[1]>1:
            self.family='mgaussian'
        
        
        alphas=self.alphas
        cvms=[]
        lambdas=[]
        for alpha2use in self.alphas:        
            clf_obj=cvglmnet(x = X_norm.copy(),y = y.copy(),
                             foldid=foldid2use,alpha=alpha2use,
                             ptype=self.ptype,
                             family=self.family)            
            self.estimators.append(clf_obj)
            cvms.append(clf_obj['cvm'].min()) 
            # clf_obj['lambda_min']==clf_obj['lambdau'][np.argmin(clf_obj['cvm'])]
            lambdas.append(clf_obj['lambda_min'])
        min_cvm_idx=cvms.index(min(cvms))
        self.best_estimator=self.estimators[min_cvm_idx]
        self.best_cvm=min(cvms)
        self.best_alpha=alphas[min_cvm_idx]
        self.best_lambda=lambdas[min_cvm_idx]
        return self         
    def predict(self, X):
        if self.not2preprocess is None:
            X2use=X.copy()
            X2win=self.sd.transform(X2use)
            X_win=self.win.transform(X2win)
            X_norm=X_win.copy()
        else:
            X2exclude=X[:,self.not2preprocess]
            X2use=np.delete(X,self.not2preprocess,axis=1)
            X2win=self.sd.transform(X2use)
            X_win=self.win.transform(X2win)
            X_norm=np.hstack([X_win,X2exclude])
        pred=cvglmnetPredict(self.best_estimator, X_norm, s='lambda_min')
        if self.family=='gaussian':
            pred2return=pred.reshape(-1)
        else:
            pred2return=np.squeeze(pred)
        return pred2return
    def get_info(self):
        info={}
        info['best_alpha']=self.best_alpha
        info['best_l1']=self.best_lambda[0]
        if self.family=='guassian':
            info['coef']=cvglmnetCoef(self.best_estimator, s='lambda_min').reshape(-1)
        else:
            info['coef']=cvglmnetCoef(self.best_estimator, s='lambda_min')
        return info
    def diagnostic_plot1(self,saveto=None):
        fig,axn=plt.subplots(len(self.alphas),1,figsize=(8,60), sharey=True)
        for i, ax in enumerate(axn.flat):
            plt.axes(ax)
            cvglmnetPlot(self.estimators[i])
            textstr='alpha='+str(np.round_(self.alphas[i],3))
            props=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                                            verticalalignment='top', bbox=props)
            plt.tight_layout()
        if saveto!=None:
            plt.savefig(saveto,dpi=300)
    def diagnostic_plot2(self,saveto=None):
        plt.hold(True)
        for idx, alpha2use in enumerate(self.alphas):
            obj2plot=self.estimators[idx]
            col2use=np.random.rand(3,)
            plt.plot(scipy.log(obj2plot['lambdau']),obj2plot['cvm'],color=col2use)
        plt.xlabel('log(lambda)')
        plt.ylabel('Mean-Squared Error')
        plt.title('alpha-lambda-mse')
        plt.legend(np.round_(self.alphas,3),loc='best')  
        if saveto!=None:
            plt.savefig(saveto,dpi=300)


def clf2parallel(clf_in,X,y,train_index,test_index):
    fold_test_idx=[]
    fold_coef=[]
    fold_alpha=[]
    fold_l1=[]
    fold_y_pred=[]
    fold_score=[]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_in.fit(X_train,y_train) # clf_in has standarized function
    y_pred = clf_in.predict(X_test)
    # put y_pred to the same order!
    fold_y_pred.append(y_pred)
#            save pred y, coef, and mse for each main fold
    mse_score=mean_squared_error(y_test, y_pred,multioutput='raw_values')
    fold_score.append(mse_score)
#           save parameter tuning results for each main fold            
    tuning_info=clf_in.get_info()
    fold_coef.append(tuning_info['coef']) # coef
    fold_alpha.append(tuning_info['best_alpha'])
    fold_l1.append(tuning_info['best_l1'])
    fold_test_idx.append(test_index)
    return fold_coef, fold_alpha, fold_l1, fold_y_pred, fold_test_idx,fold_score
    

def parallel_cv(clf_in, X, y, Nfold=10, n_jobs=-1, 
                verbose=False, pre_dispatch='2*n_jobs'):
    cv=KFold(n_splits=Nfold,shuffle=True)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results=parallel(delayed(clf2parallel)(
            clone(clf_in),X, y, train_index, test_index)
        for train_index, test_index in cv.split(X)) 
    zipped_results=list(zip(*results))
    fold_coef, fold_alpha, fold_l1, fold_y_pred, fold_test_idx,fold_score=zipped_results
    sorted_y_pred=np.empty((y.shape))*np.nan
    if y.shape[1]>1:
        for fold_i, fold_y_pred_i in zip(fold_test_idx,fold_y_pred):
            sorted_y_pred[fold_i]=np.array(fold_y_pred_i)
    else:
        for fold_i, fold_y_pred_i in zip(fold_test_idx,fold_y_pred):
            sorted_y_pred[fold_i]=np.array(fold_y_pred_i).T
    fold_coef_ = np.squeeze(np.vstack(fold_coef).T)
    fold_alpha_ = np.array(fold_alpha).reshape(-1)
    fold_l1_ = np.array(fold_l1).reshape(-1)
    fold_score_ = np.vstack(fold_score).T
    sorted_y_pred_ = sorted_y_pred
    return fold_coef_, fold_alpha_, fold_l1_,fold_score_, sorted_y_pred_


def repeated_parallel_cv(clf_in,X,y,Nfold=10,rep=50,shuffle_y=False):
    results={}
    all_y_pred=[]
    all_alpha=[]
    all_l1=[]
    all_coef=[]
    all_score=[]
    seed0=np.random.randint(1000,size=1)
    for rep_i in range(rep):
        print('running~~~~~~~~~~'+str(rep_i+1)+' rep')
        if shuffle_y:
            seed1=seed0+np.random.randint(100,size=1)
            y2use=shuffle(y.copy(),random_state=int(seed1))
        else:
            y2use=y.copy()   
        
        fold_coef, fold_alpha, fold_l1,fold_score,sorted_y_pred=parallel_cv(
                clf_in, X, y2use, Nfold=Nfold, n_jobs=-1, verbose=False, pre_dispatch='2*n_jobs')
        all_y_pred.append(np.array(sorted_y_pred))
        all_alpha.append(np.array(fold_alpha))
        all_l1.append(np.array(fold_l1))
        all_coef.append(np.array(fold_coef))
        all_score.append(np.array(fold_score))
    results['all_y_pred']=np.stack(all_y_pred,axis=-1)
    results['mean_y_pred']=np.mean(np.array(all_y_pred),axis=0)
    results['all_alpha']=np.array(all_alpha).T
    results['all_l1']=np.array(all_l1).T
    results['all_coef']=np.stack(all_coef,axis=-1)
    results['all_score']=np.array(all_score).T
    results['true_y']=y.copy()
    return results

def corr_y(y_pred,y):
    if np.ndim(y_pred)==1:
        y_pred=y_pred[:,None]
    y_dim=y.shape[-1]
    r_value=np.empty((y.shape[-1]))*np.nan
    p_value=np.empty((y.shape[-1]))*np.nan
    for y_i in np.arange(y_dim):
        r_value[y_i], p_value[y_i]=scipy.stats.pearsonr(y_pred[:,y_i],y[:,y_i])
    return r_value, p_value
        
def once_csv_pred_test(data_csv):
    csv_name, ext = os.path.splitext(data_csv)
    start_time=time.time()
    all_df=pd.read_csv(data_csv)
    flag_y=[col for col in all_df.columns if 'y_' in col]
    y=np.array(all_df[flag_y])
    fs_df=all_df.drop(columns=flag_y)
    
    # find col idx that with 'flag' indicating no need to normalize the col
    flag_col=np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs=np.array(fs_df)
    clf=glmnet_wrapper(not2preprocess=flag_col)
    y_pred = cross_val_predict(clf, scipy.float64(fs), scipy.float64(y),cv=10,n_jobs=-1)
    r_value, p_value=corr_y(y_pred,y)
    e = int(time.time() - start_time)
    e_time='{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
    with open('appending_results.txt','a+') as fo:
        fo.write('\r\n filename---{} r---: {} p---: {} time--: {} \r\n'.format(csv_name, r_value, p_value, e_time))
    return r_value, p_value, e_time

def only_make_plot(data_csv):
    csv_name, ext = os.path.splitext(data_csv)
    start_time=time.time()
    all_df=pd.read_csv(data_csv)
    flag_y=[col for col in all_df.columns if 'y_' in col]
    y=np.array(all_df[flag_y])
    fs_df=all_df.drop(columns=flag_y)
    
    # find col idx that with 'flag' indicating no need to normalize the col
    flag_col=np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs=np.array(fs_df)
    clf=glmnet_wrapper(not2preprocess=flag_col)
    clf.fit(scipy.float64(fs), scipy.float64(y))
    clf.diagnostic_plot1(csv_name+'1.png')

def batch_csv_pred_test(data_path):
    os.chdir(data_path)
    filenames = os.listdir(data_path)
    csv2test=[ filename for filename in filenames if filename.endswith('.csv')]
    all_r=[]
    all_p=[]
    all_time=[]
    for csv in csv2test:
        tmp_r,tmp_p,tmp_time=once_csv_pred_test(csv)
        all_r.append(str(tmp_r))
        all_p.append(str(tmp_p))
        all_time.append(str(tmp_time))
    df=pd.DataFrame([])
    df['r value']=np.array(all_r)
    df['p value']=np.array(all_p)
    df['time']=all_time
    df.index=csv2test
    return df

def repeat_EN_csv(data_csv,reps=50,shuffle_mark=False):
#    repeated_cross_validate(clf_in,X,y,cv_fold=10,rep=50)
    csv_name, ext = os.path.splitext(data_csv)
    all_df=pd.read_csv(data_csv)
    flag_y=[col for col in all_df.columns if 'y_' in col]
    y=np.array(all_df[flag_y])
    fs_df=all_df.drop(columns=flag_y)
    # find col idx that with 'flag' indicating no need to normalize the col
    flag_col=np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs=np.array(fs_df)
    clf=glmnet_wrapper(not2preprocess=flag_col)
    results=repeated_parallel_cv(clf,scipy.float64(fs),scipy.float64(y),Nfold=10,rep=reps,shuffle_y=shuffle_mark)
    return results

if '__main__'==__name__:
    if len(sys.argv)<3:
        print('Not enough arguement.')
        sys.exit()
    elif sys.argv[1]=='-once':
        start_time=time.time()
        results=once_csv_pred_test(sys.argv[2])
        e = int(time.time() - start_time)
        print(results)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
    elif sys.argv[1]=='-batch':
        start_time=time.time()
        data_path=os.path.join(os.getcwd(),sys.argv[2])
        results=batch_csv_pred_test(data_path)
        results.to_csv('batch_test_result.csv')
        e = int(time.time() - start_time)
        print(results)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
    elif sys.argv[1]=='-repeat':
        csv_name, ext = os.path.splitext(sys.argv[2])
        start_time=time.time()
        results=repeat_EN_csv(sys.argv[2],reps=int(sys.argv[3]),shuffle_mark=False)
        e = int(time.time() - start_time)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
        np.save(csv_name+'_repeat_'+sys.argv[3]+'_cv_results.npy', results)
        r_value, p_value=corr_y(results['mean_y_pred'],results['true_y'])
        print('\rMean Predict y across {} repeats, correlation is:{},{}'.format(sys.argv[3], r_value, p_value))
    elif sys.argv[1]=='-permutation':
        csv_name, ext = os.path.splitext(sys.argv[2])
        start_time=time.time()
        results=repeat_EN_csv(sys.argv[2],reps=int(sys.argv[3]),shuffle_mark=True)
        e = int(time.time() - start_time)
        print('\rTime elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)+'\r')
        np.save(csv_name+'_permutation_'+sys.argv[3]+'_results.npy', results)
        r_value, p_value=corr_y(results['mean_y_pred'],results['true_y'])
        print('\rMean Predict y across {} repeats, correlation is:{},{}'.format(sys.argv[3], r_value, p_value))
    elif sys.argv[1]=='-plot':
        only_make_plot(sys.argv[2])






