# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:17:33 2018

@author: Zhipeng
"""

# collect results
import numpy as np

true_npy=r'F:\Google Drive\Alcohol and ML\ML analysis code\TF code and extraction\feature extraction\sub60_test\repeat_cv_results.npy'
null_npy=r'F:\Google Drive\Alcohol and ML\ML analysis code\TF code and extraction\feature extraction\sub60_test\permutation_results.npy'
csv_data=r'F:\Google Drive\Alcohol and ML\ML analysis code\TF code and extraction\feature extraction\sub60_test\feedback_only_sub60_FCP2_3ROI_raw2.csv'

def collect_results(true_npy, null_npy):
    true_result = np.load(true_npy).item()
    null_result = np.load(null_npy).item()
    return true_result, null_result

def select_sig_betas(true_result, null_result)    
    true_betas=true_result['all_coef'].mean(-1).mean(-1)
    null_betas=null_result['all_coef'].reshape(null_result['all_coef'].shape[0],-1) 
    sig_betas=np.zeros(np.size(true_betas))
    for idx, val in enumerate (true_betas):
        if val>=np.percentile(null_betas[idx,:],97.5) or val<=np.percentile(null_betas[idx,:],2.5):
            sig_betas[idx]=val
    return sig_betas 

def interpret_beta(csv_file,sig_betas,regions,conditions,mid_stuff):
    ''' 
    csv_file: read csv_file and get feature names;
    sig_betas: beta values that are out of 95% CI of null betas;
    regions: Regions to be seperated from feature names;
             e.g. regions=['F','C','P']
            
    conditions: conditions marks to seperated from feature names;
                e.g. conditions=['13','23','16','26']
  
    mid_stuff: to be concanated with region and condition as some pattern
                (e.g. '_event_dsample5_' for C_event_dsample5_101)
    '''
    all_data=pd.read_csv(csv_data)
    feature_names=list(all_data.drop(columns=['y']))
    beta_interpreter={}
    for region_i in regions:
        for cond_i in conditions:
            test_patern= region_i+mid_stuff+cond_i
            patern_idx=[i for i, s in enumerate(feature_names) if s.startswith(test_patern)]
            patern_beta=sig_betas[patern_idx]
            beta_interpreter[region_i+'_'+cond_i]=patern_beta
            beta_interpreter[region_i+'_'+cond_i+'_idx2check']=patern_idx # check if they are consecutive
    data2plot={}
    for cond_i in conditions:
        cond_by_regions=[]
        for region_i in regions:
            print(region_i)
            cond_by_regions.append(beta_interpreter[region_i+'_'+cond_i])
            data2plot['con_'+cond_i]=np.vstack(cond_by_regions).T
    return beta_interpreter,data2plot


def plot_interpreter(data2plot,fname2save,dpi2save,
                     regions,conditions,condition_names):
    '''
    condition_names: subplot title, need to be consistent with condition marks.
                     e.g.['Large Reward Hit','Large Reward Miss','Small Reward Hit','Small Reward Miss']
                     for conditions=['13','23','16','26']
    regions: Regions to be seperated from feature names;
             e.g. regions=['F','C','P']       
    conditions: conditions marks to seperated from feature names;
                e.g. conditions=['13','23','16','26']
    '''
    fig,axn=plt.subplots(len(conditions),1,figsize=(27,7),sharex=True, sharey=True)
    cbar_ax=fig.add_axes([.9, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        sns.heatmap(data2plot['con_'+conditions[i]].T,linewidths=0.2,square=True, 
                    vmax=0.1,vmin=-0.1,cmap='RdBu_r',
                    yticklabels=regions,ax=ax,
                    cbar=i==0,
                    cbar_ax=None if i else cbar_ax)
        ax.set_title(condition_names[i])
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(fname2save,dpi=dpi2save)
    
    

true_result, null_result, beta_results=collect_results(true_npy, null_npy)
true_npy=r'Y:\zhipeng EEG preprocessing\ML_python\new_py_code\repeat_cv_results.npy'
test=np.load(true_npy).item()
np.corrcoef(test['mean_y_pred'],test['true_y'])
mean_coef=test['all_coef'].mean(-1).mean(-1)