import os, math, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from statsmodels.stats.multitest import multipletests

BASE='/mnt/data/cvd_work/CVD Research'
OUT='/mnt/data/cvd_work/revised_outputs'
os.makedirs(OUT,exist_ok=True)
models={
'Decision Tree':'dt','Naive Bayes':'nb','k-NN':'knn','Logistic Regression':'lr','Random Forest':'rf','SVM (RBF)':'svm','MLP':'mlp','XGBoost':'xgb','LightGBM':'lgbm','CatBoost':'cat','Stacking Ensemble':'stack','TabNet':'tabnet','FT-Transformer':'ft','VIME-inspired SSL':'vime','SAINT-inspired':'saint'}
y=np.load(os.path.join(BASE,'y_test.npy')).astype(int)
probas={name:np.load(os.path.join(BASE,f'proba_{s}.npy')).astype(float).reshape(-1) for name,s in models.items()}
preds={name:np.load(os.path.join(BASE,f'pred_{s}.npy')).astype(int).reshape(-1) for name,s in models.items()}
assert all(len(v)==len(y) for v in probas.values())

# Validated fast DeLong implementation with midranks and 0.5 handling for ties.
def compute_midrank(x):
    J=np.argsort(x)
    Z=x[J]
    N=len(x)
    T=np.zeros(N,float)
    i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]:
            j+=1
        T[i:j]=0.5*(i+j-1)
        i=j
    T2=np.empty(N,float)
    T2[J]=T+1.0
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m=label_1_count
    n=predictions_sorted_transposed.shape[1]-m
    k=predictions_sorted_transposed.shape[0]
    positive=predictions_sorted_transposed[:,:m]
    negative=predictions_sorted_transposed[:,m:]
    tx=np.empty((k,m)); ty=np.empty((k,n)); tz=np.empty((k,m+n))
    for r in range(k):
        tx[r]=compute_midrank(positive[r])
        ty[r]=compute_midrank(negative[r])
        tz[r]=compute_midrank(predictions_sorted_transposed[r])
    aucs=tz[:,:m].sum(axis=1)/(m*n) - (m+1.0)/(2.0*n)
    v01=(tz[:,:m]-tx)/n
    v10=1.0-(tz[:,m:]-ty)/m
    sx=np.atleast_2d(np.cov(v01, bias=False))
    sy=np.atleast_2d(np.cov(v10, bias=False))
    cov=sx/m + sy/n
    return aucs,cov

def delong_test(y_true,p1,p2):
    order=np.argsort(-y_true)
    m=int(y_true.sum())
    preds=np.vstack([p1,p2])[:,order]
    aucs,cov=fast_delong(preds,m)
    l=np.array([[1.,-1.]])
    var=float(l@cov@l.T)
    diff=float(aucs[0]-aucs[1])
    if var <= 0 or not np.isfinite(var):
        # No fallback that discards covariance. Exact equality yields p=1; otherwise mark invalid.
        return aucs[0],aucs[1],np.nan,(1.0 if abs(diff)<1e-15 else np.nan),var
    z=diff/math.sqrt(var)
    p=2*stats.norm.sf(abs(z))
    return aucs[0],aucs[1],z,p,var

# Test AUCs and metrics
metric_rows=[]
for name in models:
    p=probas[name]; pred=preds[name]
    metric_rows.append({'Model':name,'AUC':roc_auc_score(y,p),'Accuracy':accuracy_score(y,pred),'F1':f1_score(y,pred),'Precision':precision_score(y,pred),'Recall':recall_score(y,pred),'Brier':brier_score_loss(y,p)})
pd.DataFrame(metric_rows).sort_values('AUC',ascending=False).to_csv(os.path.join(OUT,'validated_primary_metrics.csv'),index=False)

# all pairwise
rows=[]
for a,b in itertools.combinations(models.keys(),2):
    aa,bb,z,p,var=delong_test(y,probas[a],probas[b])
    rows.append({'Model A':a,'Model B':b,'AUC A':aa,'AUC B':bb,'Delta AUC':aa-bb,'z':z,'p':p,'variance_difference':var})
df=pd.DataFrame(rows)
valid=df['p'].notna()
df.loc[valid,'BH_q_all105']=multipletests(df.loc[valid,'p'],method='fdr_bh')[1]
df.loc[valid,'Holm_p_all105']=multipletests(df.loc[valid,'p'],method='holm')[1]
df.to_csv(os.path.join(OUT,'validated_delong_all_pairs.csv'),index=False)

# MLP advanced family (9 models as manuscript scope)
advanced=['XGBoost','LightGBM','CatBoost','Stacking Ensemble','TabNet','FT-Transformer','VIME-inspired SSL','SAINT-inspired','Random Forest']
mlp_rows=[]
for a in advanced:
    aa,bb,z,p,var=delong_test(y,probas[a],probas['MLP'])
    mlp_rows.append({'Model vs MLP':a,'AUC model':aa,'AUC MLP':bb,'Delta AUC':aa-bb,'z':z,'p':p})
mdf=pd.DataFrame(mlp_rows)
mdf['BH_q']=multipletests(mdf['p'],method='fdr_bh')[1]
mdf['BH_significant_0.05']=mdf['BH_q']<0.05
mdf.to_csv(os.path.join(OUT,'validated_delong_vs_mlp_bh.csv'),index=False)

# top six primary
ranked=sorted(models.keys(), key=lambda n: roc_auc_score(y,probas[n]), reverse=True)
top6=ranked[:6]
toprows=[]
for a,b in itertools.combinations(top6,2):
    aa,bb,z,p,var=delong_test(y,probas[a],probas[b])
    toprows.append({'Model A':a,'Model B':b,'Delta AUC':aa-bb,'z':z,'p':p})
tdf=pd.DataFrame(toprows)
tdf['BH_q']=multipletests(tdf['p'],method='fdr_bh')[1]
tdf['BH_significant_0.05']=tdf['BH_q']<0.05
tdf.to_csv(os.path.join(OUT,'validated_delong_top6_bh.csv'),index=False)

# calibration from exact primary probabilities for all 15
cal_rows=[]
curves={}
for name,p in probas.items():
    eps=1e-6
    pp=np.clip(p,eps,1-eps)
    logit=np.log(pp/(1-pp)).reshape(-1,1)
    # unpenalized logistic calibration y ~ intercept + slope*logit(p)
    try:
        mod=LogisticRegression(penalty=None,solver='lbfgs',max_iter=5000).fit(logit,y)
    except Exception:
        mod=LogisticRegression(C=1e12,solver='lbfgs',max_iter=5000).fit(logit,y)
    frac,mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    curves[name]=(mean,frac)
    cal_rows.append({'Model':name,'AUC':roc_auc_score(y,p),'Brier':brier_score_loss(y,p),'Calibration intercept':float(mod.intercept_[0]),'Calibration slope':float(mod.coef_[0,0])})
caldf=pd.DataFrame(cal_rows).sort_values('AUC',ascending=False)
caldf.to_csv(os.path.join(OUT,'exact_primary_calibration.csv'),index=False)

# figure top six calibration
import matplotlib.pyplot as plt
plt.figure(figsize=(8,7))
for name in top6:
    mean,frac=curves[name]
    plt.plot(mean,frac,marker='o',label=name)
plt.plot([0,1],[0,1],'--',linewidth=1,label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Observed proportion')
plt.title('Calibration curves for the six highest-AUC primary models')
plt.legend(fontsize=8)
plt.grid(alpha=.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'exact_primary_calibration_top6.png'),dpi=300)
plt.close()

# pairwise heatmap validated
names=ranked
mat=np.ones((len(names),len(names)))
for _,r in df.iterrows():
    i=names.index(r['Model A']); j=names.index(r['Model B'])
    mat[i,j]=mat[j,i]=r['p']
plt.figure(figsize=(12,10))
import matplotlib.colors as colors
im=plt.imshow(mat,cmap='viridis_r',norm=colors.LogNorm(vmin=max(np.nanmin(mat[mat>0]),1e-6),vmax=1))
plt.colorbar(im,label='DeLong p-value (log scale)')
plt.xticks(range(len(names)),names,rotation=90,fontsize=8)
plt.yticks(range(len(names)),names,fontsize=8)
plt.title('Validated pairwise DeLong tests (midrank implementation)')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'validated_delong_heatmap.png'),dpi=300)
plt.close()

print('top6',top6)
print('\nMLP comparisons')
print(mdf.sort_values('p').to_string(index=False))
print('\nTop6 comparisons')
print(tdf.sort_values('p').to_string(index=False))
print('\nSignificant all unadjusted',int((df.p<.05).sum()),'of',len(df))
print('Significant all BH105',int((df.BH_q_all105<.05).sum()))
print('\nCalibration top6')
print(caldf.head(6).to_string(index=False))
