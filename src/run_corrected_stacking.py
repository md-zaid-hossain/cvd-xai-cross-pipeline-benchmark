import argparse, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
parser=argparse.ArgumentParser(description='Recompute the corrected matched-configuration stacking ensemble.')
parser.add_argument('--data-file',default='data/local/Heart_disease_clean_stage1.csv')
parser.add_argument('--predictions-dir',default='artifacts/predictions')
parser.add_argument('--output-dir',default='reproduced-results/stacking')
args=parser.parse_args()
DATA_FILE=os.path.abspath(args.data_file);PRED=os.path.abspath(args.predictions_dir);OUT=os.path.abspath(args.output_dir);os.makedirs(OUT,exist_ok=True)
R=42
# canonical data
df=pd.read_csv(DATA_FILE)
SBP='ap_hi(Systolic blood pressure)'; DBP='ap_lo(Diastolic blood pressure)'; CH='cholesterol(1: normal, 2: above normal, 3: well above normal)'; GL='Glucose(1: normal, 2: above normal, 3: well above normal)'
df=df[df['BMI'].between(10,60)&((df[SBP]-df[DBP])>0)].reset_index(drop=True)
y=(df['cardiovascular disease']=='has cardiovascular disease').astype(int).values
df['PP']=df[SBP]-df[DBP]; df['MAP']=(df[SBP]+2*df[DBP])/3; df['risk_count']=(df[CH]>1).astype(int)+(df[GL]>1).astype(int)+df['smoking']+df['alcohol']+(1-df['active'])
features=['Gender (0: female, 1: male)','age(years)','BMI',SBP,DBP,CH,GL,'smoking','alcohol','active','PP','MAP','risk_count']
X=df[features].values
Xtmp,Xtest,ytmp,ytest=train_test_split(X,y,test_size=.15,random_state=R,stratify=y)
Xtrain,Xval,ytrain,yval=train_test_split(Xtmp,ytmp,test_size=.1765,random_state=R,stratify=ytmp)
assert np.array_equal(ytest,np.load(os.path.join(PRED,'y_test.npy')).astype(int))
# fixed iteration counts from primary validation early stopping: best_iteration is 0-indexed
params={
'xgb':dict(n_estimators=186,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,min_child_weight=3,gamma=.1,reg_alpha=.1,reg_lambda=1.5,eval_metric='auc',random_state=R,verbosity=0,tree_method='hist',device='cpu',n_jobs=-1),
'lgb':dict(n_estimators=103,num_leaves=63,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,min_child_samples=20,reg_alpha=.1,reg_lambda=1.0,random_state=R,verbose=-1,device='cpu',n_jobs=-1),
'cat':dict(iterations=217,depth=6,learning_rate=.05,l2_leaf_reg=3,bagging_temperature=.5,eval_metric='AUC',random_seed=R,verbose=0,task_type='CPU',allow_writing_files=False,thread_count=-1),
'rf':dict(n_estimators=300,max_depth=10,min_samples_leaf=5,max_features='sqrt',random_state=R,n_jobs=-1)
}
def make_models(seed):
    p={k:v.copy() for k,v in params.items()}
    p['xgb']['random_state']=seed;p['lgb']['random_state']=seed;p['cat']['random_seed']=seed;p['rf']['random_state']=seed
    return [xgb.XGBClassifier(**p['xgb']),lgb.LGBMClassifier(**p['lgb']),CatBoostClassifier(**p['cat']),RandomForestClassifier(**p['rf'])]
N=5; skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=R); oof=np.zeros((len(Xtrain),4))
fold_results=[]
for fold,(tri,vi) in enumerate(skf.split(Xtrain,ytrain),1):
    ms=make_models(R+fold)
    t=time.time()
    for j,m in enumerate(ms):
        m.fit(Xtrain[tri],ytrain[tri])
        oof[vi,j]=m.predict_proba(Xtrain[vi])[:,1]
    fold_results.append({'fold':fold,**{n:roc_auc_score(ytrain[vi],oof[vi,j]) for j,n in enumerate(['XGBoost','LightGBM','CatBoost','Random Forest'])}})
    print('fold',fold,'sec',round(time.time()-t,1),fold_results[-1],flush=True)
meta=LogisticRegression(C=1,max_iter=2000,random_state=R).fit(oof,ytrain)
# final base models exact same configs as OOF, fitted on same primary training set
final=make_models(R)
ptest=np.zeros((len(Xtest),4))
for j,m in enumerate(final):
    m.fit(Xtrain,ytrain)
    ptest[:,j]=m.predict_proba(Xtest)[:,1]
p=meta.predict_proba(ptest)[:,1]; pred=(p>=.5).astype(int)
res={'Model':'Corrected Stacking Ensemble','AUC':roc_auc_score(ytest,p),'Accuracy':accuracy_score(ytest,pred),'F1':f1_score(ytest,pred),'Precision':precision_score(ytest,pred),'Recall':recall_score(ytest,pred),'AP':average_precision_score(ytest,p),'Meta intercept':meta.intercept_[0]}
for n,c in zip(['XGB weight','LGB weight','Cat weight','RF weight'],meta.coef_[0]):res[n]=c
print(res)
pd.DataFrame(fold_results).to_csv(os.path.join(OUT,'corrected_stacking_oof_folds.csv'),index=False)
pd.DataFrame([res]).to_csv(os.path.join(OUT,'corrected_stacking_metrics.csv'),index=False)
np.save(os.path.join(OUT,'proba_stack_corrected.npy'),p);np.save(os.path.join(OUT,'pred_stack_corrected.npy'),pred)
