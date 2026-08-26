import argparse, os
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
parser=argparse.ArgumentParser(description='Recompute the guideline-defined Random Forest feature ablation.')
parser.add_argument('--data-file',default='data/local/Heart_disease_clean_stage1.csv')
parser.add_argument('--output-dir',default='reproduced-results/ablation')
args=parser.parse_args();OUT=os.path.abspath(args.output_dir);os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(os.path.abspath(args.data_file));SBP='ap_hi(Systolic blood pressure)';DBP='ap_lo(Diastolic blood pressure)';CH='cholesterol(1: normal, 2: above normal, 3: well above normal)';GL='Glucose(1: normal, 2: above normal, 3: well above normal)';G='Gender (0: female, 1: male)';AGE='age(years)'
df=df[df.BMI.between(10,60)&((df[SBP]-df[DBP])>0)].reset_index(drop=True);y=(df['cardiovascular disease']=='has cardiovascular disease').astype(int).values
df['PP']=df[SBP]-df[DBP];df['MAP']=(df[SBP]+2*df[DBP])/3;df['risk_count']=(df[CH]>1).astype(int)+(df[GL]>1).astype(int)+df.smoking+df.alcohol+(1-df.active)
subsets={
'Core clinical 3 (Age, SBP, Cholesterol)':[AGE,SBP,CH],
'Guideline-aligned 6 (Age, Sex, SBP, Cholesterol, Glucose, Smoking)':[AGE,G,SBP,CH,GL,'smoking'],
'Original 10 variables':[G,AGE,'BMI',SBP,DBP,CH,GL,'smoking','alcohol','active'],
'Original + PP + MAP (12 variables)':[G,AGE,'BMI',SBP,DBP,CH,GL,'smoking','alcohol','active','PP','MAP'],
'Full 13 variables':[G,AGE,'BMI',SBP,DBP,CH,GL,'smoking','alcohol','active','PP','MAP','risk_count']}
rows=[]
for name,fs in subsets.items():
 X=df[fs].values
 Xt,Xte,yt,yte=train_test_split(X,y,test_size=.15,random_state=42,stratify=y)
 Xtr,Xv,ytr,yv=train_test_split(Xt,yt,test_size=.1765,random_state=42,stratify=yt)
 m=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,max_features='sqrt',random_state=42,n_jobs=-1).fit(Xtr,ytr)
 p=m.predict_proba(Xte)[:,1]
 rows.append({'Subset':name,'n_features':len(fs),'Features':'; '.join(fs),'Test AUC':roc_auc_score(yte,p)})
out=pd.DataFrame(rows);out.to_csv(os.path.join(OUT,'guideline_defined_feature_ablation.csv'),index=False);print(out.to_string(index=False))
