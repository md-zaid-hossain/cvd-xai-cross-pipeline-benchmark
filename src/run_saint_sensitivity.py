import argparse, os, math
import numpy as np,pandas as pd,torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
parser=argparse.ArgumentParser(description='Recompute SAINT-inspired batch-size and row-order sensitivity.')
parser.add_argument('--data-file',default='data/local/Heart_disease_clean_stage1.csv')
parser.add_argument('--predictions-dir',default='artifacts/predictions')
parser.add_argument('--model-file',default='artifacts/models/saint_cvd_model.pt')
parser.add_argument('--output-dir',default='reproduced-results/saint-sensitivity')
args=parser.parse_args();PRED=os.path.abspath(args.predictions_dir);OUT=os.path.abspath(args.output_dir);os.makedirs(OUT,exist_ok=True)
class FeatureEmbedding(nn.Module):
 def __init__(self,n_features,d_model):
  super().__init__();self.n_features=n_features;self.feature_embeddings=nn.ModuleList([nn.Linear(1,d_model) for _ in range(n_features)]);self.feature_positions=nn.Parameter(torch.randn(1,n_features,d_model))
 def forward(self,x): return torch.stack([self.feature_embeddings[i](x[:,i:i+1]) for i in range(self.n_features)],dim=1)+self.feature_positions
class SelfAttentionBlock(nn.Module):
 def __init__(self,d_model,n_heads,dropout=.1):
  super().__init__();self.attention=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True);self.norm1=nn.LayerNorm(d_model);self.norm2=nn.LayerNorm(d_model);self.ffn=nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d_model,d_model),nn.Dropout(dropout))
 def forward(self,x):
  a,_=self.attention(x,x,x);x=self.norm1(x+a);f=self.ffn(x);return self.norm2(x+f)
class IntersampleAttentionBlock(nn.Module):
 def __init__(self,d_model,n_heads,dropout=.1):
  super().__init__();self.attention=nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True);self.norm1=nn.LayerNorm(d_model);self.norm2=nn.LayerNorm(d_model);self.ffn=nn.Sequential(nn.Linear(d_model,4*d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d_model,d_model),nn.Dropout(dropout))
 def forward(self,x):
  xt=x.transpose(0,1);a,_=self.attention(xt,xt,xt);xt=self.norm1(xt+a);f=self.ffn(xt);xt=self.norm2(xt+f);return xt.transpose(0,1)
class SAINT(nn.Module):
 def __init__(self,n_features,n_classes=2,d_model=64,n_heads=8,n_layers=4,dropout=.15):
  super().__init__();self.feature_embedding=FeatureEmbedding(n_features,d_model);self.blocks=nn.ModuleList();
  for _ in range(n_layers): self.blocks.extend([SelfAttentionBlock(d_model,n_heads,dropout),IntersampleAttentionBlock(d_model,n_heads,dropout)])
  self.norm=nn.LayerNorm(d_model);self.classifier=nn.Sequential(nn.Linear(d_model*n_features,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,n_classes))
 def forward(self,x):
  x=self.feature_embedding(x)
  for b in self.blocks:x=b(x)
  return self.classifier(self.norm(x).reshape(x.size(0),-1))
ck=torch.load(os.path.abspath(args.model_file),map_location='cpu',weights_only=False)
model=SAINT(ck['n_features'],**ck['hyperparameters']);model.load_state_dict(ck['model_state_dict']);model.eval()
# canonical data
D=pd.read_csv(os.path.abspath(args.data_file));SBP='ap_hi(Systolic blood pressure)';DBP='ap_lo(Diastolic blood pressure)';CH='cholesterol(1: normal, 2: above normal, 3: well above normal)';GL='Glucose(1: normal, 2: above normal, 3: well above normal)'
D=D[D['BMI'].between(10,60)&((D[SBP]-D[DBP])>0)].reset_index(drop=True);D['SBP']=D[SBP];D['DBP']=D[DBP];D['PP']=D[SBP]-D[DBP];D['MAP']=(D[SBP]+2*D[DBP])/3;D['RiskCount']=(D[CH]>1).astype(int)+(D[GL]>1).astype(int)+D.smoking+D.alcohol+(1-D.active);y=(D['cardiovascular disease']=='has cardiovascular disease').astype(int).values
canonical=['Gender (0: female, 1: male)','age(years)','BMI','SBP','DBP',CH,GL,'smoking','alcohol','active','PP','MAP','RiskCount']
X=D[canonical].values.astype(np.float32);_,Xt,_,yt=train_test_split(X,y,test_size=.15,random_state=42,stratify=y);Xs=ck['scaler'].transform(Xt).astype(np.float32);assert np.array_equal(yt,np.load(os.path.join(PRED,'y_test.npy')).astype(int))
@torch.no_grad()
def predict_order(X,batch,order=None):
 n=len(X);order=np.arange(n) if order is None else np.asarray(order);out=np.empty(n,float)
 for start in range(0,n,batch):
  idx=order[start:start+batch];pr=torch.softmax(model(torch.tensor(X[idx])),dim=1)[:,1].numpy();out[idx]=pr
 return out
rows=[];ps={}
# fixed stratified-like random diagnostic subset to keep CPU runtime bounded
rng_sub=np.random.default_rng(2026)
sub=np.sort(rng_sub.choice(len(Xs),size=2048,replace=False))
Xs=Xs[sub]; yt=yt[sub]
for b in [16,64,256,1024]:
 p=predict_order(Xs,b);ps[f'batch_{b}']=p;rows.append({'condition':f'batch_size_{b}','AUC':roc_auc_score(yt,p)})
ref=ps['batch_256']
for r in rows:
 p=ps[r['condition'].replace('batch_size_','batch_')];r['mean_abs_diff_vs_batch256']=np.abs(p-ref).mean();r['max_abs_diff_vs_batch256']=np.abs(p-ref).max();r['prediction_corr_vs_batch256']=np.corrcoef(p,ref)[0,1]
rng=np.random.default_rng(123)
for rep in range(3):
 order=rng.permutation(len(Xs));p=predict_order(Xs,256,order);rows.append({'condition':f'shuffled_order_{rep+1}_batch256','AUC':roc_auc_score(yt,p),'mean_abs_diff_vs_batch256':np.abs(p-ref).mean(),'max_abs_diff_vs_batch256':np.abs(p-ref).max(),'prediction_corr_vs_batch256':np.corrcoef(p,ref)[0,1]})
out=pd.DataFrame(rows);out.to_csv(os.path.join(OUT,'saint_batch_order_sensitivity.csv'),index=False);print(out.to_string(index=False))
