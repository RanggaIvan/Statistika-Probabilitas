import sys, json
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import binomtest
U='/root/.claude/uploads/47c55f18-c7ee-5521-a8c6-0772bf015a69/'
df=pd.read_csv(U+'31e06102-tokopedia_reviews_olahraga_final.csv')
raw=pd.read_csv(U+'40720732-tokopedia_product_reviews_2025.csv')
out={}
def vec(): return TfidfVectorizer(max_features=5000,min_df=2,max_df=0.9,ngram_range=(1,2))
def models(): return {'NB':MultinomialNB(),'SVM':LinearSVC(random_state=42)}
def split_eval(d,tag):
    X=d['review_text_preprocessed'].fillna(''); y=d['sentimen']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    v=vec(); A=v.fit_transform(Xtr); B=v.transform(Xte)
    res={'n':len(d),'n_train':len(Xtr),'n_test':len(Xte)}; preds={}
    for k,m in models().items():
        m.fit(A,ytr); p=m.predict(B); preds[k]=p
        pr,rc,f,_=precision_recall_fscore_support(yte,p,pos_label='positif',average='binary')
        mf=precision_recall_fscore_support(yte,p,average='macro')[2]
        cm=confusion_matrix(yte,p,labels=['negatif','positif'])
        res[k]={'akurasi':accuracy_score(yte,p),'presisi_pos':pr,'recall_pos':rc,'f1_pos':f,'f1_macro':mf,'cm[[TN,FP],[FN,TP]]':cm.tolist()}
    out[tag]=res
    return Xtr,Xte,ytr,yte,v,preds
# 1 utama
Xtr,Xte,ytr,yte,v,preds=split_eval(df,'utama_554')
# McNemar
a=(preds['NB']==yte.values); b=(preds['SVM']==yte.values)
n01=int((~a&b).sum()); n10=int((a&~b).sum())
out['mcnemar']={'NB_salah_SVM_benar':n01,'NB_benar_SVM_salah':n10,'p_exact':binomtest(min(n01,n10),n01+n10,0.5).pvalue if n01+n10>0 else 1.0}
# fitur
vv=vec(); A=vv.fit_transform(Xtr); names=np.array(vv.get_feature_names_out())
nb=MultinomialNB().fit(A,ytr); svm=LinearSVC(random_state=42).fit(A,ytr)
co=svm.coef_[0]; o=np.argsort(co)
out['top_svm_positif']=names[o[::-1][:10]].tolist(); out['top_svm_negatif']=names[o[:10]].tolist()
out['vocab_size']=int(len(names))
# salah klasifikasi
te=df.loc[Xte.index].copy(); te['pred_NB']=preds['NB']; te['pred_SVM']=preds['SVM']
te[['review_text','review_text_preprocessed','rating','sentimen','pred_NB','pred_SVM']].to_csv('prediksi_uji.csv',index=False)
# 2 CV 5-fold pada 554
def cv(d,tag):
    X=d['review_text_preprocessed'].fillna(''); y=d['sentimen']
    skf=StratifiedKFold(5,shuffle=True,random_state=42); r={}
    for k,m in models().items():
        pipe=Pipeline([('t',vec()),('m',m)])
        s=cross_validate(pipe,X,y,cv=skf,scoring={'acc':'accuracy','f1m':'f1_macro','prec':'precision','rec':'recall','f1':'f1'} if False else {'acc':'accuracy','f1m':'f1_macro'})
        r[k]={'akurasi_mean':s['test_acc'].mean(),'akurasi_std':s['test_acc'].std(),'f1_macro_mean':s['test_f1m'].mean(),'f1_macro_std':s['test_f1m'].std(),'folds_acc':s['test_acc'].tolist()}
    out[tag]=r
cv(df,'cv5_utama')
# 3 deduplikasi
d2=df.copy(); d2['review_text_preprocessed']=d2['review_text_preprocessed'].fillna('')
d2=d2[d2['review_text_preprocessed'].str.strip()!='']
d2=d2.drop_duplicates('review_text_preprocessed')
out['dedup_info']={'sebelum':len(df),'sesudah':len(d2),'dist':d2['sentimen'].value_counts().to_dict()}
split_eval(d2,'dedup_split'); cv(d2,'cv5_dedup')
# periode data
m=df.merge(raw[['review_text','review_date','sentiment_label','rating']].drop_duplicates('review_text'),on='review_text',how='left',suffixes=('','_raw'))
dt=pd.to_datetime(m['review_date'],errors='coerce')
out['periode']={'terpetakan':int(dt.notna().sum()),'min':str(dt.min()),'max':str(dt.max()),'tahun':dt.dt.year.value_counts().sort_index().to_dict()}
# sentiment_label bawaan vs rating (data mentah, rating!=3)
r=raw[raw['product_category']=='Olahraga']
out['label_bawaan']=pd.crosstab(r['rating'],r['sentiment_label']).to_dict()
out['kategori_raw']=raw['product_category'].value_counts().to_dict()
out['olahraga_total']=len(r)
json.dump(out,open('hasil_eksperimen.json','w'),indent=1,default=lambda o:float(o) if hasattr(o,'__float__') else str(o))
print(json.dumps(out,indent=1,default=lambda o:float(o) if hasattr(o,'__float__') else str(o)))
