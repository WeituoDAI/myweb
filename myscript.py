import pandas as pd
import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def mypca(X):
	pca = PCA(n_components=2,random_state=500)
	return pca.fit_transform(X)

def myumap(X):
	reducer=umap.UMAP(n_components=2,random_state=0)
	return reducer.fit_transform(X)
	
def mytsne(X):
	tsne=TSNE(n_components=2,init='random',learning_rate=200,random_state=0)
	return tsne.fit_transform(X)

def Dataset():
	return pd.read_csv('static/BN_XGB_SHAP.csv', encoding='latin') 

def getvariable(data):
	a = data.columns[:-1]
	b= len(a)
	return a[:b//2],a[b//2:],data.columns[-1]


def Myfun0(fun,pathtosave):
	data = Dataset()
	(variable, shapvariable, target) = getvariable(data)
	X=data[variable]
	X_shap = data[shapvariable]
	y = data[target]
	newX=fun(StandardScaler().fit_transform(X))
	newX_shap = fun(X_shap)
	newdata = pd.DataFrame()
	newdata['x0']=newX[:,0]
	newdata['x1']=newX[:,1]
	newdata['shap0']=newX_shap[:,0]
	newdata['shap1']=newX_shap[:,1]
	newdata['label']=y
	newdata.to_csv(pathtosave,header=True,index=None)
	
def Myfun_tsne():
	Myfun0(mytsne,"static/data_tsne.csv")

def Myfun_pca():
	Myfun0(mypca,"static/data_pca.csv")

def Myfun_umap():
	Myfun0(myumap,"static/data_umap.csv")
		
def Myskope(X,y,Feature_names):
	skp = SkopeRules(
	 max_depth=3,
	 max_depth_duplication=2,
	 n_estimators=30,
	 precision_min=0.3,
	 recall_min=0.05,
	 feature_names=Feature_names,
	 random_state=0
	 )         
	skp.fit(X, y)
	myrule = skp.rules_[0][0]
	precision = round(skp.rules_[0][1][0],2)
	recall = round(skp.rules_[0][1][1],2)
	
	
	prediction = skp.predict_top_rules(X,1)
	
	return {"rule":myrule,"precision":precision,"recall":recall},prediction


def Myfun_skope(a,filetoread):
	dataset=pd.read_csv('static/banknote.csv', encoding='latin') 	
	filename = filetoread
	newdata=pd.read_csv(filename, encoding='latin')	
	newdata['label2']=a
	
	X=dataset.drop(axis=1, labels=dataset.columns[-1],inplace=False)		
	Feature_names=dataset.columns[:-1]
	(rules,prediction)=Myskope(X,a,Feature_names)
	
	p=0
	q=0
	for i in range(len(a)):
		if a[i]==1:
			p=p+newdata['label'][i]
		if prediction[i]==1:
			q=q+newdata['label'][i]
	q=round(q/sum(prediction),2)			
	p=round(p/sum(a),2)	
	
	return rules,float(sum(a)),p,float(sum(prediction)),q,list(prediction)
	

def Myfun_skope_umap(a):
	return Myfun_skope(a,"static/data_umap.csv")

def Myfun_skope_pca(a):
	return Myfun_skope(a,"static/data_pca.csv")

def Myfun_skope_tsne(a):
	return Myfun_skope(a,"static/data_tsne.csv")

	
def mykmeans(X,N_clusters):
	kmeans = KMeans(n_clusters=N_clusters, random_state=0)
	return list(kmeans.fit_predict(X))
		
def Mykmeans_load(filename,N_clusters):
	data = pd.read_csv(filename,encoding='latin')
	X = data[['shap0','shap1']]
	z= mykmeans(X,N_clusters)
	return z
	
def mydbscan(epsilon,M,X):
	z = DBSCAN(eps = epsilon, min_samples = M+1,algorithm='brute').fit_predict(X)
	if -1 in z:
		z=z+1
	return list(z)

def Mydbscan_load(filename,epsilon,M):
	data = pd.read_csv(filename,encoding='latin')
	X = data[['shap0','shap1']]
	z= mydbscan(epsilon,M,X)
	N=len(np.unique(z))
	return z,N

