import pandas as pd
import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
from skrules import SkopeRules
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import xgboost as xgb


def mypca(X):
	pca = PCA(n_components=2,random_state=500)
	return pca.fit_transform(X)

def myumap(X):
	reducer=umap.UMAP(n_components=2,random_state=0)
	return reducer.fit_transform(X)
	
def mytsne(X):
	tsne=TSNE(n_components=2,init='random',learning_rate=200,random_state=0)
	return tsne.fit_transform(X)

def Dataset(filename):
	return pd.read_csv(filename, encoding='latin') 

# get the feature names
def getvariable(data):
	a = data.columns[:-1]
	b= len(a)
	return a[:b//2],a[b//2:],data.columns[-1]

def Myfun0(fun,pathtosave,filename):
	data = Dataset(filename)
	(variable, shapvariable, target) = getvariable(data)
	X=data[variable]
	X_shap = data[shapvariable]
	y = data[target]
	#the initial data values needs the standarization(but the shap values does not)
	newX=fun(MinMaxScaler().fit_transform(X))
	newX_shap = fun(X_shap)
	#save the file with data after pca()/umap()/tsne()
	newdata = pd.DataFrame()
	newdata['x0']=newX[:,0]
	newdata['x1']=newX[:,1]
	newdata['shap0']=newX_shap[:,0]
	newdata['shap1']=newX_shap[:,1]
	newdata['label']=y
	newdata.to_csv(pathtosave,header=True,index=None)
	
def Myfun_tsne(filename_tsne,filename):
	Myfun0(mytsne,filename_tsne,filename)

def Myfun_pca(filename_pca,filename):
	Myfun0(mypca,filename_pca,filename)

def Myfun_umap(filename_umap,filename):
	Myfun0(myumap,filename_umap,filename)
		
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
	
	#this prediction is the prediction of skope-rules
	prediction = skp.predict_top_rules(X,1)
	
	return {"rule":myrule,"precision":precision,"recall":recall},prediction

def Myfun_skope(a,filename,original_filename):
	
	dataset=Dataset(original_filename) 		
	X=dataset.drop(axis=1, labels=dataset.columns[-1],inplace=False)		
	Feature_names=dataset.columns[:-1]
	(rules,prediction)=Myskope(X,a,Feature_names)
	
	Label = Dataset(filename)['label']#this is the prediction of xgboost
	
	p=0#average prediction(xgboost) of points selected
	q=0#average prediction(xgboost) of points triggered by the rule
	for i in range(len(a)):
		if a[i]==1:
			p=p+Label[i]
		if prediction[i]==1:
			q=q+Label[i]
	q=round(q/sum(prediction),2)			
	p=round(p/sum(a),2)	
	
	return rules,float(sum(a)),p,float(sum(prediction)),q,list(prediction)
	## Rule,Nb of points selected,average prediction.Nb of points triggered by the rule,average prediction of these points,prediction.


	
def mykmeans(X,N_clusters):
	kmeans = KMeans(n_clusters=N_clusters, random_state=0)
	return list(kmeans.fit_predict(X))
		
def Mykmeans_load(filename,N_clusters):
	data = Dataset(filename)
	X = data[['shap0','shap1']]
	z= mykmeans(X,N_clusters)
	return z
	
def mydbscan(epsilon,M,X):
	z = DBSCAN(eps = epsilon, min_samples = M+1,algorithm='brute').fit_predict(X)
	# the result of dbscan is an array, with value (-1)0,1,2,...,N-1 (N is number of clusters)
	# point with value -1 is the point isolate (please check the theory of dbscan)
	if -1 in z:
		z=z+1
	return list(z)

def Mydbscan_load(filename,epsilon,M):
	data = Dataset(filename)
	X = data[['shap0','shap1']]
	z= mydbscan(epsilon,M,X)
	N=len(np.unique(z))
	return z,N
	

def myxgbclf(X,y):
	clf = xgb.XGBClassifier(
		n_estimators=50,
		enable_categorical=False,  #  si True, tree_method = "gpu_hist" or "hist" or "approx"
		tree_method="approx",      # "gpu_hist" or "hist" or "approx" or "exact"
		random_state=0
		#max_depth,booster,tree_method,learning_rate ...
	)
	clf.fit(X,y)
	return clf.feature_importances_

def myxgbrg(X,y):
	rg = xgb.XGBRegressor(
		n_estimators=50,
		enable_categorical=False,  #  si True, tree_method = "gpu_hist" or "hist" or "approx"
		tree_method="approx",      # "gpu_hist" or "hist" or "approx" or "exact"
		random_state=0
		#max_depth,booster,tree_method,learning_rate ...
	)
	rg.fit(X,y)
	return rg.feature_importances_

# a is an 0-1 array of size = data.length. it corresponds a subdataset (1-0 means inclus or not)
def Myfun_xgb(a,J,filename,original_filename,pb):
	data = Dataset(filename)
	(variable, shapvariable, target) = getvariable(data)
	X=data[variable]
	
	if(J=='1'or J=='3'):
		y = data[target]
	elif(J =='2'or J=='4'):
		dataset =Dataset(original_filename)		
		y = dataset[dataset.columns[-1]]

	X_new = []
	m = len(variable)
		
	for i in range(m):
		X_new.append([])
	y_new = []
	
	#get the new dataset corresponding to the segment (a)
	for i in range(len(y)):
		if a[i]==1:
			y_new.append(y[i])
			for j in range(m):
				X_new[j].append(X.iloc[i,j])
	X_NEW = pd.DataFrame()
	for i in range(m):
		X_NEW[variable[i]]=X_new[i]
	
	if(pb=='classification'):	
		fti = myxgbclf(X_NEW,y_new)
	else:
		fti = myxgbrg(X_NEW,y_new)
	z = {}
	for i in range(m):
		z[variable[i]]=fti[i]
	# sort the feature according to feature importance	
	c = sorted(z.items(),key = lambda d:d[1],reverse =True)
	d = {}
	for i in range(len(c)):
    		d[c[i][0]]=round(float(c[i][1]),3)
	return 'Feature Importances: '+ str(d)
	


def problemtype(data):
	if data =='banknote':
		return 'classification'
	elif data =='house':
		return 'regression'
	
	

