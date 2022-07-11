# myweb

This is a web which can only run on the local machine
(for test use, can at least run, regardless of the beauty)

We support a dataest banknote.csv
We have already used xgboost, and calculated the shap values. (stored in BN-XGB-SHAP.csv)

We can use 3 methods (pca,umap,t-sne) to reduce the dimension of the data/shapvalues and visualise them

Then we can use Skope-rules to obtain a rule(top rule) of a segment of points chosen,
we can choose the segment by hand, or by kmeans, dbscan


Just type "python myweb.py" in the terminal to run the web.


Environment:
Linux(Ubuntu)

Front-end	D3.js

Back-end	python3 Flask

Packages demanded (umap,t-sne,pca,Skope-rules,numpy,pandas,sklearn....) 




