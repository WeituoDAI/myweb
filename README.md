# myweb

This is a web which can only run on the local machine
(for test use, can at least run, regardless of the beauty)

We support two dataest banknote.csv(1352x5 classification), house.csv(20640x9 regression) 
We have already used xgboost, and calculated the shap values. (stored in BN-XGB-SHAP.csv)

We can use 3 methods (pca,umap,t-sne) to reduce the dimension of the data/shapvalues and visualise them.
(Because t-sne umap run somehow slowly, I store already the results of tsne,umap,pca in the static/ (so I comment some part of code in the myweb.py))

Then we can use Skope-rules to obtain a rule(top rule) of a segment of points chosen,
we can choose the segment by hand, or by kmeans, dbscan (Advice: do not use dbscan to the house.csv dataset, very slow sometimes)


Just type "python myweb.py" in the terminal to run the web.


Environment:
Linux(Ubuntu)

Front-end	D3.js

Back-end	python3 Flask

Packages demanded (umap,t-sne,pca,Skope-rules,numpy,pandas,sklearn....) 




