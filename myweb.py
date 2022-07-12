import flask
from flask_cors import CORS
from flask import Flask,render_template, request, session
from myscript import *

#import os
#from werkzeug.utils import secure_filename
#UPLOAD_FOLDER = ""

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

app.secret_key = "bonjour"

## transform string a to an array
def Getlabel(a):
	b = a.split(",")
	for i in range(len(b)):
		b[i]=int(b[i])
	return b	
	
@app.route('/',methods=['POST','GET'])
def index():
	return render_template("index.html")

@app.route('/tsne',methods=['POST','GET'])
def tsne():	
	if(request.method=="POST"):
		Myfun_tsne()	
		session['filename']="static/data_tsne.csv"
		session['rd']='tsne'
		return render_template("cluster_skope-rules.html", filename =session['filename'], rd=session['rd'])
	return render_template("index.html")

@app.route('/pca',methods=['POST','GET'])
def pca():
	if(request.method=="POST"):
		Myfun_pca()
		session['rd']='pca'
		session['filename']="static/data_pca.csv"
		return render_template("cluster_skope-rules.html", filename =session['filename'] ,rd=session['rd'])
	return render_template("index.html")

@app.route('/umap',methods=['POST','GET'])
def umap():	
	if(request.method=="POST"):
		Myfun_umap()
		session['rd']='umap'
		session['filename']="static/data_umap.csv"
		return render_template("cluster_skope-rules.html", filename =session['filename'],rd=session['rd'])
	return render_template("index.html")

@app.route('/run',methods=['POST','GET'])
def run():
	b=Getlabel(request.form.get("ids"))
	if sum(b)==0:
		return '0' ##so that the front-end gives an alert
	else:
		session['number']=sum(b)
		x = Myfun_skope(b)
		session['pf']=str(x[5])[1:-1]
	return {'rule':x[0],'nop':x[1],'avp':x[2],'pfr':x[3],'avpr':x[4],'pf':session['pf']}
	
@app.route('/kmeans',methods=['POST'])
def kmeans():
	a = request.form.get("n")
	N_clusters=int(a)
	z = Mykmeans_load(session['filename'],N_clusters)
	return str(z)

@app.route('/dbscan',methods=['POST'])
def DBSCAN():
	epsilon = request.form.get("epsilon")
	M = request.form.get("M")
	epsilon = float(epsilon)
	M= int(M)	
	(z,N) = Mydbscan_load(session['filename'],epsilon,M)
	return {"label":str(z),"numberofcluster":str(N)}

@app.route('/xgblocal',methods=['POST'])
def XGBlocal():
	J = request.form.get("J")
	if J=='1' or J=='2':
		b=Getlabel(request.form.get("ids"))
	else:
		b =Getlabel(session['pf'])
	return Myfun_xgb1(b,J)+J
	


if __name__ =="__main__":
    app.run(port=2024,host="127.0.0.1",debug=True)



