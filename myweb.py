import flask
from flask_cors import CORS
from flask import Flask,render_template, request, session
from myscript import *


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

CORS(app)

app.secret_key = "bonjour"


def Getlabel(a):
	b = a.split(",")
	for i in range(len(b)):
		b[i]=int(b[i])
	return b	
	
	
	
@app.route('/',methods=['POST','GET'])
def index():
	return render_template("index.html")

###########################################################
@app.route('/banknote',methods=['POST','GET'])
def banknote():
	if request.method == 'POST':
		session['data']='banknote'
		session['original_file']='static/'+session['data']+'/'+session['data']+'.csv'
		session['XGB_SHAP_file']='static/'+session['data']+'/XGB_SHAP.csv'
		session['pb']=problemtype(session['data'])
		return render_template(session['data']+".html")
	return render_template('index.html')

@app.route('/house',methods=['POST','GET'])
def house():
	if request.method == 'POST':
		session['data']='house'
		session['original_file']='static/'+session['data']+'/'+session['data']+'.csv'
		session['XGB_SHAP_file']='static/'+session['data']+'/XGB_SHAP.csv'
		session['pb']=problemtype(session['data'])
		return render_template(session['data']+".html")
	return render_template('index.html')

#############################################################
@app.route('/tsne',methods=['POST','GET'])
def tsne():	
	if(request.method=="POST"):
		filename = session['XGB_SHAP_file']
		pathtosave = "static/"+session['data']+"/data_tsne.csv"
		#Myfun_tsne(pathtosave,filename)
		session['filename']=pathtosave
		session['rd']='tsne'
		return render_template("cluster_skope-rules.html", filename =session['filename'], rd=session['rd'],pb= session['pb'])
	return render_template(session['data']+".html")

@app.route('/pca',methods=['POST','GET'])
def pca():
	if(request.method=="POST"):
		filename = session['XGB_SHAP_file']
		pathtosave = "static/"+session['data']+"/data_pca.csv"
		#Myfun_pca(pathtosave,filename)
		session['filename']=pathtosave
		session['rd']='pca'
		return render_template("cluster_skope-rules.html", filename =session['filename'] ,rd=session['rd'],pb= session['pb'])
	return render_template(session['data']+".html")

@app.route('/umap',methods=['POST','GET'])
def umap():	
	if(request.method=="POST"):
		filename = session['XGB_SHAP_file']
		pathtosave = "static/"+session['data']+"/data_umap.csv"
		#Myfun_umap(pathtosave,filename)
		session['filename']=pathtosave
		session['rd']='umap'
		return render_template("cluster_skope-rules.html", filename =session['filename'],rd=session['rd'],pb= session['pb'])
	return render_template(session['data']+".html")
##################################################

#skope-rules
@app.route('/run',methods=['POST','GET'])
def run():
	b=Getlabel(request.form.get("ids"))
	if sum(b)==0:
		return '0' ##so that the front-end gives an alert
	else:
		session['number']=sum(b)
		x = Myfun_skope(b,session['filename'],session['original_file'])
		session['pf']=str(x[5])[1:-1]
	return {'rule':x[0],'nop':x[1],'avp':x[2],'pfr':x[3],'avpr':x[4],'pf':session['pf']}
#
	

@app.route('/kmeans',methods=['POST'])
def kmeans():
	a = request.form.get("n")
	N_clusters=int(a)
	z = Mykmeans_load(session['filename'],N_clusters)
	return str(z)
#	return the result of kmeans

@app.route('/dbscan',methods=['POST'])
def DBSCAN():
	epsilon = request.form.get("epsilon")
	M = request.form.get("M")
	epsilon = float(epsilon)
	M= int(M)	
	(z,N) = Mydbscan_load(session['filename'],epsilon,M)
	return {"label":str(z),"numberofcluster":str(N)}
# 	return result of dbscan and number of cluster

# do the xgboost on the segment
@app.route('/xgblocal',methods=['POST'])
def XGBlocal():
	J = request.form.get("J")
	if J=='1' or J=='2':
		b=Getlabel(request.form.get("ids"))
	else:
		b =Getlabel(session['pf'])
	return Myfun_xgb(b,J,session['XGB_SHAP_file'],session['original_file'],session['pb'])+J
#	return a string contains the feature importances, and an index J(will be used in frontend)	


if __name__ =="__main__":
    app.run(port=2024,host="127.0.0.1",debug=True)



