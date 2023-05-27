'''
	Contoh Deloyment untuk Domain Natural Language Processing (NLP)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from joblib import load
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from fungsi import *

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

stopwords_ind = None
key_norm      = None
factory       = None
stemmer       = None
vocab         = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]		
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk string input 
	text_input = ""
	
	if request.method=='POST':
		# Set nilai string input dari pengguna
		text_input = request.form['data']
					
		# Memuat model TF-IDF Vectorizer dari file .pickle
		with open('../tfidf_model_selected.pickle', 'rb') as handle:
			tfidf_vectorizer = pickle.load(handle)

		# Memuat model SVM dari file .pickle
		with open('../svm_model_selected.pickle', 'rb') as handle:
			loaded_model = pickle.load(handle)

		# Contoh kalimat baru untuk inferensi
		kalimat_baru = [text_input]

		# Melakukan transformasi pada kalimat baru
		kalimat_baru_tfidf = tfidf_vectorizer.transform(kalimat_baru)

		# Melakukan inferensi dengan kalimat baru menggunakan model yang ada
		hasil_inferensi = loaded_model.predict(kalimat_baru_tfidf)

		print(hasil_inferensi)
		if(hasil_inferensi=="no"):
			hasil_prediksi = "Teks Normal"
		else:
			hasil_prediksi = 'Teks Cyberbullying'
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"data": hasil_prediksi,
		})

# =[Main]========================================

if __name__ == '__main__':
	model = load('../model_1.joblib')
	print(model)
	
	# Setup
	stopwords_ind = stopwords.words('indonesian')
	stopwords_ind = stopwords_ind + more_stopword
	
	key_norm = pd.read_csv('../key_norm.csv')
	
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	
	vocab = pickle.load(open('../kbest_feature.pickle', 'rb'))

	
	# Load model yang telah ditraining

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)