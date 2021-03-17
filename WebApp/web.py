# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/report')
def report():
    return render_template('report.html')
@app.route('/spam')
def quality():
    return render_template('spam.html')
@app.route('/predict',methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    
    #Std Scaling
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    final_features=sc.fit_transform(final_features)

    prediction = model.predict(final_features) 

    if(prediction==1):
        
        return render_template('spam.html', predict=' :Salary will be >50K')
    else:
         return render_template('spam.html', predict=' :Salary will be <=50K')
if __name__ == "__main__":
    app.run()
    
