from flask import Flask, request, render_template
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df.head()
df.isna().sum()
df.info()
df.describe()
print(df.columns)
X = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1)
y = df['Personal Loan']

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form.get('age'))
        exp = float(request.form.get('exp'))
        income = float(request.form.get('income'))
        family = int(request.form.get('family'))
        ccavg = float(request.form.get('ccavg'))
        edu = int(request.form.get('edu'))
        mortgage = float(request.form.get('mortgage'))
        sercurity = int(request.form.get('secur'))
        cd = int(request.form.get('cd'))
        online = int(request.form.get('online'))
        cred = int(request.form.get('cred'))

        print("***********",exp)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        print('\nConfusion Matrix:')
        print(conf_matrix)
        print('\nClassification Report:')
        print(classification_rep)
        new_data = np.array([[age, exp, income, family, ccavg, edu, mortgage, sercurity, cd, online, cred]])
        new_data_scaled = scaler.transform(new_data)
        # new_df = pd.DataFrame(new_data)
        # new_data_scaled = scaler.transform(new_df)
        new_prediction = model.predict(new_data_scaled)
        # print("***********",result)s
        result=new_prediction[0]
        if result == 0:
            result1 = "Approved"
        else:
            result1 = "Approval Failed"
        return render_template('index.html', predicted_cost=result1)
    else:
        return render_template('index.html')

@app.route('/visual')
def visual():
    if not os.path.exists("static/plots"):
        os.makedirs("static/plots")

    # Clear old plots (optional)
    plt.clf()

    # Age Distribution
    plt.figure()
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.savefig('static/plots/age.png')
    plt.close()

    # Income Distribution
    plt.figure()
    sns.histplot(df['Income'], bins=20, kde=True)
    plt.title('Income Distribution')
    plt.savefig('static/plots/income.png')
    plt.close()

    # Education Count
    plt.figure()
    sns.countplot(x='Education', data=df)
    plt.title('Education Distribution')
    plt.savefig('static/plots/education.png')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('static/plots/heatmap.png')
    plt.close()

    return render_template('index2.html')

if __name__=="__main__":
    app.run(debug=True)