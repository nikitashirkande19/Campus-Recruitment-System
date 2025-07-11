import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, session
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# Secret key and upload folder
app.secret_key = '31206b7b80bb51fafd95fcea504e7edc'
app.config['UPLOAD_FOLDER'] = 'static'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and Flask-Migrate
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Contact model
class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
# Create flask app
#app = Flask(__name__)
model = pickle.load(open("xgboost_model.pkl", "rb"))


# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 12)
	loaded_model = pickle.load(open("xgboost_model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route("/home")
def Home():
    print('Request for index page received')
    return render_template("index.html")

@app.route("/result", methods = ["POST"])
def result():
    print('Request for predict page received')
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)== 1:
            prediction ='Placed'
        else:
            prediction ='Not Placed'
    return render_template("result.html", prediction_text = prediction)



#Routes
@app.route('/')
def main_page():
    return render_template('main.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(full_name=full_name, email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('Home'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        new_message = Contact(name=name, email=email, message=message)
        db.session.add(new_message)
        db.session.commit()

        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('about'))

    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)