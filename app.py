import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'signin'
migrate = Migrate(app, db)  # Initialize Flask-Migrate

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

matplotlib.use('Agg')  # Use non-interactive backend

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    
    # Establishing relationship with Patient
    patients = db.relationship('Patient', backref='user', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    condition = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created!', 'success')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('user_dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('signin.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route('/patient_register', methods=['GET', 'POST'])
@login_required
def patient_register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        condition = request.form['condition']
        patient = Patient(name=name, age=age, condition=condition, user_id=current_user.id)
        db.session.add(patient)
        db.session.commit()
        flash('Patient registered successfully', 'success')
        return redirect(url_for('patient_dashboard'))
    return render_template('patient_register.html')

@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    patients = Patient.query.filter_by(user_id=current_user.id).all()
    return render_template('patient_dashboard.html', patients=patients)

@app.route('/upload')
@login_required
def upload_file():
    return render_template('upload.html', tool_name="Athena - Analytics Tool for Healthcare Data")

@app.route('/uploader', methods=['GET', 'POST'])
@login_required
def uploader():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('analyze.html', filename=filename, titles=df.columns.values, tool_name="Athena - Analytics Tool for Healthcare Data")
    return redirect(url_for('upload_file'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    x_axis = request.form['x_axis']
    y_axis = request.form['y_axis']
    chart_type = request.form['chart_type']
    filename = request.form['filename']
    
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    plot_file_path = os.path.join(app.root_path, 'static', 'plot1.png')
    if chart_type == 'scatter':
        sns.scatterplot(data=df, x=x_axis, y=y_axis)
    elif chart_type == 'line':
        sns.lineplot(data=df, x=x_axis, y=y_axis)
    elif chart_type == 'bar':
        sns.barplot(data=df, x=x_axis, y=y_axis)
    elif chart_type == 'box':
        sns.boxplot(data=df, x=x_axis, y=y_axis)
    
    plt.title(f"{chart_type.capitalize()} Plot")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(plot_file_path)
    plt.close()
    
    df.dropna(inplace=True)
    X = df.drop(columns=[y_axis])
    y = df[y_axis]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    accuracy = 100 - rmse  # Calculate accuracy

    # Save model report as CSV
    model_report_path = os.path.join(app.root_path, 'static', 'model_report.csv')
    model_report_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    model_report_df.to_csv(model_report_path, index=False)

    return render_template('result.html', tool_name="Athena - Analytics Tool for Healthcare Data", accuracy=accuracy, plot_url=url_for('static', filename='plot1.png'))

@app.route('/download_pdf/<accuracy>')
@login_required
def download_pdf(accuracy):
    accuracy = float(accuracy)
    pdf_file_path = os.path.join(app.root_path, 'static', 'result.pdf')
    plot_file_path = os.path.join(app.root_path, 'static', 'plot1.png')
    model_report_path = os.path.join(app.root_path, 'static', 'model_report.csv')

    # Generate PDF report
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    c.drawString(100, 750, "Athena - Analytics Tool for Healthcare Data")
    c.drawString(100, 730, f"Model Accuracy (inverted RMSE as percentage): {accuracy:.2f}%")
    
    # Add patient details
    patients = Patient.query.filter_by(user_id=current_user.id).all()
    if patients:
        c.drawString(100, 700, "Patient Details:")
        y_position = 680
        for patient in patients:
            patient_info = f"Name: {patient.name}, Age: {patient.age}, Condition: {patient.condition}"
            c.drawString(100, y_position, patient_info)
            y_position -= 20
    
    # Add plot image
    if os.path.exists(plot_file_path):
        c.drawImage(ImageReader(plot_file_path), 100, 400, width=400, height=300)
    
    # Add model report data
    if os.path.exists(model_report_path):
        model_report_df = pd.read_csv(model_report_path)
        c.drawString(100, 370, "Model Report:")
        y_position = 350
        for i, row in model_report_df.iterrows():
            c.drawString(100, y_position, f"Actual: {row['Actual']}, Predicted: {row['Predicted']}")
            y_position -= 20
            if y_position < 50:
                c.showPage()
                y_position = 750
    
    c.save()

    return send_file(pdf_file_path, as_attachment=True)

@app.route('/download/<filename>')
@login_required
def download_file(filename):
    return send_file(os.path.join(app.root_path, 'static', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
