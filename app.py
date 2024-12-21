import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from Model.StockPrediction import run_stock_prediction

# Configure Flask app
app = Flask(__name__)
app.secret_key = 'secret_key_for_csrf_protection'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run stock prediction
            next_day_value, output_file = run_stock_prediction(filepath)
            
            # Render result template with the next day value and graph image path
            return render_template('result.html', 
                                   next_day_value=next_day_value, 
                                   graph_path=output_file)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
