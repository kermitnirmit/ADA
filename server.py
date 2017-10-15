from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(_name_)

@app.route('/upload')
def uploading_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename()
      f.save(secure_filename('/Users/krrishdholakia/Documents/AdaHackGT'))
      return 'file uploaded successfully'

if _name_ == '_main_':
   app.run(debug = True)
