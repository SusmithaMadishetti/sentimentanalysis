from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
#from Prj import displayGUI
import food_svm
import politics_svm
app = Flask(__name__)

@app.route("/")
def home():
     return render_template('home.html')
@app.route("/index")

@app.route('/trend', methods=['GET', 'POST'])
def trend():
   print "entered in /trend >>>"
   message = None
   if request.method == 'GET':
      tweettopic = request.args.get('tweettopic')
      print tweettopic
      searchstr = request.args.get('keyword')
      print searchstr
      if (tweettopic == 'politics'):
          execfile("politics_svm.py")
          return render_template('result_politics.html')
      elif tweettopic == 'food':
          food_svm.displayGUI()
          return render_template('result_food.html')
   if request.method == 'POST':
        print "entered in if POST >>>"
        #datafromjs = request.form['mydata']
        datafromjs = request.form['tweettopic']
        print datafromjs
        result = "return this"
        resp = make_response('{"response": '+result+'}')
        resp.headers['Content-Type'] = "application/json"
        food_svm.displayGUI()
        return 'resp'
        #return render_template('result.html')



if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)