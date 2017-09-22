from flask import Flask, render_template, request, jsonify
import intent_classify

app = Flask(__name__)

@app.route("/message", methods=['POST'])
def index() :
    intent = intent_classify.intent_classify(False, request.form['msg'])
    return jsonify({'text':intent_classify.get_response(intent)})

@app.route("/")
def main():
    return render_template('index.html')

def init():
    app.run()

