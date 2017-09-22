from flask import Flask, render_template, request, jsonify
import intent_classify

app = Flask(__name__)

@app.route("/receiveSentence",methods=['POST'])
def index() :
    intent = intent_classify.intent_classify(request.form.get("firstname"))
    return jsonify({'var1':intent_classify.get_response(intent)})

@app.route("/")
def main():
    return render_template('entry.html')

def init():
    app.run()

if __name__ == '__main__':
    app.run()