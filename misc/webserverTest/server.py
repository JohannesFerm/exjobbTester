from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder="templates")

def record(label):
    ...
    #In here record audio clips

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/label/<label>", methods = ['GET'])
def readLabel(label):
    record(label)
    return label

if __name__ == '__main__':
    app.run()

