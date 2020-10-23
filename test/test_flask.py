# Created by Patrick Kao
from flask import Flask, render_template

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config['DEBUG'] = True


@app.route("/", methods=["POST", "GET"])
def run():
    return render_template('main.html', )


if __name__ == "__main__":
    app.run(debug=True)
