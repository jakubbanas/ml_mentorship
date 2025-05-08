from flask import Flask, jsonify
import joblib
from jira import JIRA
import os

JIRA_URL = os.getenv("JIRA_URL")
JIRA_API_KEY = os.getenv("JIRA_API_KEY")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")

app = Flask(__name__)
jira = JIRA(JIRA_URL, basic_auth=(JIRA_EMAIL, JIRA_API_KEY))

model = joblib.load("complexity_classifier.joblib")
model_global = joblib.load("complexity_classifier_all.joblib")


@app.route("/<issue_id>", methods=["GET"])
def classify(issue_id):
    try:
        issue = jira.issue(issue_id)
    except Exception as e:
        return jsonify({"error": str(e.text)})

    text = (
        issue.fields.summary + "\n" + issue.fields.description
        if issue.fields.description
        else issue.fields.summary
    )

    prediction_local = model.predict([text])
    prediction_global = model_global.predict([text])

    return jsonify({"ctools": prediction_local[0], "global": prediction_global[0]})


if __name__ == "__main__":
    app.run(debug=True)
