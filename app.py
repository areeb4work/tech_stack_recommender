#DecodeLabs Project 3 - Tech Stack Recommender 
#Based on available dataset in 'raw skills' csv file
#4-step Pipeline - Ingestion -> Scoring -> Sorting -> Filtering
#Author Areeb Ahsan
import flask
from recommender import TechStackRecommender

app = flask.Flask(__name__)
engine = TechStackRecommender("raw_skills.csv")


@app.route("/")
def home():
    skill_suggestions = engine.all_known_skills()
    return flask.render_template("index.html", skill_suggestions=skill_suggestions)


@app.route("/recommend", methods=["POST"])
def recommend():
    data = flask.request.get_json()
    user_skills = data.get("skills", [])

    # clean up input: strip whitespace, drop empties
    user_skills = [s.strip() for s in user_skills if s.strip()]

    if len(user_skills) < 3:
        return flask.jsonify({
            "error": "Please enter at least 3 skills for accurate matching."
        }), 400

    output = engine.recommend(user_skills, top_n=3)
    return flask.jsonify(output)


if __name__ == "__main__":
    print("\n🚀 Tech Stack Recommender running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
