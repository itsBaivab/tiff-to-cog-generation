from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "Test endpoint is working!"})

@app.route('/post-metadata', methods=['POST'])
def post_endpoint():
    data = request.get_json()
    print(data)
    return jsonify({"message": "Data received", "data": data})

if __name__ == '__main__':
    app.run(debug=True)