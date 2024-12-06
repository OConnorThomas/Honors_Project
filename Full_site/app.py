from flask import Flask, render_template, request, redirect, url_for
from query import load_and_predict, get_full_finances

app = Flask(__name__)

# Route for Analysis page (handles both GET and POST)
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    result = None
    info = {'Query ':'Model'}

    if request.method == 'POST':
        stock_input = request.form['stock_input']
        result = load_and_predict(stock_input)
        info = get_full_finances(stock_input)

    # Format the result to 2 decimal places
    formatted_result = f"{result:.2f} %" if result is not None else 'Query : Model'
    formatted_info = ''.join([f'{key}: {value}<br />' for key, value in info.items()])

    return render_template('analysis.html', INFO=formatted_info, RESULT=formatted_result)


# Route for Home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for Finances page
@app.route('/predictions')
def finances():
    return render_template('predictions.html')

# Route for Algorithms page
@app.route('/algorithms')
def algorithms():
    return render_template('algorithms.html')

# Route for Algorithms page features subpage
@app.route('/algorithms_fin')
def algorithms__fin():
    return render_template('algorithms.html')

# Route for Algorithms page target subpage
@app.route('/algorithms_tgt')
def algorithms__tgt():
    return render_template('algorithms.html')

if __name__ == '__main__':
    app.run(debug=True)
