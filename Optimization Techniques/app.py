from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)

def hungarian_algorithm(cost_matrix):
    # Check if the matrix is square
    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        # Make the matrix square by adding dummy rows or columns
        max_dim = max(cost_matrix.shape)
        diff = abs(cost_matrix.shape[0] - cost_matrix.shape[1])
        if cost_matrix.shape[0] < cost_matrix.shape[1]:
            dummy_rows = np.zeros((diff, max_dim))
            cost_matrix = np.vstack((cost_matrix, dummy_rows))
        else:
            dummy_cols = np.zeros((max_dim, diff))
            cost_matrix = np.hstack((cost_matrix, dummy_cols))
    
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate total cost
    total_cost = cost_matrix[row_indices, col_indices].sum()
    
    # Remove assignments involving dummy rows or columns
    assignment = [(row, col) for row, col in zip(row_indices, col_indices)
                  if row < cost_matrix.shape[0] and col < cost_matrix.shape[1]]
    
    return assignment, total_cost

@app.route('/', methods=['GET', 'POST'])
def index():
    assignment = None
    total_cost = None
    if request.method == 'POST':
        cost_matrix_text = request.form['cost_matrix']
        cost_matrix = parse_input(cost_matrix_text)
        assignment, total_cost = hungarian_algorithm(cost_matrix)
    return render_template('index.html', assignment=assignment, total_cost=total_cost)

def parse_input(input_text):
    rows = input_text.strip().split('\n')
    matrix = [list(map(int, row.strip().split())) for row in rows]
    return np.array(matrix)

if __name__ == '__main__':
    app.run(debug=True)
