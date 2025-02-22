# BERT Attention Visualization

This project provides a web-based interface for visualizing attention weights in the BERT model. It includes a Flask-based API backend and a Streamlit frontend.

## Features
- Tokenizes input text using BERT's tokenizer.
- Extracts and visualizes attention weights from all layers and heads of the BERT model.
- Provides interactive selection of attention layers and heads.
- Displays attention matrices as heatmaps.

## Installation
### Prerequisites
- Python 3.7+
- pip
- Virtual environment (optional but recommended)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/bert-attention-visualization.git
   cd bert-attention-visualization
   ```

2. Create and activate a virtual environment (optional):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application
### Start the Backend (Flask API)
```sh
python backend.py
```
The API will run on `http://127.0.0.1:5001`.

### Start the Frontend (Streamlit App)
```sh
streamlit run frontend.py
```

## API Endpoints
### `POST /attention`
- **Request Body:**
  ```json
  {
    "text": "Sample input text"
  }
  ```
- **Response:**
  ```json
  {
    "tokens": ["Sample", "input", "text"],
    "num_layers": 12,
    "attention_weights": [[[...]]]
  }
  ```

## Usage
1. Open the Streamlit app in your browser.
2. Enter the text you want to analyze.
3. Click "Visualize Attention" to fetch and display attention weights.
4. Use the sliders to explore attention across different layers and heads.

## Dependencies
- `transformers`
- `torch`
- `flask`
- `streamlit`
- `requests`
- `matplotlib`
- `seaborn`
- `numpy`



