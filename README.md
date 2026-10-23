# Dark Store Location Optimizer

## Files
- `app.py`: Streamlit app and MILP optimization model.
- `requirements.txt`: Python dependencies.

## Run locally (Windows)
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ .
2. Open Command Prompt in this folder.
3. Create a virtual environment:
   `py -m venv .venv`
4. Activate it:
   `.venv\Scripts\activate`
5. Install dependencies:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
6. Start the app:
   `streamlit run app.py`
7. Open the local URL printed in the terminal, usually http://localhost:8501 .

## Model assumptions
- Geography is reproducible synthetic data inside a Chennai-area bounding box, not real addresses.
- Revenue contribution, per-km delivery cost, and fixed store cost are editable illustrative values.
- Customers may remain unserved if no selected store is within the maximum distance.
- The model maximizes total contribution minus distance-based delivery cost and fixed store costs.
- It is a binary MILP solved using `scipy.optimize.milp` (HiGHS). A time-limited solve may return a feasible incumbent that is not proven globally optimal.

## Required experiment
Click **Run 2, 5, 10, 20, 25-store comparison**. It always generates 10,000 customers and 50 candidate sites using the same random seed and financial assumptions, then solves the five store-limit cases.

## Deployment
Use a host that supports Streamlit, Python packages, and SciPy's MILP solver. Add `app.py` and `requirements.txt` to the repository and configure the host to run:
`streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`

The assignment mentions https://share.streamline.io. Verify the platform's current upload, repository, runtime, and port requirements; do not assume it supports arbitrary Streamlit apps without checking its documentation. If it does not, use a Streamlit-compatible host approved by your instructor and submit that public URL.
