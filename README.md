LTI closed-loop simulation

This small repo contains a simple simulator for a discrete-time linear time-invariant system

    x_{k+1} = A x_k + B u_k,    u_k = K x_k

Files
- `main.py`: Implements `simulate(A,B,K,x0,N,noise_std=0.0,random_seed=None)` and a demo that runs when invoked as a script.
- `requirements.txt`: Lists `numpy` and optionally `matplotlib` for plotting.

Quick start

1. (Optional) Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies (optional if you already have numpy):

   pip install -r requirements.txt

3. Run the demo:

   python3 main.py

The demo prints the closed-loop eigenvalues, final state, and will show a plot if `matplotlib` is installed.
