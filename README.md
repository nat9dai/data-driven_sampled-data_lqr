# Data-Driven Sampled-Data LQR

Implementation of the data-driven sampled-data LQR algorithm from:

**"Data-Driven Sampled-Data Optimal Control"**  
[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5503850](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5503850)

## Quick Start

```bash
python main.py
```

## Overview

This repository implements a data-driven approach to sampled-data LQR control that:
- Learns optimal control gains directly from system trajectories
- Handles continuous-time costs with discrete-time measurements
- Requires no model knowledge beyond system dimensions

## Files

- `main.py` - Main simulation script
- `system.py` - Cart-pole system dynamics
- `controller.py` - DD-SDLQR and SD-LQR controllers
- `simulation.py` - Simulation environment
- `visualiser.py` - Plotting utilities

## Requirements

```bash
pip install numpy scipy matplotlib cvxpy
```

## Citation

If you use this code, please cite:
```
@article{gerdpratoom5503850data,
  title={Data-Driven Sampled-Data LQR: Certainty-Equivalence Control via Lifted Cost and Riccati Analysis},
  author={Gerdpratoom, Nuthasith and Rantzer, Anders and Yamamoto, Kaoru},
  journal={Available at SSRN 5503850}
}