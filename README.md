# Comprehensive Comparison of Physics-Informed Machine Learning for Data-Driven Modeling of HVAC Systems

This repository provides a complete implementation of a comparison study between **non-physics-informed** and **physics-informed machine learning (PIML)** approaches for data-driven modeling of HVAC (Heating, Ventilation, and Air Conditioning) systems.

## Repository Structure

The repository includes source code, data, and pretrained model weights for 12 different models, organized into three main categories:

### 1. Resistor-Capacitor (RC) Grey-Box Model
- `RC.py` – Classical grey-box model for HVAC thermal dynamics.

### 2. Non-Physics-Informed Machine Learning Models
- `ANN.py` – Artificial Neural Network  
- `LR.py` – Linear Regression  
- `GP.py` – Gaussian Process  
- `NODE.py` – Neural Ordinary Differential Equation  
- `ResNet.py` – Residual Network  
- `RNN.py` – Recurrent Neural Network  

### 3. Physics-Informed Machine Learning Models
- `BoundedNN.py` – Neural network with bounded physical constraints  
- `SSN.py` – Structured State-Space Neural Network  
- `MinMax.py` – Physics-guided min-max bounded modeling  
- `LatticeNN.ipynb` – TensorFlow Lattice-based physics-informed neural network

### Data and Weights
- `.pth` files – Pretrained weight files for initializing neural networks  

---

## Getting Started

### 1. Clone the Repository
```python
git clone https://github.com/nxt-lab/Code-PIML-for-HVAC-systems-public.git

cd hvac-piml-comparison
```

### 2. Install Required Packages
Install all required Python libraries from the `requirements.txt` file:

```python
pip install -r requirements.txt
```

### 3. Run an Example Model

Let’s run the **ANN model** as an example:

```python
python ANN.py
```

You will be prompted to enter:
```python
"Choose dataset [filtered/original]: "
```
- Choose the scenario by typing: `filtered` or `original`

```python
"Enter number of training samples (e.g., 100): "
```
- Choose the training data size by typing: a number of N, where N = 20, 40, ..., 140
- For example, type `100`.
