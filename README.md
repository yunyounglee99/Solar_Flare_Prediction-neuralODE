# Solar Magnetic Field Prediction with Neural ODE

_A physics-informed, continuous-time model for forecasting the Sun’s surface magnetic vector field._

## 👀 Sample

<img width="518" alt="스크린샷 2025-05-14 오전 10 20 30" src="https://github.com/user-attachments/assets/f3f4ecff-eaae-4c7d-be67-55993b642d9a" />

---

## 📄 Overview

This repository implements a **Neural Ordinary Differential Equation** (Neural ODE) to predict the temporal evolution of the solar magnetic field components $[B_x, B_y, B_z]$. By learning continuous dynamics directly from time-series magnetogram data, our approach captures fine-grained temporal patterns and enforces physical consistency via a divergence-based regularization.

---

## 🚀 Motivation

- **Continuous dynamics**  
  Solar magnetic fields evolve smoothly, with events (e.g., flares, coronal mass ejections) occurring at irregular intervals.

- **Limitations of discrete-step models (RNNs, ConvRNNs)**  
  - Fixed time steps—poor handling of irregular sampling  
  - Vanishing/exploding gradients over long sequences  
  - High memory footprint for backprop through time

- **Why Neural ODE?**  
  - Models the instantaneous rate of change:  
    $$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)$$  
  - Supports **adaptive time stepping** (e.g., Dormand–Prince solver)  
  - Reduces memory via the **adjoint sensitivity method**  
  - Naturally enforces **smooth, physically plausible predictions**

---

## 📊 Dataset

![image](https://github.com/user-attachments/assets/9da24f8f-58eb-4d10-b0bf-51928149cd77)


<img width="838" alt="스크린샷 2025-05-14 오전 10 41 38" src="https://github.com/user-attachments/assets/b1de4478-3de2-437d-855b-ea30d9a67b54" />


We use the **HMI** (Helioseismic and Magnetic Imager) vector magnetogram data from NASA’s Solar Dynamics Observatory:

1. **Data Selection**  
   - Extract 2D grids of $B_x$, $B_y$, and $B_z$ at each timestamp.
2. **Preprocessing**  
   - Normalize each channel to zero mean, unit variance.  
   - Resample to a consistent grid size (e.g., $128 \times 128$).  
   - Stack into a $(3, H, W)$ tensor per time step.

---

## 🛠️ Method

### 1. Convolutional ODE Function

We define the ODE function $f_\theta$ as a small convolutional network that concatenates a time channel:

### 2. Neural ODE Solver

We integrate $$dot{\mathbf{h}} = f_\theta(\mathbf{h}, t)$$ using an adaptive solver (dopri5) from torchdiffeq

### 3. Physics-Informed Loss

We combine standard data fidelity (MSE) with a divergence loss to enforce the physical law $\nabla \cdot \mathbf{B} = 0$:
#### 1.	Mean Squared Error (MSE)

$$\mathcal{L}_\text{MSE} = |\hat{\mathbf{h}} - \mathbf{h}|_2^2$$

#### 2.	Divergence Loss(Boundary Condition Loss)

Solar magnetic fields are divergence-free ((\nabla\cdot \mathbf{B} = 0)). We approximate divergence on the 2D grid:

$$\nabla\cdot \hat{\mathbf{h}}(t)= \frac{\partial \hat B_x}{\partial x}$$
	•	$\frac{\partial \hat B_y}{\partial y}$
	•	$\frac{\partial \hat B_z}{\partial z}\approx 0$
]
On a 2D surface we enforce:

$$\mathcal{L}\text{div} = \lambda\text{div},\bigl|\partial_x \hat B_x + \partial_y \hat B_y\bigr|_2^2$$

	
 #### 3. Total Loss

$$\mathcal{L} = \mathcal{L}\text{MSE} + \mathcal{L}\text{div}$$


⸻

## 🧪 Experiments

•	Boundary-condition weight $\lambda_\text{div} = 0.1$ (balance bias vs. noise)

•	Grid resolution: $64\times64$ vs. $128\times128$

•	Batch size: 8

•	Learning rate: $1\times10^{-3}$

### Ablation

<img width="510" alt="스크린샷 2025-05-14 오전 10 21 22" src="https://github.com/user-attachments/assets/7b72b4a9-60fb-41bc-9a5e-fc66f1453cb4" />

- The best performance occurred when the weight was 0.1, and increasing the weight led to an increase in loss.
- Having a boundary condition with a weight of 0.1 performed better than having no boundart condition.
- The data already contains the representation of the boundary condition, so larger weights introduced more noise.
- A weight of 0.1 helped reduce the noise, resulting in better performance compared to no boundary condition.

### Prediction Result

<img width="518" alt="스크린샷 2025-05-14 오전 10 20 30" src="https://github.com/user-attachments/assets/f3f4ecff-eaae-4c7d-be67-55993b642d9a" />

- The grid size increases, the loss tends to be higher.
- Even a 10-unit difference in grid size leads to an exponential increase in data complexity.

⸻

## 🎯 Results & Insights

•	Reduced overfitting compared to Conv-RNN baseline on limited data (~200 time steps).

•	Qualitative fidelity: magnetogram predictions align closely with ground truth (see visualizations/).

•	Physical consistency: divergence loss keeps $\nabla\cdot \mathbf{B}$ near zero (<1e-3 average).

•	Adaptive cost: solver calls (NFE) adjust per tolerance; overall training remains tractable.

⸻

## ⚠️ Limitations & Future Work

•	Data scarcity: only ~200 time steps → consider temporal augmentation.

•	Task variance: loss curves exhibit fluctuations—investigate solver tolerances and regularization.

•	Divergence trade-off: stronger $\lambda_\text{div}$ can introduce smoothing artifacts; explore adaptive weighting.

•	Inference speed: test fixed-step RK4 for faster deployment.

---

## 🙋‍♀️ Authors & Contributions

•	Yun-Young Lee (2019160102)

	•	Model design, divergence-based loss
 
	•	Hyperparameter tuning & overfitting experiments
 
•	Mi-Young Choi (2020160150)

	•	Data collection & preprocessing
 
	•	Visualization & solver analysis
