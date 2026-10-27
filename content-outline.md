# COMP 432 content outline (extracted from the slides)

Source: `/Users/jammie/Documents/comp432/docs/`. Page numbers are PDF pages.
Note: `handout.pdf` is **Lecture 4: From Linear Models to Neural Networks**.
Midterm (L3 p2): Mon 2 Nov, 17:45, H-435, lectures 1–5 + labs, closed book, 2 h —
"read, understand, and reason about code", "reason about concepts (with and without math)".

Notation (L1 p3): x scalar, **x** vector, **X** matrix, N examples, D features, K outputs, P parameters, θ parameters, y label, ŷ prediction.

## Lecture 1 — Introduction
- AI ⊃ ML ⊃ DL; traditional programming vs ML (p4–6)
- Inputs x ∈ R^D; regression (real outputs) vs classification (category) (p7, p13)
- Ice cream: (10,0),(20,20),(30,40); ŷ = wx+b, θ=(b,w); (10,1) fits one point, (−20,2) fits all (p8)
- Training / testing / inference (p9–12)
- Four components: dataset, model, objective, optimizer (p15)
- Dataset D, feature matrix X ∈ R^{N×D}; iris X ∈ R^{150×4} (p17–18)
- Train/test split (p19); supervised (X, Y ∈ R^{N×K}), unsupervised, RL (p20–22)
- Parameter space (p24–25)
- MSE table: ŷ=x+10 → 166.7, ŷ=2x−20 → 0 (p27); MSE = 1/N Σ (y_n − ŷ_n)^2 (p28)
- Objective = loss = cost = criterion; smaller is better (p29)
- Accuracy = 1/N Σ 1[ŷ_n = y_n]; worked 3-example table, Acc = 2/3 (p30–31)
- Accuracy is hard; Model 1 vs Model 2 (p32)
- CCE = −1/N Σ_n Σ_k y_{n,k} ln p_{n,k}; Model 1 ≈ 0.71, Model 2 ≈ 0.34 (p33–34)
- θ* = argmin J; brute force / guessing / iterative (p36–37)
- Naive probe-and-step loop: w=1,b=0 MSE 77.33 → probe ±0.05 → 3^P−1 evaluations (p38–41)

## Lecture 2 — Optimization
- CCE term by term: y=(0,1,0), p=(0.3,0.5,0.2), −ln p=(1.20,0.69,1.61), contribution 0.69 (p4)
- Capacity, P as proxy; linear/quadratic/degree-11 fits; under/overfitting; epochs; dataset size; Occam's razor (p8–13)
- θ* = argmin J; local minima, saddles, plateaus (p15)
- Derivative limit, critical points (p16); partial derivatives, gradient ∈ R^P (p17)
- Analytical vs numerical (p18)
- Taylor: J(θ+εd) ≈ J(θ) + ε‖∇J‖cos α; steepest ascent/descent (p20)
- GD: θ_{t+1} = θ_t − η∇J(θ_t); algorithm; init sensitivity; learning rate too small/right/large (p21–23)
- Linear least squares: ∂J/∂w = 2/N Σ (ŷ−y)x, ∂J/∂b = 2/N Σ (ŷ−y); training code w,b=0.5,1.0, lr 0.025 (p25–27)
- Autodiff, PyTorch nn.Linear / MSELoss / SGD loop (p28–29)
- Quadratic model J = ½ θᵀ diag(h1,h2) θ; narrow valleys zigzag (p31–32)
- Momentum v_{t+1} = ρ v_t − η∇J, θ_{t+1} = θ_t + v_{t+1}; ball physics (p33–36)
- Cost per step; per-sample losses J_D = 1/N Σ J_n (p37–38)
- SGD with minibatch B: g_B = 1/B Σ_{n∈B} ∇J_n (unbiased); epochs & shuffling; batch size trade-off (p39–41)
- Noise floor; LR scheduling η_t = η_0 γ^t (p42–43)
- Adaptive: s_{t+1} = β2 s_t + (1−β2) g⊙g; θ −= η/(ε+√s) ⊙ g (p46)
- Adam: v, s moving averages, update with v; defaults η=1e−3, β1=.9, β2=.999, ε=1e−8; no bias correction here (p47)
- Hyperparameters (η, B, epochs, θ0, ρ, γ, β1, β2, ε); validation set 60–70/10–20/20; early stopping; search strategies (p50–54)

## Lecture 3 — Linear models
- Iris running example; mini dataset x=(1,3,5), y=(0.2,1.1,1.7); X with sepal length (p7–10)
- ŷ = θᵀx̃, x̃ = (x,1), θ = (w,b); affine vs linear; linear in θ (p11–12)
- Dot product = ‖a‖‖b‖cos; examples 5, 0, −5 (p13); hyperplanes (p14)
- One step: θ=(0.3,0.1) → ŷ=(0.4,1.0,1.6), r=(0.2,−0.1,−0.1), J=0.02; ∇J = 2/N X̃ᵀr = (−0.4, 0); η=0.05 → θ1=(0.32,0.1), J≈0.017 (p16–18)
- Loss is quadratic; closed form θ* = (X̃ᵀX̃)⁻¹X̃ᵀy = (0.375, −0.125), J=0.005 (p19–20)
- Hessian H = 2/N X̃ᵀX̃; convexity (chord/tangent); PSD; condition number κ (p21–24)
- Logistic regression: σ(z) = 1/(1+e^{−z}), σ' = σ(1−σ); BCE; worked x̃=(4,1.3,1), θ=(1,1,−5), z=0.3, ŷ≈0.57, J≈0.85 (p27–29)
- Decision boundary θᵀx̃ = 0 (p30)
- ∂J_n/∂ŷ = (ŷ−y)/(ŷ(1−ŷ)) ≈ 2.35; ∇J_n = (ŷ−y)x̃ ≈ (2.30,0.75,0.57); ∇J = 1/N X̃ᵀ(ŷ−y) (p32–33)
- Error × input structure (p34); no closed form, Hessian 1/N X̃ᵀSX̃ PSD (p35); NumPy code (p36)
- Multiclass: W̃ = [[1,0,−3],[0,1,−1],[−1,0,2]], z=(1,0.3,−2), softmax ≈ (0.65,0.32,0.03), J ≈ 1.14 (p39–43)
- Decision boundaries argmax (p44); Ŷ = softmax(X̃W̃ᵀ), ∇_W̃ J = 1/N (Ŷ−Y)ᵀX̃; code (p46–47); MNIST (p49); summary table (p52–53)

## Lecture 4 (handout.pdf) — From linear models to neural networks
- Taylor with Hessian recap (p3)
- Feature transforms φ(x) = (x1², x2²), sin(2πx); polynomials (Taylor) and Fourier (p9–12)
- Linear basis function models table; still convex (p13)
- Polynomial features φ̃(x) = (x,…,x^M,1); design matrix Φ̃; θ* = (Φ̃ᵀΦ̃)⁻¹Φ̃ᵀy (p15–16)
- M = 1/3/9, overfitting, huge weights (p17–18)
- L2: J̃ = J + λ/2‖θ‖²; ∇J̃ = ∇J + λθ; θ* = (Φ̃ᵀΦ̃ + Nλ/2 I)⁻¹Φ̃ᵀy; λ effect (p19–20)
- NN: ŷ = wᵀφ_ν(x) + b; neuron; layer z = Wx + b, W ∈ R^{H×D}; MLP ŷ = W2 f(W1x+b1)+b2 (p23–27)
- Computational graph (p28); identity activation collapses (p29); sigmoid/tanh/ReLU + derivatives (p30)
- Non-convex counterexample θ=(1,0,1,0), θ'=(0,1,0,1), midpoint J=1/4 (p33–34)
- Universal approximation; sigmoid switches & pulses, h1=σ(500x), h2=σ(500x−100), ŷ=0.55h1−0.55h2 (p36–41)
- Deep nets, compositionality (p43–44)
- Scalar chain rule; backprop example (x,y,w1,w2)=(2,4,1,3): h1=2,h2=6,J=4; 4, 8, 12, 24 (p47–48)
- Vector chain rule, Jacobians, VJPs: ⊙f'(z), Wᵀ, (∇zJ)xᵀ, bias (p49–50)
- Try at home MLP: W1=[[1,1],[1,−1]], b1=(1,0), W2=(1,2), b2=−4, y=1, x=(1,2) (p51)
- Param count (D+1)H + (H+1)K; summary table (p55)

## Lecture 5 (handout5.pdf) — Deep networks in practice, and CNNs
- Deep scalar chain: each layer multiplies the gradient by w_k σ'(z_k); slide setup x = ½, biases −w/2 → σ' = ¼ (p8–9)
- Vanishing (factor ¼): ratios 0.25, 0.004, 4e−6, 4e−12 for L = 2, 5, 10, 20; exploding (factor 2): 2, 16, 512, 5e5 (p10)
- Real 20-layer net: sigmoid gradient shrinks per layer, ReLU level (p11–12); cliffs and plateaus (p13)
- Gradient clipping g ← g·min(1, C/‖g‖); C = 2.5, g = (3, 4) → (1.5, 2) (p14)
- Saturating vs non-saturating activations, leaky ReLU α ≈ 0.01–0.1 (p16–17)
- Shortcuts: sum node copies gradient; factor 1 + wσ'; ResNet, skip-to-output, DenseNet (p18–19)
- Feature scales and GD; standardization with training-set μ, σ (p22–24)
- Batch norm per unit over the batch, γ, β, running averages (PyTorch momentum = 1 − ρ); worked Z matrix (p25–29)
- Layer norm per sample; same at train and test; transformers/RNNs vs CNNs (p30)
- Regularization: augmentation, label smoothing (ε = 0.15 → 0.05/0.90/0.05), ensembles, dropout m⊙h/(1−p), train vs inference table (p32–38)
- CNNs: MLP on images too many weights (75M); local connections, weight sharing (p40–41)
- 1-D conv example x = (3,5,2,−1,0,4,−2,−3), kernel (−1,0,1), P = 1 → (5,−1,−6,−2,5,−2,−7,2); stride 2 → (5,−6,5,−7); D_out = ⌊(D+2P−F)/s⌋+1 (p42–45)
- Max pooling (3,5,0,4,0); channels; Conv1d shapes (p44, 46)
- Blocks Conv→BN→ReLU→MaxPool; residual block h' = ReLU(h + f(h)); ResNet-18 = 18 weight layers (p47–48)
- Head: Flatten → Linear → softmax; parameter counts F·C_in·C_out + C_out, 2-D F1·F2·C_in·C_out + C_out (p49–51)
- Hierarchical features; transfer learning (feature extraction vs fine-tuning, PyTorch code); multitask learning, negative transfer (p53–56)
- Summary table (p57)

## Midterm format (handout5.pdf p61–68)
- Mon 2 Nov, H-435, 17:45–19:45; HB pencil, eraser, photo ID; no smart devices
- 31 questions, 90 points + 2 bonus (4 points) on extra lectures; lectures 1–5 and labs 1–5
- Five options, exactly one correct; text, figure or code; no proofs, no code writing
- Closed book, no calculator; formula sheet: Lecture 3's summary table, GD, SGD, momentum
- Wrong answer −¼ of the points, blank 0; only the scantron is graded
- Three example questions (logistic GD directions; arrow augmentation; CCE summed over 100 examples = 230.26)

## Labs and assignments (comp432/Labs, comp432/assignment1–5)
- A1: Python references, filter/list comprehension, pickle, lambda closure for central differences; NumPy views vs copies (strides), vectorization, linear-system residual check, paired shuffle
- A2: matplotlib; sigmoid and leaky ReLU; finite-difference accuracy for ε = 1, 1e−5, 1e−15; normal pdf
- A3: meshgrid evaluation, Rosenbrock gradient and GD; linear regression by GD (with gradient tolerance) and direct solve; logistic gradient X̃ᵀ(ŷ − y)/N; scikit-learn LinearRegression(fit_intercept=False), LogisticRegression(C=np.inf)
- Tutorials (lab 3): linear/logistic/multiclass regression in NumPy, BCE/CCE, polar-coordinate feature transform, stable softmax (subtract row max), shift invariance, sklearn C = 1/(λN)
- A4: MNIST; MinMaxScaler fit on train; LogisticRegression(C=0.01) weights as digit templates; MLPClassifier hidden_layer_sizes=(), (32,), (128, 64) with tanh; coefs_ transposed; PyTorch Sequential, CrossEntropyLoss on logits, SGD momentum, test error via argmax
- Tutorials (lab 4): PyTorch tensors, autograd example x.grad = (4/3, 2, 8/3), nn.Module, DataLoader, BCEWithLogitsLoss, train/eval, no_grad, state_dict; MLP backprop in NumPy for batches; polynomial fitting (orders, more data, ridge, GD as implicit early stopping)
- A5: 1-D CNN pattern detector (Conv1d, MaxPool1d over full width, BCEWithLogitsLoss, weight_decay); 5 kernels get stuck, 7 work; MNIST CNN with 31,530 parameters
- Tutorials (lab 5): conv2d as loops, (N, C, H, W), Conv2d weight (C_out, C_in, F, F), edge kernels, max pooling; debugging strategy and five bugs (device, flatten, zero_grad, no_grad memory, drop_last with batch norm)
