import { MCQuestion, NumQuestion, Question, QuizKind } from "./types";

const r = String.raw;

function mc(kind: QuizKind, prompt: string, options: string[], answer: number, explain: string, code?: string): MCQuestion {
  return { type: "mc", kind, prompt, options, answer, explain, code };
}
function num(kind: QuizKind, prompt: string, answer: number, explain: string, code?: string, tol?: number): NumQuestion {
  return { type: "num", kind, prompt, answer, explain, code, tol };
}

/** Questions per section, keyed "lectureId/sectionId". */
export const quiz: Record<string, Question[]> = {
  // ---------------- Lecture 1 ----------------
  "l1/notation": [
    mc("formula", r`The feature matrix is $\mathbf X\in\mathbb R^{N\times D}$. What does row $n$ contain?`,
      ["All $N$ values of feature $n$", "All $D$ features of example $n$", "The label of example $n$", "The $n$-th parameter"], 1,
      r`One row per example, one column per feature. Row $n$ is example $n$'s feature vector $\mathbf x_n^\top$ with $D$ entries.`),
    mc("formula", "Predicting a house's price (in dollars) from its number of floors, bedrooms and age is…",
      ["classification", "regression", "clustering", "reinforcement learning"], 1,
      "The output is a real number, so it's regression. A category as output would make it classification."),
    num("calc", r`The iris feature matrix is $\mathbf X\in\mathbb R^{150\times 4}$. How many numbers does it hold?`, 600,
      r`$N\cdot D = 150\cdot 4 = 600$.`),
  ],
  "l1/pipeline": [
    mc("formula", "Why must the test set contain different examples from the training set?",
      ["Otherwise training is slower", "Otherwise the model's performance on new data is overestimated", "Because the test set has no labels", "It doesn't have to"], 1,
      "On examples it has already seen, a model can look good just by memorizing. Only unseen data measures generalization."),
    mc("formula", "You group customers by purchase behaviour, with no labels. This is…",
      ["supervised learning", "unsupervised learning", "reinforcement learning", "regression"], 1,
      "No labels → unsupervised. Grouping similar points is clustering."),
    num("code", "What does this print?", 1.8,
      r`{{X[:, 3]}} is column 3 (Python counts from 0), i.e. the 4th feature, petal width: $0.2 + 0.2 + 1.4 = 1.8$.`,
      r`import numpy as np
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [7.0, 3.2, 4.7, 1.4]])
print(round(X[:, 3].sum(), 2))`),
  ],
  "l1/model-mse": [
    num("calc", r`Data $x=(1,2,3)$, $y=(2,4,7)$, model $\hat y = 2x$. What is the MSE?`, 1 / 3,
      r`$\hat y = (2,4,6)$, errors $(0,0,1)$, squared $(0,0,1)$, average $\tfrac13 \approx 0.333$.`),
    mc("formula", "Why does MSE square the errors?",
      ["To make the number smaller", "So positive and negative errors don't cancel, and big errors count more", "Because the derivative of a square is zero", "To make it a probability"], 1,
      "Squaring makes every error positive and punishes large errors much more than small ones."),
    num("code", "What does this print?", 1.0,
      r`$\hat y = 2x = (0,2,4)$, errors $y-\hat y = (1,1,1)$, squared $(1,1,1)$, mean $1.0$.`,
      r`import numpy as np
x = np.array([0., 1., 2.])
y = np.array([1., 3., 5.])
w, b = 2, 0
print(((y - (w * x + b)) ** 2).mean())`),
  ],
  "l1/classification-metrics": [
    num("calc", r`The model predicts $\mathbf p = (0.1, 0.7, 0.2)$ for (cat, dog, bird); the label is dog. What is this example's cross-entropy?`, -Math.log(0.7),
      r`Only the correct class counts: $-\ln 0.7 \approx 0.357$.`),
    num("code", "What does this print? (Round to 3 decimals.)", 2 / 3,
      r`argmax per row is $(1, 0, 1)$; compared with $y=(1,1,1)$ that's 2 of 3 correct: $0.667$.`,
      r`import numpy as np
P = np.array([[0.2, 0.8],
              [0.6, 0.4],
              [0.3, 0.7]])
y = np.array([1, 1, 1])
print((P.argmax(axis=1) == y).mean())`),
    mc("formula", "Two models have the same accuracy. Which score can still tell which one is better?",
      ["Accuracy", "Cross-entropy", "Neither", "The number of parameters"], 1,
      "Accuracy only looks at the argmax (hard). Cross-entropy also rewards putting more probability on the correct class (soft)."),
    mc("formula", r`What is the CCE of an example where the model gives probability 1 to the correct class?`,
      ["0", "1", r`$\infty$`, "−1"], 0, r`$-\ln 1 = 0$: a perfect prediction costs nothing.`),
  ],
  "l1/naive-training": [
    num("calc", "Probe-and-step with P = 3 parameters, trying +δ, 0, −δ for each. How many neighbours are evaluated per step?", 26,
      r`$3^P - 1 = 27 - 1 = 26$ (all combinations except "no change").`),
    mc("formula", "Why doesn't probe-and-step scale to real models?",
      ["It needs labels", "The number of probes grows exponentially with the number of parameters", "It can't decrease the loss", "It only works for classification"], 1,
      r`$3^P - 1$ evaluations per step is hopeless for millions of parameters. The gradient gives the direction directly instead.`),
  ],

  // ---------------- Lecture 2 ----------------
  "l2/capacity": [
    mc("formula", "Training loss 0.01, test loss 0.90. Diagnosis?", ["underfitting", "good fit", "overfitting", "can't tell"], 2,
      "Very low training loss with a large train–test gap means the model memorized the training data: overfitting."),
    mc("formula", "Training loss 0.50, test loss 0.52 (and a much better model is possible). Diagnosis?", ["underfitting", "good fit", "overfitting", "can't tell"], 0,
      "Both losses are high and close together: the model is too simple to fit even the training data."),
    mc("formula", "A high-capacity model overfits. Which change is most likely to help?",
      ["Train for more epochs", "Collect more training data", "Use a more complex model", "Evaluate on the training set"], 1,
      "More data makes memorizing harder. Training longer or adding capacity usually makes overfitting worse."),
  ],
  "l2/gradient": [
    num("calc", r`$J(\theta) = 3\theta_1^2 + \theta_1\theta_2$. What is $\partial J/\partial\theta_1$ at $\theta = (1, 2)$?`, 8,
      r`Treat $\theta_2$ as a constant: $\partial J/\partial\theta_1 = 6\theta_1 + \theta_2 = 6 + 2 = 8$.`),
    mc("formula", "In which direction does the loss decrease fastest (for a small step)?",
      [r`$\nabla_\theta J$`, r`$-\nabla_\theta J$`, r`perpendicular to $\nabla_\theta J$`, "any direction"], 1,
      r`Change $\approx \epsilon\|\nabla J\|\cos\alpha$, most negative at $\alpha = 180°$, i.e. against the gradient.`),
    mc("formula", r`A model has $P$ parameters. What is the shape of $\nabla_\theta J$?`,
      ["a scalar", r`$\mathbb R^P$`, r`$\mathbb R^{P\times P}$`, r`$\mathbb R^N$`], 1,
      "One partial derivative per parameter: same shape as θ."),
  ],
  "l2/gd": [
    num("calc", r`$J(\theta)=\theta^2$, $\theta_0 = 3$, $\eta = 0.1$. What is $\theta_1$ after one GD step?`, 2.4,
      r`$J'(\theta)=2\theta = 6$. $\theta_1 = 3 - 0.1\cdot 6 = 2.4$.`),
    num("calc", r`For $J(\theta) = 2\theta^2$, above which learning rate does GD diverge?`, 0.5,
      r`$\theta \leftarrow \theta - \eta\cdot 4\theta = (1-4\eta)\theta$. It grows when $|1-4\eta| > 1$, i.e. $\eta > 0.5$.`),
    num("code", "What does this print?", 0.25,
      r`Each step: $\theta \leftarrow \theta - 0.25\cdot 2\theta = 0.5\theta$. Twice: $1 \to 0.5 \to 0.25$.`,
      r`theta = 1.0
for _ in range(2):
    theta -= 0.25 * 2 * theta
print(theta)`),
  ],
  "l2/least-squares": [
    num("calc", r`Data $x=(1,2)$, $y=(1,1)$, model $\hat y = wx+b$ with $w=1$, $b=0$, MSE loss. What is $\partial J/\partial w$?`, 2,
      r`$\hat y = (1,2)$, residuals $(0,1)$. $\frac2N\sum r_nx_n = \frac22(0\cdot1 + 1\cdot2) = 2$.`),
    num("calc", r`Same setup ($x=(1,2)$, $y=(1,1)$, $w=1$, $b=0$). What is $\partial J/\partial b$?`, 1,
      r`$\frac2N\sum r_n = \frac22(0 + 1) = 1$.`),
    num("code", "What does this print?", 0,
      r`residual $=(0,1)$, gradient $2\cdot\text{mean}((0,2)) = 2$, so $w = 1 - 0.5\cdot 2 = 0$.`,
      r`import numpy as np
x = np.array([1., 2.])
y = np.array([1., 1.])
w, b, lr = 1., 0., 0.5
residual = w * x + b - y
w = w - lr * 2 * (residual * x).mean()
print(w)`),
    mc("formula", "Why does this course's MSE gradient contain a factor 2?",
      ["Because there are two parameters", "Because the objective has no ½ in front, so the 2 from the square stays", "Because of the learning rate", "It's a typo in the slides"], 1,
      r`$\frac{d}{du}u^2 = 2u$. Some books define MSE with $\tfrac12$ to cancel it; this course (like PyTorch's MSELoss) doesn't.`),
  ],
  "l2/autodiff": [
    mc("formula", "In a PyTorch training loop, which line computes the gradients?",
      ["{{opt.zero_grad()}}", "{{loss = loss_fn(model(X), Y)}}", "{{loss.backward()}}", "{{opt.step()}}"], 2,
      "{{backward()}} runs backpropagation and stores each gradient in {{.grad}}. {{opt.step()}} then uses them to update."),
    mc("formula", "What goes wrong if you remove {{opt.zero_grad()}}?",
      ["Nothing", "Gradients from previous iterations add up, so the updates are wrong", "The loss is not computed", "The model is reset each epoch"], 1,
      "PyTorch accumulates gradients into {{.grad}}. Without resetting, each step uses the sum of all past gradients."),
    mc("formula", "Which line performs $\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta J$?",
      ["{{opt.step()}}", "{{loss.backward()}}", "{{nn.MSELoss()}}", "{{opt.zero_grad()}}"], 0,
      "The optimizer's {{step()}} applies the update rule using the stored gradients."),
  ],
  "l2/momentum": [
    num("calc", r`Momentum with $\rho = 0.9$, $\eta = 0.1$, $\mathbf v_0 = 0$, and the gradient is $1$ at both steps. What is $v_2$?`, -0.19,
      r`$v_1 = 0.9\cdot0 - 0.1\cdot1 = -0.1$. $v_2 = 0.9\cdot(-0.1) - 0.1\cdot1 = -0.19$.`),
    mc("formula", r`What does momentum reduce to when $\rho = 0$?`, ["Adam", "plain gradient descent", "SGD with B = 1", "it stops moving"], 1,
      r`$\mathbf v_{t+1} = -\eta\nabla J$, so $\theta_{t+1} = \theta_t - \eta\nabla J$: plain GD.`),
    num("code", "What does this print?", -0.05,
      r`Step 1: $v = -0.1$, $\theta = -0.1$. Step 2: $v = 0.5\cdot(-0.1) - 0.1\cdot(-1) = 0.05$, $\theta = -0.05$. The opposite gradients partly cancel.`,
      r`v, theta = 0.0, 0.0
for g in [1, -1]:
    v = 0.5 * v - 0.1 * g
    theta += v
print(round(theta, 3))`),
  ],
  "l2/sgd": [
    num("calc", "N = 1000 examples, batch size B = 50, 3 epochs. How many parameter updates?", 60,
      r`$N/B = 20$ updates per epoch, times 3 epochs $= 60$.`),
    mc("formula", "SGD with batch size B = N is…", ["online SGD", "gradient descent", "Adam", "momentum"], 1,
      "Every step uses the whole dataset: the exact gradient, i.e. plain GD."),
    mc("formula", "SGD's loss stops improving at a 'noise floor'. What's the standard fix?",
      ["Increase the learning rate", "Decrease the learning rate over training (a schedule)", "Use batch size 1", "Stop shuffling"], 1,
      "Smaller steps shrink the effect of gradient noise, so the floor goes down."),
    num("calc", r`Exponential schedule $\eta_t = \eta_0\gamma^t$ with $\eta_0 = 0.1$, $\gamma = 0.5$. What is $\eta_3$?`, 0.0125,
      r`$0.1\cdot 0.5^3 = 0.1\cdot0.125 = 0.0125$.`),
  ],
  "l2/adam": [
    mc("formula", r`In Adam, what does $\mathbf s$ (with decay rate $\beta_2$) track?`,
      ["the average gradient (direction)", "the average squared gradient (scale) per parameter", "the learning rate schedule", "the loss"], 1,
      r`$\mathbf s_{t+1} = \beta_2\mathbf s_t + (1-\beta_2)\mathbf g\odot\mathbf g$. Its square root sets each parameter's step size. $\mathbf v$ ($\beta_1$) tracks the direction.`),
    num("calc", r`First Adam step, $\mathbf s_0 = 0$, $\beta_2 = 0.999$, gradient $g = 2$. What is $s_1$?`, 0.004,
      r`$s_1 = 0.999\cdot0 + 0.001\cdot 2^2 = 0.004$.`),
    mc("code", "What does this print?",
      ["[0.0316 0.0316]", "[0.001 0.01]", "[0.0316 0.316]", "[0.1 1.]"], 0,
      r`$v = 0.1g$ and $\sqrt s = \sqrt{0.001}\,|g|$, so the step is $0.01\cdot0.1/0.0316 \approx 0.0316$ for both, whatever the gradient's size. Adam normalizes away the scale.`,
      r`import numpy as np
g = np.array([1., 10.])
v = (1 - 0.9) * g
s = (1 - 0.999) * g * g
print((0.01 / np.sqrt(s) * v).round(4))`),
    mc("formula", r`What does $\odot$ mean in $\mathbf g\odot\mathbf g$?`,
      ["dot product (a scalar)", "element-wise product (a vector)", "outer product (a matrix)", "matrix inverse"], 1,
      r`Multiply entry by entry: $(g_1^2, \dots, g_P^2)$. In NumPy: {{g * g}}.`),
  ],
  "l2/hyperparams": [
    mc("formula", "Why shouldn't you choose hyperparameters by their test-set performance?",
      ["The test set is too small", "Test information leaks into your choices, so the final estimate of generalization is too optimistic", "It would make training slower", "The test set has no labels"], 1,
      "The test set must stay unseen until the very end. Use a separate validation set for tuning."),
    mc("formula", "Early stopping decides when to stop based on…", ["the training loss", "the validation loss", "the test loss", "the learning rate"], 1,
      "Stop once validation performance starts to get worse."),
    mc("formula", "Which of these is NOT a hyperparameter?", ["learning rate η", "batch size B", "the weight w of a linear model", "number of epochs"], 2,
      "Weights are parameters: they're learned by training. Hyperparameters are set by hand."),
  ],

  // ---------------- Lecture 3 ----------------
  "l3/vector-form": [
    num("calc", r`$\mathbf x = (2, 1)$, $\mathbf w = (3, -1)$, $b = 0.5$. What is $\hat y = \theta^\top\tilde{\mathbf x}$?`, 5.5,
      r`$\tilde{\mathbf x} = (2,1,1)$, $\theta = (3,-1,0.5)$: $6 - 1 + 0.5 = 5.5$.`),
    num("calc", r`What is $\mathbf a^\top\mathbf b$ for $\mathbf a = (2,-1)$, $\mathbf b = (1,2)$?`, 0,
      r`$2\cdot1 + (-1)\cdot2 = 0$: the vectors are orthogonal.`),
    mc("formula", r`A linear regression model with $D$ features has how many parameters?`, [r`$D$`, r`$D+1$`, r`$2D$`, r`$N$`], 1,
      "D weights plus one bias."),
    mc("code", "What does this print?", ["(2,)", "(2, 2)", "(1, 2)", "(4,)"], 1,
      "Two examples (rows), each with one feature plus the appended 1 (two columns).",
      r`import numpy as np
X = np.column_stack([np.array([1., 2.]), np.ones(2)])
print(X.shape)`),
  ],
  "l3/one-step": [
    num("calc", r`$\tilde{\mathbf X}$ rows $(1,1)$ and $(2,1)$, $\mathbf y = (1,3)$, $\theta = (1, 0)$. What is the MSE?`, 0.5,
      r`$\hat{\mathbf y} = (1,2)$, $\mathbf r = (0,-1)$, $J = \tfrac12(0 + 1) = 0.5$.`),
    num("calc", r`Same setup. What is the first entry of $\nabla_\theta J = \tfrac2N\tilde{\mathbf X}^\top\mathbf r$ (the derivative w.r.t. $w$)?`, -2,
      r`$\tilde{\mathbf X}^\top\mathbf r = (1\cdot0 + 2\cdot(-1),\ 0 - 1) = (-2,-1)$, times $\tfrac22 = 1$: $(-2,-1)$.`),
    mc("formula", r`$\tilde{\mathbf X}\in\mathbb R^{N\times(D+1)}$, $\mathbf r\in\mathbb R^N$. What is the shape of {{X.T @ r}}?`,
      [r`$(N,)$`, r`$(D+1,)$`, r`$(N, D+1)$`, "it's an error"], 1, "$(D+1, N)$ times $(N,)$ gives $(D+1,)$: the same shape as θ, as a gradient must be."),
    mc("code", "What does this print?", ["[1.2 0.1]", "[0.8 -0.1]", "[1. 0.]", "[3. 1.]"], 0,
      r`$\mathbf r = (0,-1)$, gradient $\tfrac22\tilde{\mathbf X}^\top\mathbf r = (-2,-1)$, so $\theta = (1,0) - 0.1\cdot(-2,-1) = (1.2, 0.1)$.`,
      r`import numpy as np
X = np.array([[1., 1.], [2., 1.]])
y = np.array([1., 3.])
theta = np.array([1., 0.])
r = X @ theta - y
theta = theta - 0.1 * 2 / 2 * X.T @ r
print(theta)`),
  ],
  "l3/closed-form": [
    num("calc", r`$\tilde{\mathbf X}^\top\tilde{\mathbf X} = \begin{pmatrix}2&1\\1&1\end{pmatrix}$ and $\tilde{\mathbf X}^\top\mathbf y = (3, 2)$. What is $w$ in $\theta^\star = (w, b)$?`, 1,
      r`$\det = 2-1 = 1$, inverse $\begin{pmatrix}1&-1\\-1&2\end{pmatrix}$. $\theta^\star = (3-2,\ -3+4) = (1, 1)$, so $w = 1$.`),
    mc("formula", "A twice-differentiable loss is convex exactly when…",
      ["its gradient is zero", "its Hessian is positive semi-definite everywhere", "it has a closed-form minimum", "it is a polynomial"], 1,
      r`Convex ⇔ $\mathbf v^\top\mathbf H(\theta)\mathbf v \ge 0$ for all $\mathbf v$, $\theta$ (only upward curvature).`),
    mc("formula", r`The Hessian has condition number $\kappa = \lambda_{\max}/\lambda_{\min} = 10\,000$. What does this mean for GD?`,
      ["It converges in one step", "The loss bowl is very elongated, so GD converges slowly", "The loss is not convex", "The learning rate doesn't matter"], 1,
      "A large κ means a narrow valley: GD must use a small η (for the steep direction) and then crawls along the flat one."),
    mc("formula", "Does the linear-regression Hessian $\\tfrac2N\\tilde{\\mathbf X}^\\top\\tilde{\\mathbf X}$ depend on θ?",
      ["Yes", "No — the loss bowl has one fixed shape"], 1, "θ doesn't appear in it: the loss is quadratic, so its curvature is constant."),
  ],
  "l3/logistic": [
    num("calc", r`What is $\sigma(0)$?`, 0.5, r`$1/(1+e^0) = 1/2$.`),
    num("calc", r`BCE of one example with $y = 1$ and $\hat y = 0.8$?`, -Math.log(0.8),
      r`$-[1\cdot\ln0.8 + 0\cdot\ln0.2] = -\ln 0.8 \approx 0.223$.`),
    num("calc", r`What is the sigmoid's derivative $\sigma'(z)$ at $z = 0$?`, 0.25, r`$\sigma(0)(1-\sigma(0)) = 0.5\cdot0.5 = 0.25$.`),
    mc("formula", r`$\theta = (w_1, w_2, b) = (1, 1, -5)$. Where is the decision boundary?`,
      [r`$x_1 + x_2 = 5$`, r`$x_1 + x_2 = -5$`, r`$x_1 = x_2$`, r`$\sigma(x_1+x_2) = 5$`], 0,
      r`Boundary: $\theta^\top\tilde{\mathbf x} = 0 \iff x_1 + x_2 - 5 = 0$.`),
  ],
  "l3/logistic-train": [
    num("calc", r`One example, $\hat y = 0.9$, $y = 0$, $\tilde{\mathbf x} = (2, 1)$. What is the first entry of $\nabla_\theta J_n = (\hat y - y)\tilde{\mathbf x}$?`, 1.8,
      r`$(0.9 - 0)\cdot(2, 1) = (1.8, 0.9)$.`),
    mc("formula", "Why does logistic regression have no closed-form solution?",
      ["Its loss is not convex", "θ sits inside the non-linear sigmoid, so ∇J = 0 can't be solved for θ", "It has too many parameters", "BCE has no gradient"], 1,
      r`$\tilde{\mathbf X}^\top\sigma(\tilde{\mathbf X}\theta) = \tilde{\mathbf X}^\top\mathbf y$ can't be rearranged for θ. It is still convex, so GD finds the minimum.`),
    mc("formula", "Which gradient contains a factor 2?",
      ["linear regression with MSE", "logistic regression with BCE", "both", "neither"], 0,
      r`MSE: $\frac2N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)$. BCE: $\frac1N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)$.`),
    mc("code", "What does this print?", ["[-0.5  0. ]", "[0.5 0.5]", "[-1.  0.]", "[ 1. -0.5]"], 0,
      r`$\hat{\mathbf y} = \sigma(0) = (0.5, 0.5)$, error $(0.5, -0.5)$. $\tilde{\mathbf X}^\top$error $= (0.5 - 1.5,\ 0.5 - 0.5) = (-1, 0)$, divided by 2.`,
      r`import numpy as np
X = np.array([[1., 1.], [3., 1.]])
y = np.array([0., 1.])
y_hat = 1 / (1 + np.exp(-np.zeros(2)))
print(X.T @ (y_hat - y) / 2)`),
  ],
  "l3/multiclass": [
    num("calc", r`What is the first softmax probability for logits $\mathbf z = (0, 0, 0)$?`, 1 / 3, r`All $e^0 = 1$, so each is $1/3$.`),
    num("calc", r`Logits $\mathbf z = (\ln 3,\ 0)$. What is $\hat y_1 = \mathrm{softmax}(\mathbf z)_1$?`, 0.75,
      r`$e^{\ln3} = 3$, $e^0 = 1$: $3/(3+1) = 0.75$.`),
    num("calc", "Multiclass logistic regression on MNIST: D = 784 pixels, K = 10 classes. How many parameters?", 7850,
      r`$K(D+1) = 10\cdot785 = 7850$.`),
    mc("code", "What does this print?", ["(2,)", "(2, 1)", "(1, 2)", "(2, 2)"], 1,
      "Summing over axis 1 (the classes) with keepdims keeps a column: one sum per row, shape (2, 1).",
      r`import numpy as np
Z = np.array([[1., 1.], [0., 2.]])
E = np.exp(Z)
print(E.sum(axis=1, keepdims=True).shape)`),
    mc("formula", "You add the same constant c to all K logits. What happens to the softmax probabilities?",
      ["They all increase", "They don't change", "They become uniform", "They no longer sum to 1"], 1,
      r`$e^{z_k + c} = e^c e^{z_k}$; the $e^c$ cancels between numerator and denominator.`),
  ],
  "l3/summary": [
    mc("formula", "Which of the three linear models has a closed-form solution?",
      ["linear regression", "logistic regression", "multiclass logistic regression", "all three"], 0,
      "Only the MSE loss is quadratic in θ; the other two need gradient descent."),
    mc("formula", "Logistic regression is trained with which objective?", ["MSE", "BCE", "CCE", "accuracy"], 1, "Binary cross-entropy (CCE with K = 2)."),
    mc("formula", "What structure do all three models' gradients share?",
      ["(prediction − label) × input", "label × weight", "loss × learning rate", "they have nothing in common"], 0,
      r`Linear: $\tfrac2N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)$, logistic: $\tfrac1N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)$, multiclass: $\tfrac1N(\hat{\mathbf Y}-\mathbf Y)^\top\tilde{\mathbf X}$.`),
  ],

  // ---------------- Lecture 4 ----------------
  "l4/basis": [
    mc("formula", r`Why is linear regression on transformed features $\tilde\phi(\mathbf x)$ still convex?`,
      ["Because φ is linear", "Because the model is still linear in the parameters θ", "Because the data is separable", "It isn't convex"], 1,
      "The non-linearity is only in the inputs; as a function of θ the model (and MSE loss) has the same form as before."),
    mc("formula", "One class lies inside a circle around the origin, the other outside. Which transform makes them linearly separable?",
      [r`$\phi(\mathbf x) = (x_1^2, x_2^2)$`, r`$\phi(\mathbf x) = (x_1, x_2)$`, r`$\phi(\mathbf x) = (x_1 + x_2)$`, r`$\phi(\mathbf x) = (2x_1, 2x_2)$`], 0,
      r`The circle $x_1^2 + x_2^2 = r^2$ becomes the line $\phi_1 + \phi_2 = r^2$.`),
    num("calc", r`Polynomial features of order $M=3$: $\tilde\phi(x) = (x, x^2, x^3, 1)$. With $\theta = (1, 0, 0.5, -1)$, what is $\hat y$ at $x = 2$?`, 5,
      r`$\tilde\phi(2) = (2, 4, 8, 1)$: $2 + 0 + 4 - 1 = 5$.`),
  ],
  "l4/polyfit": [
    num("calc", r`N = 10 points, polynomial order M = 9. How many columns does the design matrix $\tilde{\boldsymbol\Phi}$ have?`, 10,
      r`$M + 1 = 10$: the powers $x$ to $x^9$ plus the column of ones.`),
    mc("formula", "With M = 9 and N = 10 training points, what is the training MSE of the closed-form fit (typically)?",
      ["about 0", "about 1", "very large", "undefined"], 0,
      "10 parameters, 10 points: the polynomial can pass through every point exactly. Zero training error, but it overfits."),
    mc("code", "What does this print?", ["[2. 4. 8. 1.]", "[1. 2. 4. 8.]", "[1. 1. 1. 1.]", "[2. 4. 6. 1.]"], 0,
      "Row 1 is x = 2: columns x, x², x³, then the 1.",
      r`import numpy as np
x = np.array([1., 2.])
M = 3
Phi = np.column_stack([x**m for m in range(1, M + 1)] + [np.ones_like(x)])
print(Phi[1])`),
  ],
  "l4/regularization": [
    num("calc", r`$\nabla_\theta J = (1, -2)$, $\lambda = 0.5$, $\theta = (2, 2)$. What is the first entry of $\nabla_\theta\tilde J = \nabla J + \lambda\theta$?`, 2,
      r`$(1, -2) + 0.5\cdot(2,2) = (2, -1)$.`),
    num("calc", r`Weight decay: $\theta \leftarrow (1-\eta\lambda)\theta - \eta\nabla J$. With $\eta = 0.1$, $\lambda = 0.5$, by what factor is θ shrunk each step?`, 0.95,
      r`$1 - 0.1\cdot0.5 = 0.95$.`),
    mc("formula", "What happens when λ is much too large?",
      ["overfitting", "underfitting: the weights are pushed too close to 0", "nothing", "GD diverges"], 1,
      "The penalty dominates the data term, so the model becomes too simple."),
    num("calc", r`In $\theta^\star = (\tilde{\boldsymbol\Phi}^\top\tilde{\boldsymbol\Phi} + \tfrac{N\lambda}{2}\mathbf I)^{-1}\tilde{\boldsymbol\Phi}^\top\mathbf y$ with $N = 4$, $\lambda = 0.5$: what is added to each diagonal entry?`, 1,
      r`$N\lambda/2 = 4\cdot0.5/2 = 1$.`),
  ],
  "l4/mlp": [
    num("calc", "A two-layer MLP with D = 4 inputs, H = 5 hidden neurons, K = 3 outputs. How many parameters?", 43,
      r`$(D+1)H + (H+1)K = 5\cdot5 + 6\cdot3 = 25 + 18 = 43$.`),
    mc("formula", r`What is the shape of $\mathbf W^{(1)}$ (inputs $D$ → $H$ hidden neurons)?`, [r`$D\times H$`, r`$H\times D$`, r`$H\times H$`, r`$(D+1)\times H$`], 1,
      r`One row per neuron: $\mathbf z = \mathbf W\mathbf x$ maps $\mathbb R^D \to \mathbb R^H$, so $\mathbf W\in\mathbb R^{H\times D}$.`),
    mc("formula", "What if every activation is the identity f(z) = z?",
      ["The network becomes more expressive", "The whole MLP collapses to a single linear (affine) model", "Training becomes impossible", "Nothing changes"], 1,
      r`$\mathbf W^{(2)}(\mathbf W^{(1)}\mathbf x + \mathbf b^{(1)}) + \mathbf b^{(2)} = \mathbf W\mathbf x + \mathbf b$ with $\mathbf W = \mathbf W^{(2)}\mathbf W^{(1)}$.`),
    mc("code", "What does this print?", ["[0. 0.]", "[0. 2.]", "[0. -1.]", "[1. 1.]"], 0,
      r`$\mathbf W\mathbf x = (1-1,\ 2+0) = (0, 2)$, plus $\mathbf b$: $(0, -1)$. ReLU: $(0, 0)$.`,
      r`import numpy as np
W1 = np.array([[1., -1.], [2., 0.]])
b1 = np.array([0., -3.])
x = np.array([1., 1.])
print(np.maximum(0, W1 @ x + b1))`),
  ],
  "l4/activations": [
    num("calc", r`What is $\mathrm{ReLU}'(-2)$?`, 0, r`$\mathbb 1[z > 0] = 0$ for $z = -2$.`),
    num("calc", r`What is $\tanh'(0)$?`, 1, r`$1 - \tanh^2(0) = 1 - 0 = 1$.`),
    mc("formula", "Why can sigmoid/tanh activations slow down learning in deep networks?",
      ["They are expensive to compute", "They saturate: for large |z| the derivative is ≈ 0, so gradients vanish", "Their output is not a probability", "They are linear"], 1,
      "Backprop multiplies by f′(z) at every layer; tiny factors shrink the gradient."),
  ],
  "l4/nonconvex-uat": [
    mc("formula", "What does the universal approximation theorem NOT guarantee?",
      ["that a one-hidden-layer MLP can represent any continuous function closely", "that gradient descent will find those parameters", "that more neurons can help", "that the activation must be non-polynomial"], 1,
      "Representation ≠ learnability. The loss is non-convex, so training may fail."),
    num("calc", "Approximating a function with sigmoid pulses on 15 intervals. How many hidden neurons?", 30, "Two neurons (two switches) per pulse: 15 · 2 = 30."),
    num("calc", r`Tiny net $\hat y = w_1\mathrm{ReLU}(\nu_1) + w_2\mathrm{ReLU}(\nu_2)$, data $(x,y) = (1,1)$, $J = (\hat y - y)^2$. What is $J$ at $\theta = (\tfrac12,\tfrac12,\tfrac12,\tfrac12)$?`, 0.25,
      r`$\hat y = \tfrac12\cdot\tfrac12 + \tfrac12\cdot\tfrac12 = \tfrac12$, $J = (\tfrac12 - 1)^2 = \tfrac14$. That's above the chord (0), proving non-convexity.`),
  ],
  "l4/backprop-scalar": [
    num("calc", r`$h_1 = w_1x$, $h_2 = w_2h_1$, $J = (h_2 - y)^2$ with $x = 1$, $y = 0$, $w_1 = 2$, $w_2 = 3$. What is $\partial J/\partial w_2$?`, 24,
      r`Forward: $h_1 = 2$, $h_2 = 6$. $dJ/dh_2 = 2(6-0) = 12$. $\partial J/\partial w_2 = 12\cdot h_1 = 24$.`),
    num("calc", r`Same setup. What is $\partial J/\partial w_1$?`, 36,
      r`$dJ/dh_1 = 12\cdot w_2 = 36$, then $\partial J/\partial w_1 = 36\cdot x = 36$.`),
    mc("formula", "What makes backpropagation efficient?",
      ["It uses finite differences", "Each intermediate derivative dJ/dh is computed once and reused for everything before it", "It skips the forward pass", "It only computes the largest gradients"], 1,
      "The backward pass passes dJ/dh from layer to layer instead of recomputing the whole chain for every weight."),
  ],
  "l4/backprop-vector": [
    mc("formula", r`For $\mathbf z = \mathbf W\mathbf x + \mathbf b$, what is $\nabla_{\mathbf x}J$?`,
      [r`$\mathbf W\nabla_{\mathbf z}J$`, r`$\mathbf W^\top\nabla_{\mathbf z}J$`, r`$(\nabla_{\mathbf z}J)\mathbf x^\top$`, r`$\nabla_{\mathbf z}J$`], 1,
      r`Passing the error back through a layer multiplies by $\mathbf W^\top$. $(\nabla_{\mathbf z}J)\mathbf x^\top$ is the gradient for $\mathbf W$.`),
    num("calc", r`$\nabla_{\mathbf z}J = (1, -2)$ and $\mathbf x = (3, 1)$. What is entry $[2,1]$ (row 2, column 1) of $\nabla_{\mathbf W}J = (\nabla_{\mathbf z}J)\mathbf x^\top$?`, -6,
      r`Outer product: row 2 is $-2\cdot(3, 1) = (-6, -2)$, so entry $[2,1] = -6$.`),
    mc("code", "What does this print?", ["[1. 0. 1.]", "[2. -1. 0.5]", "[1. 1. 1.]", "[0. 1. 0.]"], 0,
      "This is the backward step through ReLU: multiply by 1[z > 0] = (1, 0, 1).",
      r`import numpy as np
z = np.array([2., -1., 0.5])
dh = np.array([1., 1., 1.])
print((z > 0) * dh)`),
    num("calc", r`Sigmoid output with BCE: $dJ/dz = \hat y - y$. What is it for $\hat y = 0.3$, $y = 1$?`, -0.7, r`$0.3 - 1 = -0.7$.`),
  ],

  // ---------------- Lecture 5 ----------------
  "l5/vanishing": [
    num("calc", r`A scalar chain where every layer multiplies the gradient by $\tfrac14$. With $L = 3$ layers, what is $\frac{\partial J/\partial w_1}{\partial J/\partial w_L}$?`, 1 / 16,
      r`$(\tfrac14)^{L-1} = (\tfrac14)^2 = \tfrac1{16} = 0.0625$.`),
    mc("formula", r`Why does the sigmoid chain on the slides lose a factor of 4 per layer when $w = 1$?`,
      [r`Because $\sigma(0) = \tfrac14$`, r`Because $\sigma'(z) \le \tfrac14$, and here $\sigma'(0) = \tfrac14$ exactly`, "Because the loss is squared", "Because the bias is −w/2", "Because of the learning rate"], 1,
      r`Each layer multiplies by $w\,\sigma'(z_k)$. The sigmoid's derivative is at most $\tfrac14$ (at $z = 0$), so with $w = 1$ the factor is $\tfrac14$.`),
    num("calc", r`Gradient clipping with $C = 1$ and $\mathbf g = (6, 8)$. What is the first entry of the clipped gradient?`, 0.6,
      r`$\|\mathbf g\| = 10$, factor $\min(1, 1/10) = 0.1$, so $\mathbf g = (0.6, 0.8)$.`),
    mc("code", "What does this print?", ["[3. 4.]", "[0.6 0.8]", "[1.5 2. ]", "[0.3 0.4]"], 0,
      r`$\|\mathbf g\| = 5 \le C = 10$, so $\min(1, C/\|\mathbf g\|) = 1$: the gradient is unchanged. Clipping only acts on gradients that are too long.`,
      r`import numpy as np
g = np.array([3., 4.])
C = 10.0
print(g * min(1.0, C / np.linalg.norm(g)))`),
  ],
  "l5/remedies": [
    mc("formula", "Why does ReLU help against vanishing gradients compared to sigmoid?",
      ["It is cheaper to compute", "Its derivative is exactly 1 for active units, so it doesn't shrink the gradient", "It outputs probabilities", "Its derivative is never 0", "It makes the loss convex"], 1,
      "Sigmoid's derivative is at most ¼ and near 0 when saturated. ReLU passes the gradient unchanged for z > 0 (but gives 0 for z < 0)."),
    num("calc", r`In a one-unit chain with shortcuts, each layer's factor becomes $1 + w\sigma'(z)$. What is it for $w = 0.4$, $\sigma'(z) = 0.25$?`, 1.1,
      r`$1 + 0.4\cdot0.25 = 1.1$. Without the shortcut it would be $0.1$.`),
    mc("formula", r`A residual block computes $\mathbf h' = \mathbf h + f(\mathbf h)$. What does it compute if $f$ outputs 0?`,
      ["nothing: the gradient is zero", "the identity: h′ = h", "a random function", "the negative of h", "a constant"], 1,
      "That's why adding residual blocks never hurts representational power: the block can always fall back to passing h through unchanged."),
  ],
  "l5/normalization": [
    num("calc", r`Standardize the feature values $(1, 3, 5)$ (variance divides by N). What is $\hat x$ for the value 5? (2 decimals)`, 3 / Math.sqrt(8),
      r`$\mu = 3$, $\sigma^2 = \tfrac13(4 + 0 + 4) = \tfrac83$, $\sigma \approx 1.633$. $\hat x = (5-3)/1.633 \approx 1.22$.`),
    mc("formula", "Batch norm computes one mean per…",
      ["sample, over its units", "unit (feature), over the samples in the minibatch", "layer, over everything", "parameter", "epoch"], 1,
      "The normalization axis is the batch axis. Layer norm is the other way round: one mean per sample, over its units."),
    mc("formula", "At inference, where does batch norm get μ and σ from?",
      ["from the test batch", "running averages collected during training", "it recomputes them from the training set", "they are learned parameters γ and β", "it skips normalization"], 1,
      "There may be no batch at inference (a single image), so BN uses the running averages it kept during training."),
    mc("code", "What does this print?", ["(3,)", "(2,)", "(2, 3)", "(1, 3)"], 0,
      "The mean over axis 0 (the samples) gives one number per unit: 3 units → shape (3,). That's batch norm's μ_B.",
      r`import numpy as np
Z = np.array([[1., 2., 3.],
              [3., 4., 5.]])     # (B=2 samples, 3 units)
print(Z.mean(axis=0).shape)`),
  ],
  "l5/regularization": [
    num("calc", r`Label smoothing with $K = 4$ classes and $\epsilon = 0.2$. What is the smoothed target for the correct class?`, 0.85,
      r`$(1 - 0.2)\cdot 1 + 0.2/4 = 0.8 + 0.05 = 0.85$. Each wrong class gets $0.05$.`),
    mc("code", "What does this print?", ["[0.  0.8 0.  0.4]", "[0.  0.4 0.  0.2]", "[0.6 0.4 0.2 0.2]", "[0.  0.2 0.  0.1]"], 0,
      "The mask keeps units 2 and 4 (values 0.4 and 0.2) and zeros the others. Dividing by 1 − p = 0.5 doubles the survivors: (0, 0.8, 0, 0.4).",
      r`import numpy as np
h = np.array([0.6, 0.4, 0.3, 0.2])
m = np.array([0, 1, 0, 1])
p = 0.5
print(m * h / (1 - p))`),
    mc("formula", "Which of these is still active at inference / validation time?",
      ["dropout masks", "random data augmentation", "standardization with the training-set μ, σ", "minibatch statistics in batch norm", "the L2 penalty"], 2,
      "Standardization uses the same training-set μ and σ everywhere. Dropout and augmentation switch off, BN switches to running averages, and penalties only affect the training loss."),
  ],
  "l5/convolution": [
    num("calc", r`Output length of a 1-D convolution with $D = 10$, kernel $F = 3$, padding $P = 0$, stride $s = 1$?`, 8,
      r`$\lfloor(10 + 0 - 3)/1\rfloor + 1 = 8$.`),
    num("calc", r`Same input, $D = 10$, $F = 3$, but $P = 1$, stride $s = 2$. Output length?`, 5,
      r`$\lfloor(10 + 2 - 3)/2\rfloor + 1 = \lfloor 4.5\rfloor + 1 = 5$.`),
    num("calc", r`Kernel $(1, 0, -1)$ applied to the window $(4, 7, 1)$. What is the output?`, 3,
      r`$1\cdot4 + 0\cdot7 + (-1)\cdot1 = 3$.`),
    mc("code", "What does this print?", ["[3. 2.]", "[2. 4.]", "[3. 4.]", "[1. 2.]"], 0,
      "Max pooling with size 2, stride 2: windows (1, 3) and (2, 0) → maxima 3 and 2.",
      r`import numpy as np
x = np.array([1., 3., 2., 0.])
print(x.reshape(-1, 2).max(axis=1))`),
  ],
  "l5/cnn": [
    num("calc", "Parameters of a 1-D convolution with F = 3, C_in = 2, C_out = 4 (with biases)?", 28,
      r`$F\cdot C_{\text{in}}\cdot C_{\text{out}} + C_{\text{out}} = 3\cdot2\cdot4 + 4 = 28$.`),
    num("calc", "Parameters of a 2-D convolution with a 3×3 kernel, C_in = 1, C_out = 8?", 80,
      r`$3\cdot3\cdot1\cdot8 + 8 = 72 + 8 = 80$.`),
    mc("formula", "How does a convolutional layer's parameter count depend on the input length D?",
      ["it grows linearly with D", "it does not depend on D at all", "it grows with D²", "it shrinks as D grows", "only through the padding"], 1,
      "Weight sharing: the same kernel slides over any length. A dense layer's count would grow with the input size."),
  ],
  // ---------------- Labs ----------------
  "labs/python": [
    mc("code", "What does this print?", ["[2]", "[1, 2]", "an IndexError", "[2, None]"], 0,
      "The starts are 0, 2, 4. The last slice {{perm[4:6]}} runs past the end and is cut short, so the last minibatch has one element.",
      r`perm = [3, 0, 4, 1, 2]
batches = [perm[s:s + 2] for s in range(0, 5, 2)]
print(batches[-1])`),
    num("code", "What does this print?", 7,
      "{{g(1) = f(f(1)) = f(3) = 2·3 + 1 = 7}}.",
      r`f = lambda x: 2 * x + 1
g = lambda x: f(f(x))
print(g(1))`),
    mc("code", "What does this print?", ["10", "5", "an error: Double has no __call__", "None"], 0,
      "{{Double}} inherits {{__call__}} from {{Module}}, so {{Double()(5)}} runs {{forward(5)}}. This is how {{model(x)}} works in PyTorch.",
      r`class Module:
    def __call__(self, x):
        return self.forward(x)

class Double(Module):
    def forward(self, x):
        return 2 * x

print(Double()(5))`),
    mc("code", "What does this print?", ["0.67", "0.66", "0.6667", "2/3"], 0,
      "{{:.2f}} rounds to 2 decimals: 0.666… → 0.67.",
      r`print(f"{2/3:.2f}")`),
    num("code", "What does this print?", 32,
      "{{zip}} pairs (1, 4), (2, 5), (3, 6): 4 + 10 + 18 = 32, a dot product.",
      r`print(sum(a * b for a, b in zip([1, 2, 3], [4, 5, 6])))`),
  ],
  "labs/numpy": [
    mc("code", "What does this print?", ["100.0", "0.0", "1.0", "an error"], 0,
      "{{x[:, :2]}} is a view: it shares memory with x, so writing into y changes x too.",
      r`import numpy as np
x = np.arange(6.).reshape(2, 3)
y = x[:, :2]
y[0, 0] = 100
print(x[0, 0])`),
    mc("code", "What does this print?", ["(3,)", "(4,)", "(4, 3)", "(1, 3)"], 0,
      "axis=0 makes the first dimension (the 4 rows) disappear: one mean per column.",
      r`import numpy as np
X = np.ones((4, 3))
print(X.mean(axis=0).shape)`),
    mc("formula", "To shuffle a dataset, which code keeps every input with its label?",
      ["{{perm = rng.permutation(N); X, Y = X[perm], Y[perm]}}", "{{X, Y = X[rng.permutation(N)], Y[rng.permutation(N)]}}", "{{rng.shuffle(X); rng.shuffle(Y)}}", "{{X, Y = X.T, Y.T}}"], 0,
      "One permutation applied to both arrays. Two separate shuffles break the pairing between inputs and labels."),
    mc("code", "What does this print?", ["(4, 4)", "(4, 1)", "(4,)", "an error"], 0,
      "Right-aligned, (4, 1) and (4,) become (4, 1) and (1, 4): both axes stretch. The silent broadcasting trap.",
      r`import numpy as np
print((np.ones((4, 1)) - np.ones(4)).shape)`),
    mc("code", "What does this print?", ["[3 5 7]", "[ 3 12]", "15", "[0 3]"], 0,
      "The rows are [0 1 2] and [3 4 5]. {{axis=0}} adds them column by column.",
      r`import numpy as np
X = np.arange(6).reshape(2, 3)
print(X.sum(axis=0))`),
    num("code", "What does this print?", 0.75,
      "Predicted classes [0, 1, 1, 0] vs labels [0, 1, 0, 0]: 3 of 4 agree. The mean of a True/False array is the fraction of True.",
      r`import numpy as np
y_hat = np.array([0.2, 0.7, 0.9, 0.4])
y = np.array([0, 1, 0, 0])
print(((y_hat > 0.5).astype(float) == y).mean())`),
    mc("code", "What does this print?", ["[0 3 2]", "3", "[-1  3  2]", "0"], 0,
      "{{np.maximum}} compares entry by entry with 0 (this is the ReLU). {{np.max}} would return the single largest value, 3.",
      r`import numpy as np
print(np.maximum(np.array([-1, 3, 2]), 0))`),
  ],
  "labs/finite-differences": [
    num("calc", r`Central difference for $f(x) = x^2$ at $x = 1$ with $\epsilon = 0.5$: $\frac{f(1.5) - f(0.5)}{2\cdot0.5}$?`, 2,
      r`$(2.25 - 0.25)/1 = 2$, exactly $f'(1) = 2$: for a quadratic the central difference has no error at all.`),
    mc("formula", r`Why is the central-difference estimate with $\epsilon = 10^{-15}$ inaccurate?`,
      ["ε is too large, so the curve's shape matters", "f(x+ε) and f(x−ε) are almost equal, and floating-point rounding wipes out most digits of their difference", "Central differences only work for linear functions", "Python can't represent 10⁻¹⁵", "It isn't inaccurate"], 1,
      "Subtracting two nearly equal floating-point numbers (cancellation) leaves few correct digits, and dividing by the tiny 2ε amplifies the error."),
    num("calc", r`For $f(x) = x^3$ the central difference equals $3x^2 + \epsilon^2$ exactly. What does it give at $x = 1$ with $\epsilon = 0.1$?`, 3.01,
      r`$3\cdot1^2 + 0.1^2 = 3 + 0.01 = 3.01$. The true derivative is 3.`),
    mc("formula", "Why is the central difference usually more accurate than the forward difference (f(x+ε) − f(x))/ε for the same ε?",
      ["It uses more function evaluations per step", "Its error shrinks like ε², the forward difference's only like ε", "It avoids floating-point rounding", "It is exact for every polynomial", "It isn't more accurate"], 1,
      "In the subtraction f(x+ε) − f(x−ε) the even-power terms cancel, so the leading error is of order ε²."),
  ],
  "labs/from-scratch": [
    mc("formula", "scikit-learn's {{LogisticRegression()}} with default settings, fit on the same data as our plain gradient descent on BCE, gives different weights. Why?",
      ["It uses MSE", "By default it adds an L2 penalty (C = 1); C = np.inf switches it off", "It doesn't use a bias", "Our gradient lacks a factor 2", "It trains on the test set"], 1,
      "C is the inverse regularization strength. The default C = 1 regularizes; with C = np.inf both minimize the same objective and agree."),
    mc("formula", "In scikit-learn, what happens when you decrease C from 1 to 0.01?",
      ["Regularization gets 100 times weaker", "Regularization gets 100 times stronger, so the weights shrink", "The learning rate decreases", "The number of iterations decreases", "Nothing"], 1,
      r`$C$ multiplies the loss, not the penalty: $\lambda = 1/(CN)$, so a smaller C means a stronger penalty.`),
    mc("code", "What does this print?", ["[-0.25  0.  ]", "[0.25 0.  ]", "[-0.5  0. ]", "[0. 0.]"], 0,
      "All predictions are 0.5, errors [0.5, −0.5]. Entry 0: (1·0.5 + 2·(−0.5))/2 = −0.25. Entry 1: (0.5 − 0.5)/2 = 0.",
      r`import numpy as np
X_tilde = np.array([[1.0, 1], [2.0, 1]])
y = np.array([0.0, 1.0])
theta = np.zeros(2)
y_hat = 1 / (1 + np.exp(-(X_tilde @ theta)))
print(X_tilde.T @ (y_hat - y) / y.size)`),
    mc("formula", "Multiclass logistic regression with N = 100, D = 2 and K = 4. What is the shape of {{(Y_hat - Y).T @ X_tilde / len(Y)}}?",
      ["(4, 3)", "(3, 4)", "(100, 4)", "(4, 2)", "(4,)"], 0,
      "(K, N) @ (N, D+1) → (K, D+1) = (4, 3): one row of parameters per class, the shape of W."),
    num("code", "What does this print?", 0.5,
      "Subtracting the row max gives [0, 0], so both entries are e⁰/2 = 0.5. Without the subtraction, e¹⁰⁰⁰ overflows to inf and the result is NaN.",
      r`import numpy as np
def softmax(Z):
    e_Z = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e_Z / e_Z.sum(axis=1, keepdims=True)
print(softmax(np.array([[1000.0, 1000.0]]))[0, 0])`),
    mc("formula", "What does {{LinearRegression().fit(X, y).score(X, y)}} return?",
      ["the MSE", "the R² coefficient of determination", "the accuracy", "the number of iterations", "the closed-form θ"], 1,
      "For regressors scikit-learn's score is R²; for classifiers it is the accuracy."),
  ],
  "labs/pytorch-basics": [
    mc("code", "What does this print?", ["tensor(8.)", "tensor(15.)", "tensor(6.)", "None"], 0,
      r`$\frac{dy}{dx} = 2x + 2 = 8$ at $x = 3$.`,
      r`x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad)`),
    mc("code", "What does this print?", ["tensor(8.)", "tensor(4.)", "tensor(16.)", "an error"], 0,
      "Each backward() adds its gradient 2x = 4 to .grad: 4 + 4 = 8. That's why training loops call zero_grad().",
      r`x = torch.tensor(2.0, requires_grad=True)
(x ** 2).backward()
(x ** 2).backward()
print(x.grad)`),
    mc("code", "What does this print?", ["torch.float64", "torch.float32", "torch.int64", "numpy.float64"], 0,
      "{{from_numpy}} keeps NumPy's float64. The labs add {{.float()}} to match the model's float32 parameters.",
      r`import numpy as np
print(torch.from_numpy(np.array([1.0, 2.0])).dtype)`),
    num("calc", "How many parameters does {{nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))}} have?", 17,
      "(2·4 + 4) + (4·1 + 1) = 12 + 5 = 17. The Tanh has none."),
    mc("code", "What does this print?", ["torch.Size([1, 1, 10, 12])", "torch.Size([1, 10, 12, 1])", "torch.Size([10, 12, 1, 1])", "torch.Size([1, 12, 10, 1])"], 0,
      "permute(2, 0, 1) moves the last axis first: (1, 10, 12); unsqueeze(0) adds the batch axis: (1, 1, 10, 12).",
      r`print(torch.zeros(10, 12, 1).permute(2, 0, 1).unsqueeze(0).shape)`),
  ],
  "labs/pytorch": [
    mc("formula", "{{torch.nn.Linear(784, 10)}}: what is the shape of its {{weight}}?",
      ["(10, 784)", "(784, 10)", "(785, 10)", "(10, 785)"], 0,
      "One row per output unit: (out, in). The bias is a separate (10,) vector. scikit-learn's coefs_[0] would be the transpose, (784, 10)."),
    mc("formula", "A model ends with {{torch.nn.Softmax(dim=1)}} and is trained with {{torch.nn.CrossEntropyLoss()}}. What's the problem?",
      ["None", "CrossEntropyLoss applies softmax itself, so softmax is applied twice and training is hampered", "CrossEntropyLoss needs one-hot float targets", "Softmax should use dim=0", "The model has no bias"], 1,
      "CrossEntropyLoss expects raw logits. Remove the Softmax layer (apply torch.softmax only when you want probabilities, e.g. at evaluation)."),
    num("calc", "How many parameters does a 784-32-10 MLP have (with biases)?", 25450,
      r`$785\cdot32 + 33\cdot10 = 25120 + 330 = 25450$.`),
    mc("formula", "Which scikit-learn model is the same as multiclass logistic regression?",
      ["{{MLPClassifier(hidden_layer_sizes=(10,))}}", "{{MLPClassifier(hidden_layer_sizes=())}}", "{{MLPClassifier(hidden_layer_sizes=(784,))}}", "{{LinearRegression()}}"], 1,
      "No hidden layers: the 784 inputs connect directly to the 10 softmax outputs."),
    mc("formula", "A training loop calls {{loss.backward()}} but never {{optimizer.step()}}. What happens?",
      ["Training works but is slower", "The gradients are computed but the parameters never change", "An error is raised", "The loss becomes NaN", "Only the biases are updated"], 1,
      "step() is the line that applies p ← p − lr·p.grad. Without it nothing is learned, and nothing crashes."),
    num("calc", "1,000 training examples, batch_size = 64, 10 epochs, and every minibatch (including the smaller last one) gets a step. How many times is {{optimizer.step()}} called?", 160,
      "range(0, 1000, 64) has 16 starts (the last minibatch has 1000 − 15·64 = 40 examples), so 16 steps per epoch × 10 = 160."),
    mc("formula", "In the MLP tutorial's NumPy backward pass with H = 32 hidden units and D = 2 inputs, what is the shape of {{W1_grad = Z1_grad.T @ X}}?",
      ["(32, 2)", "(2, 32)", "(N, 32)", "(32,)", "(N, 2)"], 0,
      "Z1_grad is (N, H), so Z1_grad.T @ X is (H, N) @ (N, D) = (H, D) = (32, 2), the shape of W1."),
    mc("formula", "The model ends with {{nn.Sigmoid()}} and is trained with {{nn.BCEWithLogitsLoss()}}. What's wrong?",
      ["Nothing", "The sigmoid is applied twice", "BCEWithLogitsLoss needs integer labels", "The labels must be one-hot", "Sigmoid can't be the last layer"], 1,
      "BCEWithLogitsLoss expects logits and applies the sigmoid itself. Remove the Sigmoid layer, or switch to BCELoss."),
  ],
  "labs/conv-code": [
    mc("formula", "What is the shape of {{torch.nn.Conv2d(3, 8, 5).weight}}?",
      ["(8, 3, 5, 5)", "(3, 8, 5, 5)", "(5, 5, 3, 8)", "(8, 5, 5)"], 0,
      "(C_out, C_in, F, F): one F × F kernel per pair of output and input channel."),
    num("calc", "Without padding, stride 1, a 3 × 3 kernel on a 10 × 12 image gives an output of size 8 × ?", 10,
      r`$(H - F + 1)\times(W - F + 1) = 8 \times 10$.`),
    num("calc", "Conv2d(1, 4, 3) without padding on (N, 1, 28, 28), then MaxPool2d(2), then Flatten. What is in_features of the next Linear layer?", 676,
      "28 − 3 + 1 = 26, pooled to 13: (N, 4, 13, 13) → 4 · 13 · 13 = 676."),
    num("calc", "How many parameters does {{Conv2d(3, 16, 3)}} have (with biases)?", 448,
      "3 · 3 · 3 · 16 + 16 = 432 + 16 = 448, for any image size."),
    num("calc", "Conv2d(3, 8, 3, stride=2) without padding on a 32 × 32 image: the output is 8 × H' × H'. What is H'?", 15,
      "⌊(32 − 3)/2⌋ + 1 = 14 + 1 = 15."),
    mc("formula", "What does {{torch.nn.MaxPool2d(2)}} do to a tensor of shape (1, 3, 7, 9)?",
      ["(1, 3, 3, 4)", "(1, 3, 4, 5)", "(1, 1, 7, 9)", "(1, 3, 3.5, 4.5)", "(1, 6, 7, 9)"], 0,
      "It halves height and width, rounding down: 7 → 3, 9 → 4. Channels and batch stay."),
  ],
  "labs/debugging": [
    mc("formula", "You train on just 8 examples for 500 epochs with a large learning rate, and the training loss stays high. What does this suggest?",
      ["The model generalizes well", "Probably a bug: a correct model should be able to memorize 8 examples", "You need more data", "The test set is too small", "Batch norm is working"], 1,
      "That's the overfitting test from the debugging tutorial. Memorization isn't the goal, but a model that can't overfit a tiny set most likely has a bug."),
    mc("formula", "The loss spikes at the last step of every epoch. 60,000 examples, batch size 131, the model has batch norm. Fix?",
      ["A smaller learning rate", "{{drop_last=True}} on the training DataLoader", "{{shuffle=False}}", "Remove {{model.eval()}}", "Gradient clipping on the test set"], 1,
      "60000 = 458·131 + 2: the last minibatch has 2 examples, and batch norm on 2 samples maps every feature to ±1. Dropping that incomplete batch removes the spike."),
    mc("formula", "Memory grows with every evaluation minibatch until the program crashes. Which pair of fixes belongs together?",
      ["Bigger batches and a GPU", "Wrap evaluation in {{torch.no_grad()}} and store {{loss.item()}} instead of the loss tensor", "Call {{loss.backward()}} during evaluation", "Use {{model.train()}}", "Lower the learning rate"], 1,
      "Without no_grad, PyTorch builds a computation graph for every forward pass, and keeping the loss tensors keeps all those graphs alive."),
    mc("formula", "Someone calls {{optimizer.zero_grad()}} after {{loss.backward()}} and before {{optimizer.step()}}, with plain SGD. What happens?",
      ["Training works normally", "The parameters never change: the gradients are erased before they're used", "Gradients accumulate", "An error is raised", "The learning rate doubles"], 1,
      "zero_grad() resets .grad to zero, so step() moves by lr · 0."),
    mc("formula", "Why should the *test* DataLoader not use {{drop_last=True}}?",
      ["It makes evaluation slower", "It silently skips test examples, so the reported error is computed on fewer examples", "Batch norm needs the last batch", "It reshuffles the test set", "It has no effect on test loaders"], 1,
      "drop_last only fixes the batch-norm problem during training. At test time the model is in eval() mode and every example must count."),
    mc("formula", "A model with Dropout is evaluated without {{model.eval()}}. What do you observe?",
      ["Nothing changes", "The test error is worse than it should be and changes from run to run", "An error is raised", "Training gets faster", "Memory grows"], 1,
      "Dropout keeps switching random units off at test time, so the predictions are noisy and degraded."),
  ],

  "l5/transfer": [
    mc("formula", "In transfer learning by feature extraction, which parameters are trained?",
      ["all of them, with a large learning rate", "only the new head", "only the first layer", "none", "only batch norm's γ and β"], 1,
      "The pretrained layers are frozen; only the new head is trained. Fine-tuning would update everything, with a smaller learning rate."),
    mc("formula", "Why do pretrained CNN layers transfer to a new image task?",
      ["They memorized the new task's images", "Early layers learn generic features (edges, textures) that are useful for many tasks", "Convolutions have no parameters", "ImageNet contains every class", "Because of label smoothing"], 1,
      "The hierarchy pixels → edges → parts → objects means early features are reusable; only the task-specific head must be replaced."),
    mc("formula", "In multitask learning, what is 'negative transfer'?",
      ["sharing layers between unrelated tasks hurts their performance", "the gradient changes sign", "the head is frozen", "the learning rate is negative", "the model forgets the pretraining task"], 0,
      "Shared layers often generalize better, but tasks that don't relate can interfere with each other."),
  ],
};
