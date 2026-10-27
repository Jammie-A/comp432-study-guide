// Mock-midterm question pool, in the real exam's format: five options, exactly
// one correct, no calculator needed. Text supports $math$ and {{code}}.

export type FigureId = "early-stopping" | "learning-rate" | "arrows" | "zigzag";

export type ExamQuestion = {
  id: string;
  lecture: number;
  /** Section to revisit, "lectureId/sectionId". */
  section: string;
  points: number;
  /** Bonus questions (extra lectures): add points, never count toward the 90. */
  bonus?: boolean;
  /** One of the three example questions from the Lecture 5 handout. */
  sample?: boolean;
  prompt: string;
  code?: string;
  figure?: FigureId;
  options: [string, string, string, string, string];
  answer: number;
  explain: string;
};

const r = String.raw;

export const examPool: ExamQuestion[] = [
  // ---------------- The three examples from the handout ----------------
  {
    id: "sample-1", lecture: 3, section: "l3/logistic-train", points: 2, sample: true,
    prompt: r`Logistic regression $\hat y = \sigma(w_1x_1 + w_2x_2 + b)$ predicts $\hat y = 0.2$ for a training example $\mathbf x = (2, -1)^\top$ with label $y = 1$. One gradient descent step on this example's binary cross-entropy, with a small positive learning rate, changes $w_1$, $w_2$ and $b$. In which directions?`,
    options: [
      r`$w_1$ decreases, $w_2$ increases and $b$ decreases`,
      r`$w_1$, $w_2$ and $b$ all increase`,
      r`$w_1$ increases, $w_2$ decreases and $b$ increases`,
      r`$w_1$ increases, $w_2$ decreases and $b$ does not change`,
      r`$w_1$, $w_2$ and $b$ all decrease`,
    ],
    answer: 2,
    explain: r`The gradient is error × input: $(\hat y - y)\tilde{\mathbf x} = (0.2 - 1)(2, -1, 1) = (-1.6,\ 0.8,\ -0.8)$. GD subtracts it, so $w_1$ goes up, $w_2$ goes down, $b$ goes up. Shortcut: the error is negative (the prediction is too low), so each weight moves in the direction of the sign of its input.`,
  },
  {
    id: "sample-2", lecture: 5, section: "l5/regularization", points: 2, sample: true, figure: "arrows",
    prompt: "A classifier must sort 9 × 9 images of arrows into four classes (shown). Which random augmentation keeps every label correct?",
    options: ["A horizontal flip", "A shift by one pixel in a random direction", "A vertical flip", "A rotation by 90 degrees", "A rotation by 180 degrees"],
    answer: 1,
    explain: "A horizontal flip turns left into right; a vertical flip turns up into down; rotations by 90° or 180° change every direction. A one-pixel shift moves the arrow but keeps where it points (the arrows leave a one-pixel border, so nothing is cut off).",
  },
  {
    id: "sample-3", lecture: 3, section: "l3/multiclass", points: 2, sample: true,
    prompt: r`A 10-class classifier starts with weights close to 0. Before any training step, on a batch of 100 examples, this prints 230.26 where about $\ln 10 \approx 2.30$ was expected. What explains the printed value?`,
    code: r`Y_hat = torch.softmax(logits, dim=1)       # logits (100, 10)
loss = -torch.sum(Y * torch.log(Y_hat))     # Y one-hot (100, 10)
print(loss.item())                           # 230.26`,
    options: [
      "{{torch.log}} is natural, base 10 is needed",
      "Near-zero weights give one-hot outputs",
      "The learning rate is too large",
      r`The sum runs over the 100 examples too: no $1/N$`,
      "{{Y}} should be {{1 - Y}}",
    ],
    answer: 3,
    explain: r`With near-zero weights all logits ≈ 0, so softmax is uniform: every $\hat y_k = 1/10$ and each example's CCE is $-\ln(1/10) = \ln 10 \approx 2.30$. {{torch.sum}} adds this up over all 100 examples: $100 \cdot 2.30 = 230.26$. CCE needs the mean, $\frac1N\sum_n$. (The learning rate can't matter: no training step has happened.)`,
  },

  // ---------------- Lecture 1 ----------------
  {
    id: "l1-1", lecture: 1, section: "l1/pipeline", points: 2,
    prompt: "A model reaches 99% accuracy, measured on the same examples it was trained on. What can you conclude about its performance on new data?",
    options: [
      "It will also be about 99%",
      "It will be exactly 1% lower",
      "Nothing reliable: performance on seen examples overestimates generalization",
      "It must be overfitting",
      "It must be underfitting",
    ],
    answer: 2,
    explain: "Only held-out (test) data measures generalization. 99% on training data is compatible with a good model and with a memorizing one; you can't tell which without a test set.",
  },
  {
    id: "l1-2", lecture: 1, section: "l1/model-mse", points: 2,
    prompt: r`Predictions $\hat{\mathbf y} = (1, 2, 3)$, labels $\mathbf y = (1, 2, 5)$. What is the MSE?`,
    options: [r`$4/3$`, r`$2/3$`, r`$2$`, r`$4$`, r`$1/3$`],
    answer: 0,
    explain: r`Errors $(0, 0, -2)$, squared $(0, 0, 4)$, mean $4/3$. Option $2/3$ forgets to square; $4$ forgets the $1/N$.`,
  },
  {
    id: "l1-3", lecture: 1, section: "l1/classification-metrics", points: 2,
    prompt: "What does this print?",
    code: r`import numpy as np
P = np.array([[0.1, 0.9],
              [0.8, 0.2],
              [0.6, 0.4]])
y = np.array([1, 1, 0])
print((P.argmax(axis=1) == y).mean())`,
    options: ["0.333…", "0.5", "0.666…", "1.0", "0.0"],
    answer: 2,
    explain: r`{{P.argmax(axis=1)}} picks the most probable class per row: $(1, 0, 0)$. Compared with $y = (1, 1, 0)$: correct, wrong, correct → $2/3$.`,
  },
  {
    id: "l1-4", lecture: 1, section: "l1/classification-metrics", points: 3,
    prompt: r`A classifier puts probability $1$, $\tfrac12$ and $\tfrac14$ on the correct class of three examples. What is the categorical cross-entropy (mean over examples)?`,
    options: [r`$\ln 2$`, r`$\ln 4$`, r`$\tfrac13\ln 2$`, r`$3\ln 2$`, r`$0$`],
    answer: 0,
    explain: r`$-\tfrac13(\ln 1 + \ln\tfrac12 + \ln\tfrac14) = \tfrac13(0 + \ln 2 + 2\ln 2) = \tfrac13\cdot 3\ln 2 = \ln 2$. No calculator needed: use $\ln\tfrac14 = -2\ln 2$.`,
  },
  {
    id: "l1-5", lecture: 1, section: "l1/pipeline", points: 2,
    prompt: "Which task is unsupervised learning?",
    options: [
      "Predicting house prices from labelled sales",
      "Classifying emails as spam using labelled emails",
      "Grouping news articles into topics without any labels",
      "Learning to play a game from rewards",
      "Predicting tomorrow's temperature from past labelled data",
    ],
    answer: 2,
    explain: "No labels at all → unsupervised (here: clustering). Rewards from interaction are reinforcement learning; the rest use labels (supervised).",
  },
  {
    id: "l1-6", lecture: 1, section: "l1/naive-training", points: 2,
    prompt: r`Lecture 1's probe-and-step loop tries $+\delta$, $0$ or $-\delta$ for each parameter and evaluates every combination except "no change". How many loss evaluations per step for $P = 4$ parameters?`,
    options: ["80", "81", "8", "16", "12"],
    answer: 0,
    explain: r`$3^P - 1 = 81 - 1 = 80$. This exponential growth is why we use gradients instead.`,
  },

  // ---------------- Lecture 2 ----------------
  {
    id: "l2-1", lecture: 2, section: "l2/hyperparams", points: 3, figure: "early-stopping",
    prompt: "The figure shows the training and validation loss of a network. With early stopping, which parameters do you keep?",
    options: [
      "Those after epoch 100, where the training loss is lowest",
      "Those around epoch 40, where the validation loss is lowest",
      "Those after epoch 1",
      "Those where training and validation loss are furthest apart",
      "Those around epoch 40, where the training loss is lowest",
    ],
    answer: 1,
    explain: "Early stopping monitors the validation loss and keeps the parameters where it is lowest (≈ epoch 40). After that, training loss keeps falling but validation loss rises: overfitting. The training loss is not lowest at epoch 40.",
  },
  {
    id: "l2-2", lecture: 2, section: "l2/gd", points: 3,
    prompt: r`Gradient descent on $J(\theta) = \theta^2$ with $\theta_0 = 1$ and learning rate $\eta = 1$. What happens?`,
    options: [
      "It converges to 0 in one step",
      "It oscillates between 1 and −1 forever",
      "It diverges to infinity",
      "It converges slowly to 0",
      "It stays at 1",
    ],
    answer: 1,
    explain: r`$\theta \leftarrow \theta - 1\cdot 2\theta = -\theta$. So $1 \to -1 \to 1 \to \dots$: it never converges and never grows. ($\eta = 0.5$ would reach 0 in one step; $\eta > 1$ diverges.)`,
  },
  {
    id: "l2-3", lecture: 2, section: "l2/gd", points: 2, figure: "learning-rate",
    prompt: "The same model is trained three times with different learning rates (curves P, Q, R). Which ordering of the learning rates is most plausible?",
    options: [
      r`$\eta_P > \eta_Q > \eta_R$`,
      r`$\eta_R > \eta_Q > \eta_P$`,
      r`$\eta_Q > \eta_R > \eta_P$`,
      r`$\eta_P > \eta_R > \eta_Q$`,
      "All three are equal; only the initialization differs",
    ],
    answer: 1,
    explain: "R rises and oscillates: too large, diverging. P decreases very slowly: too small. Q drops fast and settles: in between. So R > Q > P.",
  },
  {
    id: "l2-4", lecture: 2, section: "l2/sgd", points: 2,
    prompt: "Minibatch SGD with N = 600 training examples, batch size B = 20, for 5 epochs. How many parameter updates?",
    options: ["150", "30", "3000", "120", "600"],
    answer: 0,
    explain: r`$N/B = 30$ updates per epoch, times 5 epochs = 150. "30" is a single epoch; "3000" is the number of example visits.`,
  },
  {
    id: "l2-5", lecture: 2, section: "l2/momentum", points: 3,
    prompt: r`Momentum $\mathbf v_{t+1} = \rho\mathbf v_t - \eta\nabla J(\theta_t)$, $\theta_{t+1} = \theta_t + \mathbf v_{t+1}$ with $\rho = 0.5$, $\eta = 1$, $v_0 = 0$, $\theta_0 = 10$. The gradient is $2$ at both steps. What is $\theta_2$?`,
    options: ["5", "6", "7", "8", "4"],
    answer: 0,
    explain: r`$v_1 = -2$, $\theta_1 = 8$. $v_2 = 0.5\cdot(-2) - 2 = -3$, $\theta_2 = 8 - 3 = 5$. Plain GD would give $6$: momentum adds the remembered half step.`,
  },
  {
    id: "l2-6", lecture: 2, section: "l2/adam", points: 3,
    prompt: r`Which statement about Adam (as on the slides, without bias correction, $\epsilon \approx 0$) is true?`,
    options: [
      "Adam needs full-batch gradients",
      "Two parameters whose gradients differ by a constant factor get steps of the same size",
      r`$\beta_2$ controls the update direction`,
      "Adam has no learning rate",
      "Adam uses second derivatives (the Hessian)",
    ],
    answer: 1,
    explain: r`The step is $\eta\,\mathbf v/\sqrt{\mathbf s}$ element-wise; scaling a gradient by $c$ scales $\mathbf v$ by $c$ and $\sqrt{\mathbf s}$ by $|c|$, so the step size is unchanged. $\beta_1$ (in $\mathbf v$) sets the direction, $\beta_2$ the scale. Adam uses minibatch gradients only, and has a learning rate ($10^{-3}$ by default).`,
  },
  {
    id: "l2-7", lecture: 2, section: "l2/hyperparams", points: 2,
    prompt: "A student tries 20 learning rates, picks the one with the best test accuracy, and reports that test accuracy. What is wrong?",
    options: [
      "Nothing; that is the standard procedure",
      "The reported accuracy is likely too optimistic: the test set was used for model selection",
      "20 learning rates are too few",
      "The training loss should have been used instead",
      "The learning rate is not a hyperparameter",
    ],
    answer: 1,
    explain: "Choosing on the test set leaks it into the pipeline. Hyperparameters must be chosen on a validation set; the test set is used once, at the end.",
  },
  {
    id: "l2-8", lecture: 2, section: "l2/sgd", points: 2,
    prompt: "What does this print?",
    code: r`import numpy as np
rng = np.random.default_rng(0)
steps = 0
for epoch in range(3):
    perm = rng.permutation(100)
    for i in range(0, 100, 25):
        idx = perm[i:i + 25]
        # ... compute the minibatch gradient on idx, update ...
        steps += 1
print(steps)`,
    options: ["3", "4", "12", "25", "300"],
    answer: 2,
    explain: r`Each epoch has $100/25 = 4$ minibatches; 3 epochs → 12 updates.`,
  },
  {
    id: "l2-9", lecture: 2, section: "l2/gradient", points: 2,
    prompt: r`$J(\theta_1, \theta_2) = \theta_1^2 + 4\theta_2^2$. At $\theta = (1, 1)$, in which direction does $J$ decrease fastest?`,
    options: [r`$(-1, -4)$`, r`$(1, 4)$`, r`$(-1, -1)$`, r`$(4, -1)$`, r`$(-2, -2)$`],
    answer: 0,
    explain: r`$\nabla J = (2\theta_1, 8\theta_2) = (2, 8)$. Steepest descent is $-\nabla J = (-2, -8)$, which points the same way as $(-1, -4)$. $(4, -1)$ is perpendicular: no change to first order.`,
  },
  {
    id: "l2-10", lecture: 2, section: "l2/capacity", points: 2,
    prompt: "Training loss 0.02, test loss 0.95. Which change is most likely to lower the test loss?",
    options: [
      "Train for more epochs",
      "Use a model with more parameters",
      "Collect more training data",
      "Evaluate on the training set instead",
      "Increase the learning rate",
    ],
    answer: 2,
    explain: "Tiny training loss with a big gap is overfitting. More data makes memorizing harder. More epochs or more parameters usually make overfitting worse.",
  },

  // ---------------- Lecture 3 ----------------
  {
    id: "l3-1", lecture: 3, section: "l3/closed-form", points: 2,
    prompt: r`For linear regression, $\tilde{\mathbf X}^\top\tilde{\mathbf X} = \begin{pmatrix}2&0\\0&2\end{pmatrix}$ and $\tilde{\mathbf X}^\top\mathbf y = \begin{pmatrix}4\\2\end{pmatrix}$. What is $\theta^\star = (w, b)$?`,
    options: [r`$(2, 1)$`, r`$(8, 4)$`, r`$(4, 2)$`, r`$(1, 2)$`, r`$(0.5, 1)$`],
    answer: 0,
    explain: r`$\theta^\star = (\tilde{\mathbf X}^\top\tilde{\mathbf X})^{-1}\tilde{\mathbf X}^\top\mathbf y = \tfrac12\mathbf I\,(4, 2) = (2, 1)$. $(8, 4)$ multiplies instead of inverting.`,
  },
  {
    id: "l3-2", lecture: 3, section: "l3/logistic", points: 2,
    prompt: r`Logistic regression with $\theta = (w_1, w_2, b) = (2, -1, 0)$. For $\mathbf x = (1, 3)$, what does it predict?`,
    options: [
      r`$\hat y > 0.5$, class 1`,
      r`$\hat y < 0.5$, class 0`,
      r`$\hat y = 0.5$ exactly`,
      r`$\hat y = -1$`,
      r`$\hat y = 2$`,
    ],
    answer: 1,
    explain: r`$z = 2\cdot1 - 1\cdot3 + 0 = -1 < 0$, so $\sigma(z) < 0.5$ → class 0. No need to compute $\sigma(-1)$: only the sign of $z$ matters. σ's output is always in $(0, 1)$.`,
  },
  {
    id: "l3-3", lecture: 3, section: "l3/multiclass", points: 3,
    prompt: r`A 3-class model outputs logits $\mathbf z = (0,\ \ln 2,\ \ln 5)$. What are the softmax probabilities?`,
    options: [
      r`$(\tfrac18, \tfrac14, \tfrac58)$`,
      r`$(0, \tfrac27, \tfrac57)$`,
      r`$(\tfrac13, \tfrac13, \tfrac13)$`,
      r`$(0, \ln 2, \ln 5)/\ln 10$`,
      r`$(\tfrac16, \tfrac26, \tfrac36)$`,
    ],
    answer: 0,
    explain: r`$e^{\mathbf z} = (1, 2, 5)$, sum $8$, so $(\tfrac18, \tfrac28, \tfrac58)$. A logit of 0 gives $e^0 = 1$, not probability 0.`,
  },
  {
    id: "l3-4", lecture: 3, section: "l3/one-step", points: 2,
    prompt: r`{{X}} has shape {{(50, 4)}} (bias column included), {{theta}} shape {{(4,)}}, {{y}} shape {{(50,)}}. What is the shape of {{grad}}?`,
    code: r`grad = 2 / N * X.T @ (X @ theta - y)`,
    options: ["(4,)", "(50,)", "(50, 4)", "(4, 4)", "a scalar"],
    answer: 0,
    explain: r`{{X @ theta - y}} is (50,); {{X.T}} is (4, 50); their product is (4,), the same shape as θ, as every gradient must be.`,
  },
  {
    id: "l3-5", lecture: 3, section: "l3/one-step", points: 3,
    prompt: r`{{X}} has shape {{(100, 3)}}, {{y_hat}} and {{y}} have shape {{(100,)}}. What happens when this line runs?`,
    code: r`grad = X @ (y_hat - y) / N`,
    options: [
      "It computes the correct gradient",
      "It computes the gradient times N",
      "NumPy raises an error: the shapes (100, 3) and (100,) don't align",
      "It returns a (100,) vector of per-example gradients",
      "It returns a scalar",
    ],
    answer: 2,
    explain: r`Matrix–vector product needs the inner dimensions to match: in (100, 3) @ (100,), 3 ≠ 100, so it fails. The correct line is {{X.T @ (y_hat - y) / N}}: (3, 100) @ (100,) → (3,).`,
  },
  {
    id: "l3-6", lecture: 3, section: "l3/logistic-train", points: 2,
    prompt: "Which statement about logistic regression with BCE is true?",
    options: [
      "Its loss is non-convex, so gradient descent may get stuck in local minima",
      "Its loss is convex and it has a closed-form solution",
      "Its loss is convex, but there is no closed-form solution",
      "Its gradient is the same as linear regression's, including the factor 2",
      "Its Hessian does not depend on θ",
    ],
    answer: 2,
    explain: r`The Hessian $\tfrac1N\tilde{\mathbf X}^\top\mathbf S\tilde{\mathbf X}$ is PSD (convex) but depends on θ; θ is trapped inside σ, so ∇J = 0 can't be solved in closed form. Its gradient has no factor 2 (the factor 2 belongs to MSE).`,
  },
  {
    id: "l3-7", lecture: 3, section: "l3/multiclass", points: 3,
    prompt: r`Multiclass logistic regression. For one example with $\tilde{\mathbf x} = (1, 2, 1)$ (all entries positive), the model predicts $\hat{\mathbf y} = (0.7, 0.2, 0.1)$, but the label is class 3. After one GD step on this example, how do the rows of $\tilde{\mathbf W}$ change?`,
    options: [
      "All rows increase",
      "Rows 1 and 2 decrease, row 3 increases",
      "Row 1 decreases, rows 2 and 3 increase",
      "Only row 3 changes",
      "Row 1 increases, rows 2 and 3 decrease",
    ],
    answer: 1,
    explain: r`$\nabla_{\tilde{\mathbf W}}J_n = (\hat{\mathbf y} - \mathbf y)\tilde{\mathbf x}^\top$ with $\hat{\mathbf y} - \mathbf y = (0.7, 0.2, -0.9)$. Rows 1, 2 have positive gradients (all of $\tilde{\mathbf x}$ is positive) → they decrease; row 3 is negative → it increases. Every row changes, not only the true class.`,
  },
  {
    id: "l3-8", lecture: 3, section: "l3/multiclass", points: 2,
    prompt: "Multiclass logistic regression with D = 20 features and K = 5 classes. How many parameters?",
    options: ["105", "100", "125", "25", "21"],
    answer: 0,
    explain: r`$K(D + 1) = 5\cdot 21 = 105$: one weight per feature plus one bias, for each class. 100 forgets the biases.`,
  },
  {
    id: "l3-9", lecture: 3, section: "l3/logistic", points: 2,
    prompt: r`For an example with label $y = 1$, logistic regression computes $z = \theta^\top\tilde{\mathbf x} = 0$. What is this example's BCE?`,
    options: [r`$\ln 2$`, r`$0$`, r`$0.5$`, r`$1$`, r`$\infty$`],
    answer: 0,
    explain: r`$\hat y = \sigma(0) = \tfrac12$, so $J_n = -\ln\tfrac12 = \ln 2$.`,
  },

  // ---------------- Lecture 4 ----------------
  {
    id: "l4-1", lecture: 4, section: "l4/mlp", points: 2,
    prompt: "A two-layer MLP has D = 3 inputs, H = 4 hidden units and K = 2 outputs (with biases). How many parameters?",
    options: ["26", "24", "20", "30", "18"],
    answer: 0,
    explain: r`$(D+1)H + (H+1)K = 4\cdot4 + 5\cdot2 = 16 + 10 = 26$. 20 forgets the biases.`,
  },
  {
    id: "l4-2", lecture: 4, section: "l4/mlp", points: 2,
    prompt: "A 5-layer network uses the identity f(z) = z as every activation. Which functions can it represent?",
    options: [
      "Any continuous function, given enough units",
      "Exactly the affine functions a single linear layer can represent",
      "Only constant functions",
      "Only polynomials of degree 5",
      "Only functions through the origin",
    ],
    answer: 1,
    explain: r`Products of matrices are a matrix: $\mathbf W^{(5)}\cdots\mathbf W^{(1)}\mathbf x + \text{(combined bias)}$. Without a non-linearity, depth adds parameters but no expressivity.`,
  },
  {
    id: "l4-3", lecture: 4, section: "l4/backprop-scalar", points: 3,
    prompt: r`$h_1 = w_1x$, $h_2 = \mathrm{ReLU}(h_1)$, $\hat y = w_2h_2$, $J = (\hat y - y)^2$, with $x = 2$, $w_1 = -1$, $w_2 = 3$, $y = 1$. What is $\partial J/\partial w_1$?`,
    options: ["0", "−2", "6", "−12", "4"],
    answer: 0,
    explain: r`$h_1 = -2 < 0$, so $h_2 = 0$ and $\mathrm{ReLU}'(h_1) = 0$. Every path back to $w_1$ passes through that factor 0, so $\partial J/\partial w_1 = 0$ (a "dead" ReLU: no gradient flows).`,
  },
  {
    id: "l4-4", lecture: 4, section: "l4/backprop-scalar", points: 3,
    prompt: r`$h_1 = w_1x$, $h_2 = w_2h_1$, $J = (h_2 - y)^2$ with $x = 1$, $y = 1$, $w_1 = 1$, $w_2 = 2$. What is $\partial J/\partial w_1$?`,
    options: ["4", "2", "8", "1", "0"],
    answer: 0,
    explain: r`Forward: $h_1 = 1$, $h_2 = 2$. Backward: $dJ/dh_2 = 2(2 - 1) = 2$, $dJ/dh_1 = 2\cdot w_2 = 4$, $\partial J/\partial w_1 = 4\cdot x = 4$.`,
  },
  {
    id: "l4-5", lecture: 4, section: "l4/regularization", points: 2,
    prompt: r`A degree-9 polynomial fit to 10 points with L2 regularization. You increase $\lambda$ from $10^{-5}$ to $10$. What do you expect?`,
    options: [
      "The weights grow and the model overfits more",
      "The weights shrink towards 0 and the model underfits",
      "Nothing changes: λ only affects the bias",
      "The loss becomes non-convex",
      "The training loss decreases",
    ],
    answer: 1,
    explain: r`The penalty $\tfrac\lambda2\|\theta\|^2$ dominates, pushing weights towards 0: the curve becomes too flat (over-regularization). The training loss goes up, not down.`,
  },
  {
    id: "l4-6", lecture: 4, section: "l4/basis", points: 2,
    prompt: r`Linear regression on polynomial features $\tilde\phi(x) = (x, x^2, x^3, 1)$. Why is the MSE loss still convex?`,
    options: [
      "Because polynomials are convex functions of x",
      "Because the model is still linear in the parameters θ",
      "Because the data is linearly separable",
      "Because cubic features cancel out",
      "It is not convex",
    ],
    answer: 1,
    explain: r`$\hat y = \theta^\top\tilde\phi(x)$ is linear in θ; the features are just fixed numbers per example. The loss is the same quadratic in θ as before, with $\tilde{\mathbf X}$ replaced by $\tilde{\boldsymbol\Phi}$.`,
  },
  {
    id: "l4-7", lecture: 4, section: "l4/nonconvex-uat", points: 2,
    prompt: "Which statement matches the universal approximation theorem?",
    options: [
      "Gradient descent always finds the global minimum of an MLP",
      "A one-hidden-layer MLP with enough units can approximate any continuous function on a compact domain, but training is not guaranteed to find those weights",
      "Deep networks are needed to represent non-linear functions",
      "Any activation function works, including the identity",
      "Two hidden units suffice for any function",
    ],
    answer: 1,
    explain: "The theorem is about representation, not learnability: the loss is non-convex, so the optimizer may fail. It needs a non-polynomial activation, and the number of units can be very large.",
  },
  {
    id: "l4-8", lecture: 4, section: "l4/backprop-vector", points: 3,
    prompt: r`For a layer $\mathbf z = \mathbf W\mathbf x + \mathbf b$ with $\mathbf W \in \mathbb R^{H\times D}$, backprop has $\nabla_{\mathbf z}J \in \mathbb R^H$. Which expression is $\nabla_{\mathbf x}J$?`,
    options: [
      r`$\mathbf W^\top\nabla_{\mathbf z}J$`,
      r`$\mathbf W\nabla_{\mathbf z}J$`,
      r`$(\nabla_{\mathbf z}J)\,\mathbf x^\top$`,
      r`$\nabla_{\mathbf z}J$`,
      r`$\mathbf x^\top\nabla_{\mathbf z}J$`,
    ],
    answer: 0,
    explain: r`Check shapes: $\mathbf W^\top$ is $D\times H$, times $H$ → $D$, the shape of x ✓. $(\nabla_{\mathbf z}J)\mathbf x^\top$ is the gradient for $\mathbf W$ ($H\times D$); $\nabla_{\mathbf z}J$ itself is the bias gradient.`,
  },
  {
    id: "l4-9", lecture: 4, section: "l4/nonconvex-uat", points: 3,
    prompt: "Swapping two hidden units of an MLP (together with their incoming and outgoing weights) gives exactly the same function. What follows for the training loss?",
    options: [
      "It has a unique global minimum",
      "It is convex",
      "Every minimum comes with other equally good minima elsewhere, so the loss can have several separate minima and is in general not convex",
      "Its gradient is always zero",
      "The two units must have identical weights at the optimum",
    ],
    answer: 2,
    explain: "If θ is a minimum, the swapped θ′ is one too. The slides' counterexample shows the point halfway between two such solutions can be worse — a chord below the graph, so not convex.",
  },
  {
    id: "l4-10", lecture: 4, section: "l4/mlp", points: 2,
    prompt: "What does this print?",
    code: r`import numpy as np
W1, b1 = np.ones((4, 3)), np.zeros(4)
W2, b2 = np.ones((2, 4)), np.zeros(2)
x = np.array([1., -2., 0.5])
h = np.maximum(0, W1 @ x + b1)
y_hat = W2 @ h + b2
print(y_hat.shape, y_hat)`,
    options: ["(2,) [0. 0.]", "(4,) [0. 0. 0. 0.]", "(2,) [-2. -2.]", "(3,) [1. -2. 0.5]", "(2,) [2. 2.]"],
    answer: 0,
    explain: r`Each hidden pre-activation is $1 - 2 + 0.5 = -0.5$; ReLU makes all of them 0, so $\hat{\mathbf y} = \mathbf 0$. Shapes: (4, 3)@(3,) → (4,), then (2, 4)@(4,) → (2,).`,
  },

  // ---------------- Lecture 5 ----------------
  {
    id: "l5-1", lecture: 5, section: "l5/vanishing", points: 2,
    prompt: "In a deep sigmoid network, the first layers barely change during training while the last layers learn normally. What is the most likely cause?",
    options: [
      "Exploding gradients",
      "Vanishing gradients: each sigmoid contributes a factor of at most ¼ to the backward product",
      "The learning rate is too large",
      "Overfitting",
      "Too few training examples",
    ],
    answer: 1,
    explain: "Early layers' gradients are long products of per-layer factors. With sigmoid each factor contains σ′ ≤ ¼, so the product shrinks exponentially with depth.",
  },
  {
    id: "l5-2", lecture: 5, section: "l5/vanishing", points: 2,
    prompt: r`Gradient clipping with threshold $C = 5$ is applied to $\mathbf g = (6, 8)$. What is the result?`,
    options: [r`$(3, 4)$`, r`$(5, 5)$`, r`$(6, 8)$`, r`$(0.6, 0.8)$`, r`$(5, 8)$`],
    answer: 0,
    explain: r`$\|\mathbf g\| = 10 > 5$, factor $5/10 = 0.5$ → $(3, 4)$: same direction, length 5. Clipping each entry separately at 5 (option $(5, 5)$) would change the direction.`,
  },
  {
    id: "l5-3", lecture: 5, section: "l5/convolution", points: 2,
    prompt: "A 1-D convolution with kernel size F = 5, no padding, stride 1, on an input of length 32. Output length?",
    options: ["28", "32", "27", "30", "14"],
    answer: 0,
    explain: r`$\lfloor(32 + 0 - 5)/1\rfloor + 1 = 28$.`,
  },
  {
    id: "l5-4", lecture: 5, section: "l5/convolution", points: 3,
    prompt: "A 1-D convolution with F = 3, padding P = 1, stride 2, on an input of length 9. Output length?",
    options: ["5", "4", "9", "3", "6"],
    answer: 0,
    explain: r`$\lfloor(9 + 2 - 3)/2\rfloor + 1 = \lfloor 4 \rfloor + 1 = 5$.`,
  },
  {
    id: "l5-5", lecture: 5, section: "l5/cnn", points: 3,
    prompt: "How many parameters does {{Conv2d(3, 16, kernel_size=3)}} have (with biases)?",
    options: ["448", "432", "144", "160", "464"],
    answer: 0,
    explain: r`$F_1F_2C_{\text{in}}C_{\text{out}} + C_{\text{out}} = 3\cdot3\cdot3\cdot16 + 16 = 432 + 16 = 448$. 432 forgets the biases.`,
  },
  {
    id: "l5-6", lecture: 5, section: "l5/normalization", points: 2,
    prompt: r`A student wants batch normalization for pre-activations {{Z}} of shape (batch, units) but writes the code below. What does it compute?`,
    code: r`mu = Z.mean(axis=1, keepdims=True)
sigma = Z.std(axis=1, keepdims=True)
Z_hat = (Z - mu) / sigma`,
    options: [
      "Batch normalization, correctly",
      "Layer normalization: one mean and standard deviation per sample",
      "Input standardization with the training-set statistics",
      "Dropout",
      "An error, because of keepdims",
    ],
    answer: 1,
    explain: "axis=1 averages over the units of each row (sample): that's layer norm. Batch norm needs axis=0, one mean per unit over the batch.",
  },
  {
    id: "l5-7", lecture: 5, section: "l5/regularization", points: 2,
    prompt: "A trained network with dropout gives slightly different validation accuracy every time you evaluate it, with the same weights and data. What is the most likely cause?",
    options: [
      "The learning rate is too large",
      "The model is still in training mode: call model.eval() so dropout is switched off",
      "Batch norm needs more epochs",
      "The validation set is too small",
      "Label smoothing is on",
    ],
    answer: 1,
    explain: "In training mode dropout draws new random masks every pass. model.eval() turns it off (and switches batch norm to its running averages).",
  },
  {
    id: "l5-8", lecture: 5, section: "l5/normalization", points: 2, figure: "zigzag",
    prompt: "Feature x₁ lies in [0, 1] and feature x₂ in [0, 1000]. Gradient descent on a linear model zig-zags as shown and converges slowly. What is the best fix?",
    options: [
      "Increase the learning rate",
      "Standardize both features with the training-set mean and standard deviation",
      "Add more hidden layers",
      "Use label smoothing",
      "Remove the bias",
    ],
    answer: 1,
    explain: "The gradient for each weight scales with its feature, so one learning rate can't suit both (an elongated bowl). Standardizing puts both features on the same scale. A larger learning rate would make the zig-zag diverge.",
  },
  {
    id: "l5-9", lecture: 5, section: "l5/regularization", points: 2,
    prompt: r`Label smoothing with $K = 5$ classes and $\epsilon = 0.1$. What is the smoothed target?`,
    options: [
      "0.92 for the correct class, 0.02 for each other class",
      "0.9 for the correct class, 0.1 for each other class",
      "0.9 for the correct class, 0.025 for each other class",
      "1 for the correct class, 0.1 for each other class",
      "0.2 for every class",
    ],
    answer: 0,
    explain: r`$(1 - 0.1)\cdot 1 + 0.1/5 = 0.92$ for the correct class, $0 + 0.1/5 = 0.02$ for the others. They still sum to 1.`,
  },
  {
    id: "l5-10", lecture: 5, section: "l5/transfer", points: 4,
    prompt: "What happens when this transfer-learning code trains?",
    code: r`model = torchvision.models.resnet18(weights=weights)
model.fc = torch.nn.Linear(512, 37)        # new head for 37 classes
for p in model.parameters():
    p.requires_grad = False
opt = torch.optim.SGD(model.fc.parameters(), lr=0.01)`,
    options: [
      "Only the new head is trained: feature extraction",
      "All layers are fine-tuned",
      "Nothing can be trained: the freezing loop also froze the new head, so no parameter requires a gradient",
      "Only the first convolution is trained",
      "The head is trained, and batch norm is disabled",
    ],
    answer: 2,
    explain: "The freezing loop runs after the head was replaced, so it freezes model.fc too. No parameter requires a gradient (in PyTorch, loss.backward() then raises an error). The slides freeze first, then create the new head.",
  },
  {
    id: "l5-11", lecture: 5, section: "l5/remedies", points: 2,
    prompt: "Why can residual networks be trained with hundreds of layers?",
    options: [
      "They use no activation functions",
      "The shortcut h ↦ h + f(h) lets the gradient pass through the identity path, adding 1 to each layer's factor",
      "They have fewer parameters per layer",
      "They are convex",
      "They don't use backpropagation",
    ],
    answer: 1,
    explain: "A sum node copies its gradient to both inputs, so the gradient reaches early layers undiminished along the shortcut, even if f's own factor is tiny.",
  },

  // ---------------- Labs (mapped to the lecture they belong to) ----------------
  {
    id: "lab-1", lecture: 1, section: "labs/numpy", points: 2,
    prompt: "What does this print?",
    code: r`import numpy as np
x = np.arange(6.).reshape(2, 3)
y = x[:, :2]
y[0, 0] = 100
print(x[0, 0])`,
    options: ["100.0", "0.0", "an IndexError", "1.0", "6.0"],
    answer: 0,
    explain: "Slicing returns a view that shares memory with x, so assigning into y changes x. Only {{.copy()}} (or fancy indexing) makes independent data.",
  },
  {
    id: "lab-2", lecture: 1, section: "labs/numpy", points: 2,
    prompt: "A student shuffles a dataset like this before training. What is the consequence?",
    code: r`X = X[rng.permutation(N)]
Y = Y[rng.permutation(N)]`,
    options: [
      "Nothing: both arrays are shuffled",
      "Inputs and labels no longer match, so the model learns from mostly wrong labels",
      "Only the first example is shuffled",
      "NumPy raises an error",
      "The data is shuffled twice, which is slower but correct",
    ],
    answer: 1,
    explain: "Two calls give two different permutations. Use one: {{perm = rng.permutation(N); X, Y = X[perm], Y[perm]}}. (The debugging tutorial's first step: check that inputs and labels still match.)",
  },
  {
    id: "lab-3", lecture: 2, section: "labs/finite-differences", points: 2,
    prompt: r`Central differences $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ approximate $f'(2) = 12$ for $f(x) = x^3$. Which $\epsilon$ gives the most accurate estimate in double precision?`,
    options: [r`$\epsilon = 1$`, r`$\epsilon = 10^{-5}$`, r`$\epsilon = 10^{-15}$`, r`$\epsilon = 10^{-300}$`, "All are equally accurate"],
    answer: 1,
    explain: r`$\epsilon = 1$ is too large (the estimate is 13). $10^{-15}$ and smaller suffer floating-point cancellation (≈ 11.99, or 0). $10^{-5}$ gives 12.0000000002.`,
  },
  {
    id: "lab-4", lecture: 3, section: "labs/from-scratch", points: 2,
    prompt: "Our own gradient descent on the plain BCE and scikit-learn's {{LogisticRegression()}} (default settings) return clearly different weights on the same data. What's the most likely reason?",
    options: [
      "scikit-learn uses MSE",
      "scikit-learn adds an L2 penalty by default (C = 1); C = np.inf gives the plain BCE",
      "Our gradient is missing a factor 2",
      "scikit-learn cannot handle two features",
      "The data was not shuffled",
    ],
    answer: 1,
    explain: "C is the inverse regularization strength. With C = np.inf (no penalty) both minimize the same objective and agree.",
  },
  {
    id: "lab-5", lecture: 4, section: "labs/pytorch", points: 3,
    prompt: "This model trains poorly. What is wrong?",
    code: r`model = torch.nn.Sequential(
    torch.nn.Linear(784, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 10), torch.nn.Softmax(dim=1),
)
loss_fn = torch.nn.CrossEntropyLoss()`,
    options: [
      "ReLU should be Tanh",
      "CrossEntropyLoss expects logits and applies the softmax itself; the extra Softmax layer applies it twice",
      "Softmax must use dim=0",
      "The second Linear needs 784 inputs",
      "CrossEntropyLoss needs one-hot targets",
    ],
    answer: 1,
    explain: "Remove the Softmax layer. CrossEntropyLoss = softmax + CCE, taking raw logits and integer class labels.",
  },
  {
    id: "lab-6", lecture: 4, section: "labs/pytorch", points: 3,
    prompt: "What does this print (rounded)?",
    code: r`x = torch.arange(3, dtype=torch.float32, requires_grad=True)   # [0, 1, 2]
y = ((x + 2) ** 2 + 3).mean()
y.backward()
print(x.grad)`,
    options: [
      r`$(4/3,\ 2,\ 8/3)$`,
      r`$(4,\ 6,\ 8)$`,
      r`$(2,\ 3,\ 4)$`,
      r`$(0,\ 1,\ 2)$`,
      r`$(1/3,\ 1/3,\ 1/3)$`,
    ],
    answer: 0,
    explain: r`$\partial y/\partial x_i = \frac13\cdot2(x_i+2) = \frac23(2, 3, 4) = (4/3, 2, 8/3)$. $(4, 6, 8)$ forgets the mean's $1/3$.`,
  },
  {
    id: "lab-7", lecture: 4, section: "labs/pytorch", points: 2,
    prompt: "In Lab 4, which scikit-learn model is exactly multiclass logistic regression?",
    options: [
      "{{MLPClassifier(hidden_layer_sizes=(32,))}}",
      "{{MLPClassifier(hidden_layer_sizes=())}}",
      "{{MLPClassifier(hidden_layer_sizes=(10, 10))}}",
      "{{LinearRegression()}}",
      "{{MinMaxScaler()}}",
    ],
    answer: 1,
    explain: "No hidden layers: inputs connect directly to the softmax outputs. Note {{(32)}} without the comma isn't even a tuple.",
  },
  {
    id: "lab-8", lecture: 5, section: "labs/debugging", points: 3,
    prompt: "An MLP with batch normalization is trained on 60,000 images with batch size 131. The per-step loss spikes at the end of every epoch. What is the cause?",
    options: [
      "The learning rate is too large",
      "The last minibatch has only 2 examples, and batch norm on 2 samples maps every feature to ±1",
      "The data is not shuffled",
      "Adam's second moment explodes",
      "The test set leaks into training",
    ],
    answer: 1,
    explain: "60000 = 458·131 + 2. Fix: {{drop_last=True}} for the training loader (never for the test loader).",
  },
  {
    id: "lab-9", lecture: 5, section: "labs/debugging", points: 2,
    prompt: "This evaluation function runs out of memory on a large test set. Why?",
    code: r`model.eval()
losses = []
for images, labels in test_loader:
    losses.append(loss_fn(model(images), labels))`,
    options: [
      "model.eval() uses more memory than model.train()",
      "Without torch.no_grad(), every forward pass builds a computation graph, and the stored loss tensors keep all of them alive",
      "The test loader is not shuffled",
      "Batch norm stores running averages",
      "losses should be a NumPy array",
    ],
    answer: 1,
    explain: "Wrap the loop in {{with torch.no_grad():}} and append {{loss.item()}} (a float). Note that eval() only changes dropout/batch norm, not gradient tracking.",
  },
  {
    id: "lab-11", lecture: 5, section: "labs/debugging", points: 2,
    prompt: "An MLP built from {{Linear(784, 500)}} raises {{RuntimeError: shapes cannot be multiplied (28x28 and 784x500)}} on minibatches of MNIST images of shape (131, 1, 28, 28). Fix?",
    options: [
      "Transpose the weight matrix",
      "Flatten each image: {{torch.flatten(images, start_dim=1)}} → (131, 784)",
      "Use a batch size of 28",
      "Call {{model.eval()}}",
      "Move the model to the GPU",
    ],
    answer: 1,
    explain: "PyTorch treats every axis but the last as a batch axis, so it tried to multiply rows of length 28 by a 784 × 500 matrix. Flatten from axis 1 (axis 0 is the batch).",
  },

  // ---------------- Bonus: extra lectures ----------------
  {
    id: "bonus-1", lecture: 3, section: "l3/summary", points: 2, bonus: true,
    prompt: "(Bonus, extra lecture) Minimizing the MSE of linear regression is equivalent to maximum likelihood under which noise model?",
    options: ["Gaussian", "Bernoulli", "Categorical", "Uniform", "Poisson"],
    answer: 0,
    explain: "Lecture 3's summary table: MSE ↔ Gaussian, BCE ↔ Bernoulli, CCE ↔ categorical likelihood.",
  },
  {
    id: "bonus-2", lecture: 3, section: "l3/summary", points: 2, bonus: true,
    prompt: "(Bonus, extra lecture) Binary cross-entropy is the negative log-likelihood of which distribution?",
    options: ["Gaussian", "Bernoulli", "Categorical with K = 10", "Laplace", "Exponential"],
    answer: 1,
    explain: "One probability of a binary outcome: Bernoulli. CCE generalizes it to the categorical distribution.",
  },
  {
    id: "bonus-3", lecture: 4, section: "l4/regularization", points: 2, bonus: true,
    prompt: "(Bonus, extra lecture) L2 regularization corresponds to maximum a posteriori (MAP) estimation with which prior on θ?",
    options: ["A Gaussian prior", "A uniform prior", "A Bernoulli prior", "No prior", "A prior that favours large weights"],
    answer: 0,
    explain: "Lecture 4: extra lecture 5 derives the λ/2‖θ‖² penalty from a Gaussian prior on θ in MAP estimation.",
  },
];
