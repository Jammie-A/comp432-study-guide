import { Callout, FromLab, Formula, H, M, MB, MathCode, Slides, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { CapacityWidget, GD1DWidget, SGDWidget, ValleyWidget } from "../widgets/l2";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

export const lecture2: Lecture = {
  id: "l2",
  number: 2,
  title: "Optimization",
  sections: [
    {
      id: "capacity",
      title: "Capacity, underfitting & overfitting",
      summary: "How complex should a model be? Compare training and test loss to find out.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            A model can only learn functions it is able to represent. That range of functions is its <strong>capacity</strong>.
            A straight line can't follow a curve (too little capacity), while a wiggly degree-11 polynomial can pass through
            every training point, including the noise (too much capacity). What we really want is a model that does well
            on <em>new</em> data: one that <strong>generalizes</strong>.
          </p>
          <Table
            head={["", "Training loss", "Test loss", "What's happening"]}
            rows={[
              ["Underfitting", "high", "high", "Model too simple: it can't even fit the training data."],
              ["Good fit", "low", "low (small gap)", "Learned the pattern."],
              ["Overfitting", "very low", "much higher", "Memorized the training data, including noise. Big train–test gap."],
            ]}
          />
          <p>
            A rough measure of capacity is the <strong>number of parameters <M>P</M></strong>. The line
            <M>{t`\hat y = wx + b`}</M> has <M>P = 2</M>; a quadratic has <M>P=3</M>; a degree-11 polynomial has <M>P=12</M>.
          </p>

          <H>Visual</H>
          <CapacityWidget />
          <Think q="Set degree M = 9 with N = 10 samples. Then raise N to 100 without changing M. What happens to the test MSE, and why?">
            With N = 10 the degree-9 polynomial has 10 parameters, enough to hit all 10 points exactly, so the training MSE is about 0
            and the test MSE is large (overfitting). With N = 100 it can no longer memorize every point, so it's forced to follow the
            overall pattern and test MSE drops. This is the slide's point that <strong>more data prevents overfitting</strong>.
          </Think>

          <H>Three things that control generalization</H>
          <ol>
            <li><strong>Capacity</strong>: too low gives underfitting, too high gives overfitting.</li>
            <li><strong>Training time</strong>: a high-capacity model usually fits the general pattern first, and
              only later starts memorizing noise. So the test loss can go <em>down then up</em> as you train more epochs.</li>
            <li><strong>Number of training samples N</strong>: more data makes memorizing harder.</li>
          </ol>
          <Callout kind="intuition">
            <strong>Occam's razor:</strong> among functions that explain the data, prefer the simplest. Techniques that push towards
            simple functions (regularization, data augmentation, dropout…) come later in the course.
          </Callout>
          <Callout kind="exam">
            Be ready to diagnose from numbers: "training loss 0.00, test loss 0.17" → overfitting. "training 0.30, test 0.33" →
            underfitting (both high, small gap). You need <em>both</em> numbers to tell; one alone isn't enough.
          </Callout>
          <Slides lecture={2} pages="8–13" />
        </>
      ),
    },
    {
      id: "gradient",
      title: "Derivatives and the gradient",
      summary: "The gradient tells you which way is uphill, so you know which way is downhill.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Training means finding parameters that make the loss small: <M>{t`\theta^\star = \arg\min_\theta J(\theta)`}</M>
            ("the <M>\theta</M> that gives the smallest <M>J</M>"). In Lecture 1 we found the direction by trial (probing every
            neighbour). The derivative tells us the direction <em>directly</em>, without trying.
          </p>

          <H>The formula, decoded</H>
          <Formula
            plain="Nudge θ by a tiny ε, see how much J changes, and divide by ε: the slope of J at θ, i.e. how fast the loss changes when θ moves."
            title="Derivative (one parameter)"
            tex={t`\frac{dJ}{d\theta} = \lim_{\epsilon\to 0}\frac{J(\theta+\epsilon) - J(\theta)}{\epsilon}`}
            symbols={[
              [t`J(\theta)`, "the loss as a function of the parameter"],
              [t`\epsilon`, "a tiny nudge to θ"],
              [t`\frac{dJ}{d\theta}`, "slope: how much J changes per unit change of θ. Positive means J goes up if θ increases."],
            ]}
          />
          <p>
            <strong>Critical points</strong> are where the slope is zero: <M>{t`\frac{dJ}{d\theta}=0`}</M>. That includes minima,
            maxima and <strong>saddle points</strong> (flat, but neither a min nor a max). A <em>local</em> minimum is lower than its
            neighbours; the <em>global</em> minimum is the lowest overall.
          </p>
          <Formula
            plain="Do the same for each parameter separately (nudge only θ_m and keep the others fixed), and stack the P slopes into a vector. The gradient points in the direction in which J increases fastest."
            title="Partial derivatives and the gradient (P parameters)"
            tex={t`\frac{\partial J}{\partial \theta_m} = \lim_{\epsilon\to 0}\frac{J(\theta+\epsilon\, \mathbf{e}_m) - J(\theta)}{\epsilon}, \qquad \nabla_\theta J = \left(\frac{\partial J}{\partial\theta_1},\dots,\frac{\partial J}{\partial\theta_P}\right)^{\top} \in \mathbb{R}^P`}
            symbols={[
              [t`\mathbf{e}_m`, <>unit vector: all zeros except a 1 at position <M>m</M>. So only <M>{t`\theta_m`}</M> moves; all others are frozen.</>],
              [t`\frac{\partial J}{\partial\theta_m}`, "partial derivative: the slope along one parameter's axis"],
              [t`\nabla_\theta J`, <>the <strong>gradient</strong>: all partial derivatives stacked into a vector, same shape as <M>\theta</M> (P entries)</>],
            ]}
          />

          <H>Why the gradient points uphill</H>
          <p>Take a small step of length <M>\epsilon</M> in some unit direction <M>{t`\mathbf{d}`}</M>. A Taylor expansion says:</p>
          <MB>{t`J(\theta+\epsilon \mathbf d) - J(\theta) \approx \epsilon\,(\nabla_\theta J)^\top \mathbf d = \epsilon\,\|\nabla_\theta J\|\cos\alpha`}</MB>
          <p>where <M>\alpha</M> is the angle between the step and the gradient. So the change in loss depends only on <M>{t`\cos\alpha`}</M>:</p>
          <Table
            head={["Direction d", "α", "cos α", "Effect on J"]}
            rows={[
              ["along ∇J", "0°", "1", "steepest ascent (J increases fastest)"],
              ["against ∇J (i.e. −∇J)", "180°", "−1", "steepest descent (J decreases fastest)"],
              ["perpendicular to ∇J", "90°", "0", "no change (moving along a contour line)"],
            ]}
          />

          <H>Worked example</H>
          <Steps
            title={t`Gradient of J(θ) = θ₁² + 3θ₂² at θ = (1, 2)`}
            steps={[
              { title: "Partial derivative w.r.t. θ₁ (treat θ₂ as a constant)", body: <MB>{t`\frac{\partial J}{\partial\theta_1} = 2\theta_1 = 2\cdot 1 = 2`}</MB> },
              { title: "Partial derivative w.r.t. θ₂ (treat θ₁ as a constant)", body: <MB>{t`\frac{\partial J}{\partial\theta_2} = 6\theta_2 = 6\cdot 2 = 12`}</MB> },
              { title: "Stack them into the gradient", body: <MB>{t`\nabla_\theta J = \begin{pmatrix}2\\12\end{pmatrix}`}</MB> },
              { title: "Interpret", body: <p>J rises 6× faster along θ₂ than along θ₁ here. To go downhill fastest, move in direction <M>{t`-\nabla J = (-2,-12)`}</M>.</p> },
              { title: "Check numerically (finite differences)", body: <p>With ε = 0.001: <M>{t`\frac{J(1.001, 2) - J(1,2)}{0.001} = \frac{13.002001 - 13}{0.001} \approx 2.001`}</M> ✓ (exact value 2 when ε → 0).</p> },
            ]}
          />

          <H>Math → code</H>
          <PyCell
            code={py`
import numpy as np

def J(theta):
    return theta[0]**2 + 3 * theta[1]**2

def grad_J(theta):              # analytic gradient
    return np.array([2 * theta[0], 6 * theta[1]])

def numerical_grad(J, theta, eps=1e-6):
    g = np.zeros_like(theta)
    for m in range(len(theta)):
        e_m = np.zeros_like(theta)
        e_m[m] = 1.0            # unit vector: only theta_m moves
        g[m] = (J(theta + eps * e_m) - J(theta)) / eps
    return g

theta = np.array([1.0, 2.0])
print("analytic :", grad_J(theta))
print("numerical:", numerical_grad(J, theta))
`}
          />
          <Callout kind="note">
            Two ways to minimize (slide 17): <strong>analytically</strong>, by solving <M>{t`\nabla_\theta J = 0`}</M> in closed form
            (only possible for simple models like linear regression), or <strong>numerically</strong>, by improving step by step
            (gradient descent). Machine learning is dominated by the second.
          </Callout>
          <Callout kind="mistake">
            The gradient is a <em>vector with the same shape as θ</em>, not a number. And it points <em>uphill</em>: forgetting the
            minus sign in the update is a classic bug.
          </Callout>
          <FromLab to="labs/finite-differences" label="Finite differences">
            Labs 1–2 turned the derivative's definition into code with the <em>central</em> difference{" "}
            <M>{t`\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}`}</M>. Its accuracy depends on ε: ε = 1 is too coarse, and
            ε = 10⁻¹⁵ is ruined by floating-point rounding; around 10⁻⁵ works. Lab 3 then had you derive the Rosenbrock
            gradient by hand and plot it as arrows over the function's heatmap.
          </FromLab>
          <Slides lecture={2} pages="15–20" />
        </>
      ),
    },
    {
      id: "gd",
      title: "Gradient descent & the learning rate",
      summary: "Repeatedly take a small step against the gradient. The learning rate sets the step size.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Gradient descent is the engine behind almost all of machine learning. The idea: stand somewhere on the loss
            surface, look which way is steepest downhill, take a small step, and repeat.
          </p>
          <H>The formula, decoded</H>
          <Formula
            plain="New parameters = old parameters minus a small step (η times the gradient): a step downhill on the loss."
            title="Gradient descent update"
            sheet="yes"
            tex={t`\theta_{t+1} = \theta_t - \eta\, \nabla_\theta J(\theta_t)`}
            symbols={[
              [t`\theta_t`, "parameters after t updates (θ₀ is the random initialization)"],
              [t`\eta`, <>learning rate ("eta"): how big a step to take. A <strong>hyperparameter</strong> you choose.</>],
              [t`\nabla_\theta J(\theta_t)`, "gradient at the current parameters (points uphill)"],
              [t`-`, "minus: we go against the gradient, i.e. downhill"],
            ]}
          />
          <p>The algorithm: choose η, initialize θ randomly → compute the gradient → update θ → good enough? If not, repeat.</p>
          <H>Visual</H>
          <GD1DWidget />
          <Think q="For J(θ) = 1.5θ², the update is θ ← θ − η·3θ = (1 − 3η)θ. For which η does θ shrink towards 0?">
            θ shrinks when |1 − 3η| &lt; 1, i.e. 0 &lt; η &lt; 2/3. For η = 1/3 it jumps to 0 in one step. Between 1/3 and 2/3 it
            overshoots and flips sign each step but still converges. Above 2/3 it grows: divergence.
          </Think>
          <H>Worked example</H>
          <Steps
            title="Three GD steps on J(θ) = 1.5θ², θ₀ = −1.8, η = 0.1"
            steps={[
              { title: "Gradient formula", body: <MB>{t`J'(\theta) = 3\theta`}</MB> },
              { title: "Step 1", body: <MB>{t`\theta_1 = -1.8 - 0.1\cdot(3\cdot -1.8) = -1.8 + 0.54 = -1.26`}</MB> },
              { title: "Step 2", body: <MB>{t`\theta_2 = -1.26 - 0.1\cdot(-3.78) = -0.882`}</MB> },
              { title: "Step 3", body: <MB>{t`\theta_3 = -0.882 - 0.1\cdot(-2.646) = -0.6174`}</MB> },
              { title: "Pattern", body: <p>Each step multiplies θ by (1 − 0.3) = 0.7. Steps get <em>smaller</em> automatically as we approach the minimum, because the gradient shrinks.</p> },
            ]}
          />
          <PyCell
            code={py`
theta, eta = -1.8, 0.1
for t in range(5):
    grad = 3 * theta            # J'(theta) for J = 1.5 * theta**2
    theta = theta - eta * grad
    print(f"step {t+1}: theta = {theta:.4f}, J = {1.5 * theta**2:.4f}")
`}
          />
          <Callout kind="intuition">
            <strong>Learning rate:</strong> too small → many updates needed; just right → swiftly reaches the minimum; too large →
            overshoots and diverges. Also, GD is <em>local</em>: different starting points can end in different local minima.
          </Callout>
          <Slides lecture={2} pages="21–23" />
        </>
      ),
    },
    {
      id: "least-squares",
      title: "Example: linear least squares by GD",
      summary: "Derive the MSE gradient with the chain rule and train a line in a few lines of NumPy.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            This is the full recipe in its smallest form: model → loss → gradient (via chain rule) → update loop. Every
            later model (logistic regression, neural nets) uses exactly this skeleton. Learn this code by heart.
          </p>
          <H>The setup</H>
          <Formula
            plain="Predict with a straight line (slope w, intercept b), measure the fit with the MSE, and treat w and b together as the parameter vector θ to optimize."
            tex={t`\hat y_n = w x_n + b, \qquad J(\theta) = \frac1N\sum_{n=1}^N(\hat y_n - y_n)^2, \qquad \theta = \begin{pmatrix}w\\b\end{pmatrix}`}
            symbols={[
              [t`x_n, y_n`, "the n-th training input and label (scalars here)"],
              [t`\hat y_n`, "prediction for example n"],
              [t`\hat y_n - y_n`, <>the <strong>residual</strong> (error) of example n; in code: <code>residual</code></>],
              [t`N`, "number of training examples"],
            ]}
          />
          <H>Deriving the gradient, step by step</H>
          <Steps
            steps={[
              { title: "Chain rule: J depends on w only through ŷₙ", body: <MB>{t`\frac{\partial J}{\partial w} = \frac1N\sum_n \frac{\partial (\hat y_n - y_n)^2}{\partial \hat y_n}\cdot\frac{\partial \hat y_n}{\partial w}`}</MB> },
              { title: "Outer derivative: d/du of u² is 2u", body: <MB>{t`\frac{\partial (\hat y_n - y_n)^2}{\partial \hat y_n} = 2(\hat y_n - y_n)`}</MB> },
              { title: "Inner derivatives of ŷₙ = w xₙ + b", body: <MB>{t`\frac{\partial \hat y_n}{\partial w} = x_n, \qquad \frac{\partial \hat y_n}{\partial b} = 1`}</MB> },
              { title: "Put it together", body: <MB>{t`\frac{\partial J}{\partial w} = \frac2N\sum_n(\hat y_n - y_n)\,x_n, \qquad \frac{\partial J}{\partial b} = \frac2N\sum_n(\hat y_n - y_n)`}</MB> },
              { title: "Read it in words", body: <p>"Average of (error × input)" for w, "average of error" for b, times 2. Big errors on big inputs push w the most.</p> },
            ]}
          />
          <Callout kind="note">
            Some textbooks put ½ in front of the MSE so the 2 cancels. This course does <strong>not</strong> (consistent with
            PyTorch's <code>MSELoss</code>), so keep the 2.
          </Callout>
          <H>Worked example with numbers</H>
          <p>Data (Lecture 3's mini iris set): <M>{t`x = (1, 3, 5)`}</M>, <M>{t`y = (0.2, 1.1, 1.7)`}</M>. Start <M>{t`w=0.5,\ b=1.0`}</M>, <M>{t`\eta = 0.025`}</M>.</p>
          <Steps
            title="First epoch by hand"
            steps={[
              { title: "Forward: predictions", body: <MB>{t`\hat y = 0.5\cdot(1,3,5) + 1 = (1.5,\ 2.5,\ 3.5)`}</MB> },
              { title: "Residuals", body: <MB>{t`r = \hat y - y = (1.3,\ 1.4,\ 1.8)`}</MB> },
              { title: "Loss", body: <MB>{t`J = \tfrac13(1.3^2 + 1.4^2 + 1.8^2) = \tfrac13(1.69 + 1.96 + 3.24) \approx 2.297`}</MB> },
              { title: "Gradient for w", body: <MB>{t`\tfrac23(1.3\cdot1 + 1.4\cdot3 + 1.8\cdot5) = \tfrac23(14.5) \approx 9.667`}</MB> },
              { title: "Gradient for b", body: <MB>{t`\tfrac23(1.3 + 1.4 + 1.8) = \tfrac23(4.5) = 3.0`}</MB> },
              { title: "Update", body: <MB>{t`w = 0.5 - 0.025\cdot 9.667 \approx 0.2583, \qquad b = 1.0 - 0.025\cdot 3.0 = 0.925`}</MB> },
            ]}
          />
          <H>Math → code</H>
          <MathCode
            rows={[
              [t`\hat y_n = w x_n + b`, "y_hat = w * x_train + b", "shape (N,) — all n at once"],
              [t`r_n = \hat y_n - y_n`, "residual = y_hat - y_train", "(N,)"],
              [t`J = \frac1N\sum_n r_n^2`, "loss = (residual ** 2).mean()", "scalar; .mean() is the 1/N Σ"],
              [t`\frac{\partial J}{\partial w}=\frac2N\sum_n r_n x_n`, "grad_w = 2 * (residual * x_train).mean()", "* is element-wise"],
              [t`\frac{\partial J}{\partial b}=\frac2N\sum_n r_n`, "grad_b = 2 * residual.mean()", ""],
              [t`w \leftarrow w - \eta\,\partial J/\partial w`, "w = w - lr * grad_w", "lr is η"],
            ]}
          />
          <PyCell
            title="The slide's training loop (runnable)"
            code={py`
import numpy as np
x_train = np.array([1.0, 3.0, 5.0])
y_train = np.array([0.2, 1.1, 1.7])

w, b = 0.5, 1.0
epochs, lr = 10, 0.025
for epoch in range(epochs):
    # forward pass
    y_hat = w * x_train + b
    residual = y_hat - y_train
    loss = (residual ** 2).mean()
    # gradient
    grad_w = 2 * (residual * x_train).mean()
    grad_b = 2 * residual.mean()
    # gradient descent update
    w = w - lr * grad_w
    b = b - lr * grad_b
    print(f"epoch {epoch}: loss={loss:.4f}  w={w:.4f}  b={b:.4f}")
`}
          />
          <Think q="Notice that `loss` is printed before the update in each epoch. What is the loss printed at epoch 0?">
            2.2967, the loss of the <em>initial</em> parameters (w = 0.5, b = 1.0), because it's computed in the forward pass before
            the update. This "off by one" detail is a typical code-reading question.
          </Think>
          <Callout kind="exam">
            Try changing <code>lr</code> to <code>0.1</code> and run again: the loss explodes. Then explain why using the
            learning-rate section: here <M>{t`\eta`}</M> is above <M>{t`2/\lambda_{\max}`}</M> of the loss's curvature.
          </Callout>
          <FromLab to="labs/from-scratch" label="Linear & logistic regression from scratch">
            Lab 3 writes this loop in matrix form, with the gradient{" "}
            <M>{t`\tfrac2N\tilde{\mathbf X}^\top(\tilde{\mathbf X}\theta - \mathbf y)`}</M> from Lecture 3, and adds a stopping
            rule: quit once every entry of the gradient is below a small tolerance, because a zero gradient means you're at the
            minimum.
          </FromLab>
          <Slides lecture={2} pages="25–27" />
        </>
      ),
    },
    {
      id: "autodiff",
      title: "Automatic differentiation (PyTorch)",
      summary: "Let the library compute gradients: forward pass, loss.backward(), opt.step().",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            For a line, deriving the gradient by hand was easy. For deep networks with millions of parameters it is tedious
            and error-prone. <strong>Automatic differentiation</strong> (autodiff) computes exact gradients for you by
            tracking every operation in the forward pass (Lecture 4 explains how: backpropagation).
          </p>
          <H>The PyTorch training loop</H>
          <PyCell
            readOnly
            title="PyTorch (read-only: PyTorch can't run in the browser)"
            code={py`
import torch
import torch.nn as nn

X_train = torch.tensor([[1.0], [2.0]])   # inputs, shape (N, D)
Y_train = torch.tensor([[2.0], [4.0]])   # labels, shape (N, K)
model = nn.Linear(1, 1)                  # Linear(D, K); a deep net goes here
loss_fn = nn.MSELoss()                   # objective
opt = torch.optim.SGD(model.parameters(), lr=0.1)   # optimizer

for epoch in range(5):
    opt.zero_grad()                              # reset the stored gradients
    loss = loss_fn(model(X_train), Y_train)      # forward pass
    loss.backward()                              # all gradients, automatically
    opt.step()                                   # gradient descent update
`}
          />
          <Table
            head={["PyTorch line", "What it means mathematically"]}
            rows={[
              [<code>nn.Linear(1, 1)</code>, <>the model <M>{t`\hat y = wx + b`}</M> with D = 1 input and K = 1 output; w, b randomly initialized</>],
              [<code>nn.MSELoss()</code>, <M>{t`J = \frac1N\sum_n(\hat y_n - y_n)^2`}</M>],
              [<code>opt.zero_grad()</code>, "set stored gradients to 0. PyTorch ADDS new gradients to old ones, so forgetting this is a bug."],
              [<code>loss.backward()</code>, <>computes <M>{t`\nabla_\theta J`}</M> and stores it in each parameter's <code>.grad</code></>],
              [<code>opt.step()</code>, <M>{t`\theta \leftarrow \theta - \eta \nabla_\theta J`}</M>],
            ]}
          />
          <H>The same loop in NumPy (runnable)</H>
          <p>Here is exactly what those PyTorch lines do, with the gradient written by hand:</p>
          <PyCell
            code={py`
import numpy as np
X_train = np.array([[1.0], [2.0]])   # (N, D) = (2, 1)
Y_train = np.array([[2.0], [4.0]])   # (N, K) = (2, 1)
w, b = 0.3, -0.1                     # stand-in for nn.Linear's random init
lr = 0.1

for epoch in range(5):
    y_hat = w * X_train + b                        # model(X_train)
    loss = ((y_hat - Y_train) ** 2).mean()         # loss_fn(...)
    grad_w = 2 * ((y_hat - Y_train) * X_train).mean()   # loss.backward()
    grad_b = 2 * (y_hat - Y_train).mean()
    w, b = w - lr * grad_w, b - lr * grad_b        # opt.step()
    print(f"epoch {epoch}: loss = {loss:.4f}, w = {w:.3f}, b = {b:.3f}")
`}
          />
          <Callout kind="exam">
            Know the order <code>zero_grad → forward → loss → backward → step</code> and what each does. Questions like "what goes
            wrong if we delete <code>opt.zero_grad()</code>?" (gradients accumulate across epochs, so steps get wrong and too big)
            are typical.
          </Callout>
          <FromLab to="labs/pytorch-basics" label="PyTorch tensors & autograd">
            The Lab 4 PyTorch tutorial checks autograd by hand: for{" "}
            <code>y = ((x + 2) ** 2 + 3).mean()</code> at x = (0, 1, 2), <code>y.backward()</code> gives{" "}
            <code>x.grad</code> = (4/3, 2, 8/3). The debugging tutorial's third bug is the missing{" "}
            <code>zero_grad()</code> from this page: the code runs, but the loss stays large and erratic.
          </FromLab>
          <Slides lecture={2} pages="28–29" />
        </>
      ),
    },
    {
      id: "momentum",
      title: "Momentum: fixing narrow valleys",
      summary: "Remember past steps so that zigzags cancel and consistent directions speed up.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Real loss surfaces often curve much more steeply in one direction than in another, like a long narrow valley.
            Plain GD then zigzags across the valley and crawls along it. Momentum fixes this.
          </p>
          <H>The model problem: a quadratic valley</H>
          <Formula
            plain="A bowl with curvature h₁ along θ₁ and h₂ along θ₂. The gradient in each direction is curvature times position, so steep directions (large h) produce large gradients."
            tex={t`J(\theta) = \tfrac12\,\theta^\top\begin{pmatrix}h_1 & 0\\ 0 & h_2\end{pmatrix}\theta = \tfrac12(h_1\theta_1^2 + h_2\theta_2^2), \qquad \nabla_\theta J = \begin{pmatrix}h_1\theta_1\\ h_2\theta_2\end{pmatrix}`}
            symbols={[
              [t`h_1, h_2`, "curvature in each direction. h₂ ≫ h₁ means the valley is steep across (θ₂) and flat along (θ₁)."],
              [t`h_2\theta_2`, "large → GD overshoots across the valley"],
              [t`h_1\theta_1`, "small → GD crawls along the valley"],
            ]}
          />
          <Formula
            plain="Keep a running velocity: at each step the old velocity shrinks by the factor ρ and the new gradient pushes it downhill; then the parameters move by the velocity. Consistent directions build up speed, zig-zags cancel out."
            title="Momentum update"
            sheet="yes"
            tex={t`\mathbf v_{t+1} = \rho\,\mathbf v_t - \eta\,\nabla_\theta J(\theta_t), \qquad \theta_{t+1} = \theta_t + \mathbf v_{t+1}`}
            symbols={[
              [t`\mathbf v`, <>the "velocity": the step we actually take. Starts at <M>{t`\mathbf v_0 = 0`}</M>.</>],
              [t`\rho \in [0,1)`, "momentum term: how much of the previous step to keep. ρ = 0 gives plain GD."],
              [t`\rho\,\mathbf v_t`, "the memory: an exponentially decaying average of past gradients"],
            ]}
          />
          <Callout kind="intuition">
            Gradients that <strong>agree</strong> across steps (along the valley) add up and get <strong>amplified</strong>. Gradients
            that <strong>flip sign</strong> each step (across the valley) <strong>cancel</strong>. Physics view: GD is a massless ball in
            honey; momentum gives the ball mass (inertia).
          </Callout>
          <H>Visual</H>
          <ValleyWidget />
          <H>Worked example</H>
          <Steps
            title="GD vs momentum on h = (1, 10), θ₀ = (4, 1), η = 0.15, ρ = 0.5"
            steps={[
              { title: "Step 1 gradient", body: <MB>{t`\nabla J(\theta_0) = (1\cdot4,\ 10\cdot1) = (4,\ 10)`}</MB> },
              { title: "Step 1 (both methods agree, since v₀ = 0)", body: <MB>{t`\mathbf v_1 = 0.5\cdot\mathbf 0 - 0.15\,(4,10) = (-0.6,-1.5) \;\Rightarrow\; \theta_1 = (3.4,\ -0.5)`}</MB> },
              { title: "Step 2 gradient", body: <MB>{t`\nabla J(\theta_1) = (3.4,\ -5)`}</MB> },
              { title: "Step 2, plain GD", body: <MB>{t`\theta_2 = (3.4,-0.5) - 0.15\,(3.4,-5) = (2.89,\ 0.25)`}</MB> },
              { title: "Step 2, momentum", body: <MB>{t`\mathbf v_2 = 0.5(-0.6,-1.5) - 0.15(3.4,-5) = (-0.81,\ 0) \;\Rightarrow\; \theta_2 = (2.59,\ -0.5)`}</MB> },
              { title: "What happened", body: <p>Across the valley (θ₂) the old step −1.5·0.5 and the new step +0.75 <strong>cancelled</strong> to 0. Along the valley (θ₁) they <strong>added up</strong>: −0.81 instead of GD's −0.51. Momentum moved further towards the minimum.</p> },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
h = np.array([1.0, 10.0])
grad = lambda th: h * th              # gradient of 0.5 * sum(h * th**2)
eta, rho = 0.15, 0.5

theta_gd = np.array([4.0, 1.0])
theta_m, v = np.array([4.0, 1.0]), np.zeros(2)
for t in range(3):
    theta_gd = theta_gd - eta * grad(theta_gd)
    v = rho * v - eta * grad(theta_m)
    theta_m = theta_m + v
    print(f"step {t+1}: GD {theta_gd}   momentum {theta_m}")
`}
          />
          <Callout kind="note">
            Cost: plain GD stores one copy of θ; momentum stores two (θ and v). That's the memory price.
          </Callout>
          <Slides lecture={2} pages="31–36" />
        </>
      ),
    },
    {
      id: "sgd",
      title: "SGD, minibatches and epochs",
      summary: "Estimate the gradient from a small random batch: many cheap, noisy steps instead of one exact step.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The full loss averages over all N examples, so one exact gradient step must look at the whole dataset. With
            millions of examples that's far too slow. <strong>Stochastic gradient descent (SGD)</strong> estimates the gradient
            from a small random subset, a <strong>minibatch</strong>, and takes a step after each one.
          </p>
          <H>The formula, decoded</H>
          <Formula
            plain="The full loss is the average of the per-example losses. A minibatch loss averages only B randomly chosen examples, so its gradient is a cheap, noisy estimate of the full gradient."
            title="Minibatch loss and stochastic gradient"
            sheet="yes"
            tex={t`J_{\mathcal D}(\theta) = \frac1N\sum_{n=1}^N J_n(\theta), \qquad J_{\mathcal B}(\theta) = \frac1B\sum_{n\in\mathcal B} J_n(\theta), \qquad \mathbf g_{\mathcal B} = \nabla_\theta J_{\mathcal B}(\theta) = \frac1B\sum_{n\in\mathcal B}\nabla_\theta J_n(\theta)`}
            symbols={[
              [t`J_n(\theta)`, "loss of a single example n (e.g. (ŷₙ − yₙ)²)"],
              [t`J_{\mathcal D}`, "loss on the whole dataset D (what we really want to minimize)"],
              [t`\mathcal B \subset \mathcal D`, "a random minibatch of B examples"],
              [t`\mathbf g_{\mathcal B}`, <>the <strong>stochastic gradient</strong>: an <em>unbiased estimate</em> of the true gradient (correct on average, but noisy)</>],
            ]}
          />
          <MB>{t`\text{SGD:}\quad \theta_{t+1} = \theta_t - \eta\,\mathbf g_{\mathcal B_t}(\theta_t)`}</MB>
          <H>Epochs, steps and batches</H>
          <p>
            An <strong>epoch</strong> is one full pass through the data. Each epoch: shuffle the data, cut it into
            minibatches, take one step per minibatch. So:
          </p>
          <MB>{t`\text{steps per epoch} = \lceil N / B \rceil`}</MB>
          <Steps
            title="Example: N = 8, B = 4"
            steps={[
              { title: "Epoch 1: shuffle", body: <p>order: x₃, x₇, x₁, x₆ | x₈, x₂, x₅, x₄</p> },
              { title: "Step 1 and step 2", body: <p>Step 1 uses the gradient averaged over {"{x₃, x₇, x₁, x₆}"}; step 2 over {"{x₈, x₂, x₅, x₄}"}.</p> },
              { title: "Epoch 2: shuffle again", body: <p>New order, e.g. x₅, x₂, x₈, x₁ | x₄, x₇, x₃, x₆ → steps 3 and 4.</p> },
              { title: "Count", body: <p>2 epochs × (8/4) = 4 parameter updates. Plain GD (B = N = 8) would have made only 2.</p> },
            ]}
          />
          <Table
            head={["Batch size", "Name", "Character"]}
            rows={[
              ["B = 1", "online SGD", "very noisy, cheap steps"],
              ["1 < B < N", "minibatch SGD", "the usual choice (often chosen to fill the GPU)"],
              ["B = N", "gradient descent", "exact, expensive steps"],
            ]}
          />
          <H>Visual: the noise floor</H>
          <SGDWidget />
          <Callout kind="intuition">
            <M>{t`\mathbf g_{\mathcal B} = \underbrace{\nabla J_{\mathcal D}}_{\text{signal}} + \text{noise}`}</M>. Far from the
            minimum the signal dominates and SGD makes fast progress. Near the minimum the true gradient becomes tiny, the
            noise dominates, and the loss stops improving: the <strong>noise floor</strong>. Fix: shrink η over time
            (<strong>learning-rate scheduling</strong>), e.g. exponential <M>{t`\eta_t = \eta_0\gamma^t`}</M>, linear decay to zero, or step-wise drops.
          </Callout>
          <H>Math → code</H>
          <PyCell
            code={py`
import numpy as np
rng = np.random.default_rng(0)
N, B = 8, 4
x = np.arange(1, N + 1, dtype=float)
y = 2 * x + 1
w, b, lr = 0.0, 0.0, 0.01

for epoch in range(2):
    perm = rng.permutation(N)                 # shuffle
    for start in range(0, N, B):
        idx = perm[start:start + B]           # one minibatch
        r = (w * x[idx] + b) - y[idx]
        grad_w = 2 * (r * x[idx]).mean()      # mean over the BATCH, not all N
        grad_b = 2 * r.mean()
        w, b = w - lr * grad_w, b - lr * grad_b
        print(f"epoch {epoch}, batch {idx}, w={w:.3f}, b={b:.3f}")
`}
          />
          <Callout kind="mistake">
            "Epoch" ≠ "step". With N = 60,000 and B = 100, one epoch has 600 steps. Exam questions often ask you to count updates.
          </Callout>
          <FromLab to="labs/debugging" label="Debugging neural networks">
            In PyTorch, minibatches come from a <code>DataLoader(dataset, batch_size, shuffle=True)</code>, which reshuffles
            every epoch. When N isn't a multiple of B, the last minibatch is smaller: 60,000 = 458·131 + 2. With batch norm that
            tiny batch causes a loss spike every epoch; <code>drop_last=True</code> on the training loader fixes it.
          </FromLab>
          <Slides lecture={2} pages="37–43; Lecture 3 page 4" />
        </>
      ),
    },
    {
      id: "adam",
      title: "Adaptive step sizes and Adam",
      summary: "Give every parameter its own learning rate, scaled by how big its gradients usually are.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            GD and SGD use one η for all parameters. But in a narrow valley one direction needs small steps and another
            needs big ones. <strong>Adaptive methods</strong> keep a running average of each parameter's squared gradient and
            divide by its square root: big gradients → smaller steps, small gradients → bigger steps. <strong>Adam</strong>
            combines this with momentum and is the default optimizer today.
          </p>
          <H>Step 1: adaptive step sizes (RMSProp-style)</H>
          <Formula
            plain="Keep a running average of each parameter's squared gradient, and divide that parameter's step by its square root: parameters with large gradients take smaller steps, parameters with small gradients take larger ones."
            sheet="no"
            tex={t`\mathbf s_{t+1} = \beta_2\,\mathbf s_t + (1-\beta_2)\,\mathbf g_t\odot\mathbf g_t, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\epsilon + \sqrt{\mathbf s_{t+1}}}\odot\mathbf g_t`}
            symbols={[
              [t`\mathbf g_t`, "the (minibatch) gradient at step t"],
              [t`\odot`, <>element-wise multiplication (NumPy <code>*</code>). <M>{t`\mathbf g\odot\mathbf g`}</M> squares each entry.</>],
              [t`\mathbf s`, "running average of squared gradients (the 'second moment'), one number per parameter"],
              [t`\beta_2`, "how slowly s forgets (0.999 = long memory)"],
              [t`\sqrt{\mathbf s}`, "typical gradient size per parameter; the division is also element-wise"],
              [t`\epsilon`, "tiny constant (10⁻⁸) to avoid dividing by zero"],
            ]}
          />
          <H>Step 2: Adam = momentum + adaptive step sizes</H>
          <Formula
            plain="Momentum (a running average of the gradients: the direction) combined with RMSProp (a running average of the squared gradients: the scale). Step in the averaged direction, scaled separately for each parameter."
            sheet="no"
            tex={t`\begin{aligned}\mathbf v_{t+1} &= \beta_1\mathbf v_t + (1-\beta_1)\,\mathbf g_t &&\text{(1st moment: direction)}\\ \mathbf s_{t+1} &= \beta_2\mathbf s_t + (1-\beta_2)\,\mathbf g_t\odot\mathbf g_t &&\text{(2nd moment: scale)}\\ \theta_{t+1} &= \theta_t - \frac{\eta}{\epsilon+\sqrt{\mathbf s_{t+1}}}\odot\mathbf v_{t+1}\end{aligned}`}
            symbols={[
              [t`\mathbf v`, "moving average of gradients (like momentum): which way to go"],
              [t`\mathbf s`, "moving average of squared gradients: how much to scale each parameter's step"],
              [t`\beta_1, \beta_2`, "decay rates; defaults 0.9 and 0.999"],
              [t`\eta`, <>default <M>{t`10^{-3}`}</M></>],
            ]}
          />
          <Callout kind="note">
            The full Adam algorithm also applies a "bias correction" to v and s. The slides omit it, and so do we.
          </Callout>
          <H>Worked example</H>
          <Steps
            title="One Adam step with g = (0.2, −4.0), v₀ = s₀ = 0, η = 0.1, β₁ = 0.9, β₂ = 0.999"
            steps={[
              { title: "First moment", body: <MB>{t`\mathbf v_1 = 0.9\cdot\mathbf 0 + 0.1\cdot(0.2,-4.0) = (0.02,\ -0.4)`}</MB> },
              { title: "Second moment (square each entry first)", body: <MB>{t`\mathbf g\odot\mathbf g = (0.04,\ 16),\quad \mathbf s_1 = 0.001\cdot(0.04, 16) = (0.00004,\ 0.016)`}</MB> },
              { title: "Square root, per entry", body: <MB>{t`\sqrt{\mathbf s_1} \approx (0.00632,\ 0.1265)`}</MB> },
              { title: "Step, per entry", body: <MB>{t`\frac{0.1}{\sqrt{\mathbf s_1}}\odot\mathbf v_1 \approx \left(\frac{0.1\cdot0.02}{0.00632},\ \frac{0.1\cdot(-0.4)}{0.1265}\right) \approx (0.316,\ -0.316)`}</MB> },
              { title: "What to notice", body: <p>The gradients differed by a factor of 20, yet both parameters moved by the <strong>same</strong> amount. Adam normalizes away the gradient's scale: each parameter effectively gets its own learning rate.</p> },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
g = np.array([0.2, -4.0])
v, s = np.zeros(2), np.zeros(2)
eta, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8

v = beta1 * v + (1 - beta1) * g
s = beta2 * s + (1 - beta2) * g * g        # g * g is element-wise (⊙)
step = eta / (eps + np.sqrt(s)) * v
print("v    =", v)
print("s    =", s)
print("step =", step)
`}
          />
          <Callout kind="exam">
            Know what each of v and s does and which hyperparameter belongs to which: β₁ ↔ v (direction, momentum), β₂ ↔ s
            (scale). All operations are element-wise. Adam uses the <em>same minibatch gradient</em> as SGD; only the update rule
            differs.
          </Callout>
          <Slides lecture={2} pages="44–47" />
        </>
      ),
    },
    {
      id: "hyperparams",
      title: "Hyperparameters & the validation set",
      summary: "Settings you choose by hand, and the third dataset you need to choose them honestly.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The model <em>learns</em> its parameters θ. Everything else you have to pick: those are the <strong>hyperparameters</strong>:
            learning rate η, batch size B, number of epochs, initialization θ₀, momentum ρ, schedule γ, Adam's β₁, β₂, ε,
            polynomial degree, regularization λ…
          </p>
          <H>Which data do we tune on?</H>
          <Table
            head={["Choose by…", "Problem"]}
            rows={[
              ["training performance", "rewards overfitting: the most complex model always wins"],
              ["test performance", "test data leaks into our choices, so our final estimate of generalization becomes too optimistic"],
              ["validation performance ✓", "a third, separate set used only for choosing hyperparameters"],
            ]}
          />
          <Table
            head={["Set", "Typical share", "Used for"]}
            rows={[
              ["training", "60–70%", "fitting the parameters θ"],
              ["validation", "10–20%", "selecting hyperparameters"],
              ["test", "20%", "estimating final performance — used once, at the very end"],
            ]}
          />
          <H>Early stopping</H>
          <p>
            "How many epochs?" is a hyperparameter too. Too few → underfitting; too many → overfitting. <strong>Early
            stopping</strong>: track the validation loss every epoch and stop (keep the best θ) once it starts going up.
          </p>
          <PyCell
            code={py`
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
def make(n):
    x = rng.uniform(0, 1, n)
    return x, np.sin(2 * np.pi * x) + 0.3 * rng.normal(size=n)
x_tr, y_tr = make(15)
x_va, y_va = make(50)
feats = lambda x: np.stack([x**m for m in range(1, 10)] + [np.ones_like(x)], axis=1)  # degree 9
P_tr, P_va = feats(x_tr), feats(x_va)

theta, lr = np.zeros(10), 0.3
train_l, val_l = [], []
for epoch in range(20000):
    r = P_tr @ theta - y_tr
    theta -= lr * 2 * P_tr.T @ r / len(r)
    if epoch % 100 == 0:
        train_l.append((r**2).mean())
        val_l.append(((P_va @ theta - y_va)**2).mean())

best = int(np.argmin(val_l))
print(f"best validation loss at epoch {best * 100}: {val_l[best]:.3f}")
plt.plot(np.arange(len(train_l)) * 100, train_l, label="training")
plt.plot(np.arange(len(val_l)) * 100, val_l, label="validation")
plt.axvline(best * 100, ls="--", c="gray", label="stop here")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.yscale("log")
`}
          />
          <H>Hyperparameter search</H>
          <p>
            Loop: pick hyperparameters → train θ on the training set → evaluate on validation → repeat until good → one final
            check on the test set. Strategies: manual, <strong>grid search</strong>, <strong>random search</strong>,
            Bayesian optimization. It's expensive because every try retrains the model.
          </p>
          <Callout kind="exam">
            Classic question: "Why not tune on the test set?" Answer: the test set must stay unseen to give an honest estimate of
            generalization. Tuning on it leaks information and overestimates performance.
          </Callout>
          <Slides lecture={2} pages="50–54" />
        </>
      ),
    },
  ],
};
