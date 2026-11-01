import { Callout, FromLab, Formula, H, M, MB, MathCode, Slides, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { DotProductWidget, LogisticWidget, LossBowlWidget, SigmoidWidget, SoftmaxWidget } from "../widgets/l3";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

export const lecture3: Lecture = {
  id: "l3",
  number: 3,
  title: "Linear models",
  sections: [
    {
      id: "vector-form",
      title: "Linear regression in vector form",
      summary: "Append a 1 to the input so the bias becomes a weight: ŷ = θᵀx̃ for any number of features.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            With one feature, <M>{t`\hat y = wx + b`}</M>. With D features, <M>{t`\hat y = w_1x_1 + \dots + w_Dx_D + b`}</M>. Writing
            both as a single <strong>dot product</strong> lets us use one formula for any D, and fast matrix code (NumPy's
            <code>@</code>) for all examples at once. Lecture 3's notation is all built on this trick.
          </p>
          <H>The formula, decoded</H>
          <Formula
            tex={t`\tilde{\mathbf x} = \begin{pmatrix}x_1\\\vdots\\x_D\\1\end{pmatrix},\quad \theta = \begin{pmatrix}w_1\\\vdots\\w_D\\b\end{pmatrix}\in\mathbb R^{D+1},\qquad \hat y = \theta^\top\tilde{\mathbf x} = \sum_{d=1}^D w_dx_d + b\cdot 1`}
            symbols={[
              [t`\tilde{\mathbf x}`, <>"x-tilde": the <strong>bias-augmented</strong> input, x with a 1 appended at the end</>],
              [t`\theta`, "all parameters stacked: the weights, then the bias last"],
              [t`\theta^\top\tilde{\mathbf x}`, "dot product: multiply entry by entry and add up. The 1 × b term gives the bias."],
              [t`D`, "number of features; θ has P = D + 1 entries"],
            ]}
          />
          <Steps
            title="Example: x = (petal length) = 3.0, θ = (w, b) = (0.3, 0.1)"
            steps={[
              { title: "Augment", body: <MB>{t`\tilde{\mathbf x} = (3.0,\ 1)^\top`}</MB> },
              { title: "Dot product", body: <MB>{t`\theta^\top\tilde{\mathbf x} = 0.3\cdot3.0 + 0.1\cdot1 = 1.0`}</MB> },
              { title: "Same as the old form", body: <MB>{t`wx + b = 0.3\cdot3 + 0.1 = 1.0 \;\checkmark`}</MB> },
            ]}
          />
          <H>Why is it called "linear"?</H>
          <p>
            Strictly, <M>{t`wx+b`}</M> is <em>affine</em> in x, not linear: <M>{t`w(x_1+x_2)+b \neq (wx_1+b)+(wx_2+b)`}</M>. The model
            is linear in two senses that matter:
          </p>
          <ul>
            <li>linear in the augmented input <M>{t`\tilde{\mathbf x}`}</M>, and</li>
            <li><strong>linear in the parameters θ</strong>, which is what makes training easy (convex loss, closed form).</li>
          </ul>
          <Think q="Is ŷ = w₂w₁x + b a linear model in the sense above?">
            It's affine in x, but <strong>not linear in θ = (w₁, w₂, b)</strong>, because two parameters are multiplied together.
            (It can't represent anything more than wx + b either; this is the same reason neural nets need non-linear activations.)
          </Think>
          <H>The dot product measures alignment</H>
          <MB>{t`\mathbf a^\top\mathbf b = \sum_{m=1}^M a_mb_m = \|\mathbf a\|_2\,\|\mathbf b\|_2\cos\angle(\mathbf a,\mathbf b), \qquad \|\mathbf a\|_2 = \sqrt{\mathbf a^\top\mathbf a}`}</MB>
          <DotProductWidget />
          <p>
            Slide examples with <M>{t`\mathbf b = (3,1)`}</M>: <M>{t`\mathbf a = (1,2)`}</M> gives 5 (positive),
            <M>{t`(-1,3)`}</M> gives 0 (orthogonal), <M>{t`(-2,1)`}</M> gives −5 (negative).
          </p>
          <Callout kind="intuition">
            Geometrically, the linear model is a <strong>hyperplane</strong>: a line for D = 1, a plane for D = 2. Its normal vector
            in <M>{t`(\mathbf x, \hat y)`}</M>-space is <M>{t`(\mathbf w, -1)`}</M>.
          </Callout>
          <H>Math → code</H>
          <PyCell
            code={py`
import numpy as np
x_train = np.array([1.0, 3.0, 5.0])                         # petal lengths, shape (N,)
X_tilde = np.column_stack([x_train, np.ones_like(x_train)])  # append a column of 1s → (N, D+1)
theta = np.array([0.3, 0.1])                                 # (w, b)
print(X_tilde)
print("y_hat =", X_tilde @ theta)                            # all N predictions at once
`}
          />
          <Slides lecture={3} pages="10–14" />
        </>
      ),
    },
    {
      id: "one-step",
      title: "One training step in vector form",
      summary: "Predictions, residuals, MSE, gradient 2/N·X̃ᵀr and one GD update, with the slide's numbers.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            This is the Lecture 2 training loop rewritten with matrices. Instead of a <code>for</code> loop over examples, one
            matrix multiplication handles all N at once. You must be able to do this by hand with small numbers, and to read
            the NumPy version.
          </p>
          <H>The formulas, decoded</H>
          <Formula
            tex={t`\tilde{\mathbf X}_{\text{train}} = \begin{pmatrix}\tilde{\mathbf x}_1^\top\\\vdots\\\tilde{\mathbf x}_N^\top\end{pmatrix}\in\mathbb R^{N\times(D+1)},\quad \hat{\mathbf y} = \tilde{\mathbf X}\theta,\quad \mathbf r = \hat{\mathbf y}-\mathbf y,\quad J = \tfrac1N\mathbf r^\top\mathbf r,\quad \nabla_\theta J = \tfrac2N\tilde{\mathbf X}^\top\mathbf r`}
            symbols={[
              [t`\tilde{\mathbf X}`, "design matrix: one augmented example per ROW"],
              [t`\hat{\mathbf y}\in\mathbb R^N`, "all predictions; row n is θᵀx̃ₙ"],
              [t`\mathbf r\in\mathbb R^N`, <>residual vector (errors), <M>{t`r_n = \hat y_n - y_n`}</M></>],
              [t`\mathbf r^\top\mathbf r`, <>sum of squared errors <M>{t`\sum_n r_n^2`}</M></>],
              [t`\tilde{\mathbf X}^\top\mathbf r`, <>shape (D+1, N)·(N,) = (D+1,): same shape as θ ✓. It equals <M>{t`\sum_n r_n\tilde{\mathbf x}_n`}</M>: each example's input weighted by its error.</>],
            ]}
          />
          <H>Worked example (slides 16–18)</H>
          <p>Data <M>{t`x = (1,3,5)`}</M>, <M>{t`\mathbf y = (0.2, 1.1, 1.7)`}</M>, start <M>{t`\theta_0 = (0.3, 0.1)`}</M>, <M>{t`\eta = 0.05`}</M>.</p>
          <Steps
            steps={[
              { title: "1. Predictions", body: <MB>{t`\hat{\mathbf y} = \begin{pmatrix}1&1\\3&1\\5&1\end{pmatrix}\begin{pmatrix}0.3\\0.1\end{pmatrix} = \begin{pmatrix}0.3+0.1\\0.9+0.1\\1.5+0.1\end{pmatrix} = \begin{pmatrix}0.4\\1.0\\1.6\end{pmatrix}`}</MB> },
              { title: "2. Residuals", body: <MB>{t`\mathbf r = \hat{\mathbf y} - \mathbf y = (0.4-0.2,\ 1.0-1.1,\ 1.6-1.7) = (0.2,\ -0.1,\ -0.1)`}</MB> },
              { title: "3. Loss", body: <MB>{t`J = \tfrac13(0.2^2 + 0.1^2 + 0.1^2) = \tfrac13(0.04+0.01+0.01) = 0.02`}</MB> },
              { title: "4. Gradient (transpose turns columns into rows)", body: <MB>{t`\nabla_\theta J = \tfrac23\begin{pmatrix}1&3&5\\1&1&1\end{pmatrix}\begin{pmatrix}0.2\\-0.1\\-0.1\end{pmatrix} = \tfrac23\begin{pmatrix}0.2-0.3-0.5\\0.2-0.1-0.1\end{pmatrix} = \tfrac23\begin{pmatrix}-0.6\\0\end{pmatrix} = \begin{pmatrix}-0.4\\0\end{pmatrix}`}</MB> },
              { title: "5. Update", body: <MB>{t`\theta_1 = \begin{pmatrix}0.3\\0.1\end{pmatrix} - 0.05\begin{pmatrix}-0.4\\0\end{pmatrix} = \begin{pmatrix}0.32\\0.1\end{pmatrix}`}</MB> },
              { title: "6. Check: did the loss go down?", body: <p><M>{t`J(\theta_1) \approx 0.017 < 0.02 = J(\theta_0)`}</M> ✓. After 1000 steps: <M>{t`\theta \approx (0.375, -0.125)`}</M>, <M>{t`J \approx 0.005`}</M>.</p> },
            ]}
          />
          <Think q="The gradient for b came out exactly 0. What does that tell you?">
            The residuals sum to zero (0.2 − 0.1 − 0.1 = 0): on average the predictions are neither too high nor too low, so
            shifting the line up or down wouldn't help. Only the slope w needs to change (it's negative → increase w).
          </Think>
          <H>Math → code</H>
          <MathCode
            rows={[
              [t`\hat{\mathbf y}=\tilde{\mathbf X}\theta`, "y_hat = X @ theta", "(N, D+1) @ (D+1,) → (N,)"],
              [t`\mathbf r = \hat{\mathbf y}-\mathbf y`, "r = y_hat - y", "(N,)"],
              [t`J=\frac1N\mathbf r^\top\mathbf r`, "J = r @ r / N", "scalar"],
              [t`\nabla J = \frac2N\tilde{\mathbf X}^\top\mathbf r`, "grad = 2 / N * X.T @ r", "(D+1, N) @ (N,) → (D+1,)"],
              [t`\theta\leftarrow\theta-\eta\nabla J`, "theta = theta - lr * grad", ""],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
X = np.array([[1., 1.], [3., 1.], [5., 1.]])   # X_tilde, (N, D+1)
y = np.array([0.2, 1.1, 1.7])
theta, lr, N = np.array([0.3, 0.1]), 0.05, len(y)

y_hat = X @ theta
r = y_hat - y
J = r @ r / N
grad = 2 / N * X.T @ r
theta = theta - lr * grad
print("y_hat", y_hat, " r", r, " J", round(J, 4))
print("grad", grad.round(4), " theta_1", theta, " J_1", round(((X @ theta - y)**2).mean(), 4))
`}
          />
          <Callout kind="exam">
            "Try at home" from the slides: use <em>two</em> features, X̃ rows (4.6, 1.0, 1), (5.1, 3.0, 1), (6.7, 5.0, 1) and
            θ = (0.1, 0.2, 0.1). Edit the cell above: you should get ŷ ≈ (0.76, 1.21, 1.77), J ≈ 0.11, ∇J ≈ (2.4, 0.8, 0.5), and
            with η = 0.01, θ₁ ≈ (0.076, 0.192, 0.095).
          </Callout>
          <Callout kind="mistake">
            <code>X.T @ r</code>, not <code>X @ r</code> (shapes wouldn't match) and not <code>r @ X.T</code>. Check shapes: the
            gradient must have the same shape as θ.
          </Callout>
          <FromLab to="labs/from-scratch" label="Linear & logistic regression from scratch">
            Lab 3, exercise 2.1: for one example the prediction is <code>x_tilde.T @ theta</code>, but for a whole data matrix
            it's <code>X_tilde @ theta</code>, since <code>X_tilde.T</code> has shape (D+1, N) and can't go on the left. Check
            shapes to tell them apart: (N, D+1) @ (D+1,) → (N,).
          </FromLab>
          <Slides lecture={3} pages="16–18" />
        </>
      ),
    },
    {
      id: "closed-form",
      title: "Closed form, Hessian & convexity",
      summary: "The MSE loss is a quadratic bowl: solve ∇J = 0 directly, and use the Hessian to understand its shape.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            For linear regression we don't even need gradient descent: the loss is a perfect bowl (a quadratic), so we can
            jump straight to the bottom by solving <M>{t`\nabla J = 0`}</M>. The bowl's shape (its <strong>Hessian</strong>) also
            explains when GD is fast or slow.
          </p>
          <H>The loss is quadratic</H>
          <MB>{t`J(\theta) = \tfrac1N(\tilde{\mathbf X}\theta-\mathbf y)^\top(\tilde{\mathbf X}\theta-\mathbf y) = \tfrac1N\big(\theta^\top\tilde{\mathbf X}^\top\tilde{\mathbf X}\theta - 2\mathbf y^\top\tilde{\mathbf X}\theta + \mathbf y^\top\mathbf y\big)`}</MB>
          <H>The analytical solution (normal equations)</H>
          <Steps
            steps={[
              { title: "Set the gradient to zero", body: <MB>{t`\tfrac2N\tilde{\mathbf X}^\top(\tilde{\mathbf X}\theta - \mathbf y) = \mathbf 0`}</MB> },
              { title: "Multiply out, drop 2/N", body: <MB>{t`\tilde{\mathbf X}^\top\tilde{\mathbf X}\,\theta = \tilde{\mathbf X}^\top\mathbf y`}</MB> },
              { title: "Solve (if X̃ᵀX̃ is invertible)", body: <MB>{t`\theta^\star = (\tilde{\mathbf X}^\top\tilde{\mathbf X})^{-1}\tilde{\mathbf X}^\top\mathbf y`}</MB> },
              { title: "Numbers: X̃ᵀX̃", body: <MB>{t`\begin{pmatrix}1&3&5\\1&1&1\end{pmatrix}\begin{pmatrix}1&1\\3&1\\5&1\end{pmatrix} = \begin{pmatrix}1+9+25 & 1+3+5\\ 1+3+5 & 3\end{pmatrix} = \begin{pmatrix}35&9\\9&3\end{pmatrix}`}</MB> },
              { title: "Numbers: X̃ᵀy", body: <MB>{t`\begin{pmatrix}0.2+3.3+8.5\\0.2+1.1+1.7\end{pmatrix} = \begin{pmatrix}12\\3\end{pmatrix}`}</MB> },
              { title: "2×2 inverse: swap diagonal, negate off-diagonal, divide by determinant", body: <MB>{t`\det = 35\cdot3 - 9\cdot9 = 24,\qquad (\tilde{\mathbf X}^\top\tilde{\mathbf X})^{-1} = \tfrac1{24}\begin{pmatrix}3&-9\\-9&35\end{pmatrix}`}</MB> },
              { title: "Multiply", body: <MB>{t`\theta^\star = \tfrac1{24}\begin{pmatrix}3\cdot12 - 9\cdot3\\ -9\cdot12+35\cdot3\end{pmatrix} = \tfrac1{24}\begin{pmatrix}9\\-3\end{pmatrix} = \begin{pmatrix}0.375\\-0.125\end{pmatrix},\quad J(\theta^\star)=0.005`}</MB> },
            ]}
          />
          <p>That's exactly where 1000 GD steps ended up — but in one shot.</p>
          <H>The Hessian: second derivatives</H>
          <Formula
            tex={t`\mathbf H = \nabla^2_\theta J = \tfrac2N\tilde{\mathbf X}^\top\tilde{\mathbf X}\in\mathbb R^{P\times P},\qquad H_{ij} = \frac{\partial^2J}{\partial\theta_i\,\partial\theta_j}`}
            symbols={[
              [t`\mathbf H`, "Hessian: matrix of all second derivatives — the curvature of the loss"],
              [t`\nabla_\theta J = \tfrac2N\tilde{\mathbf X}^\top\tilde{\mathbf X}\theta - \tfrac2N\tilde{\mathbf X}^\top\mathbf y`, "differentiate this once more w.r.t. θ to get H"],
              ["\\text{constant}", "for linear regression H doesn't depend on θ: the bowl has one fixed shape"],
            ]}
          />
          <p>Running example: <M>{t`\mathbf H = \tfrac23\begin{pmatrix}35&9\\9&3\end{pmatrix}`}</M>.</p>
          <H>Convexity</H>
          <p>A function is <strong>convex</strong> if it only bends upward. Then any local minimum is the global minimum: gradient zero means done. Two equivalent pictures:</p>
          <MB>{t`\underbrace{J(t\theta + (1-t)\theta') \le tJ(\theta) + (1-t)J(\theta')}_{\text{every chord lies above the graph}}\qquad \underbrace{J(\theta')\ge J(\theta) + \nabla J(\theta)^\top(\theta'-\theta)}_{\text{every tangent lies below the graph}}`}</MB>
          <p>
            And with the Hessian: convex ⇔ <strong>H is positive semi-definite (PSD)</strong>: <M>{t`\mathbf v^\top\mathbf H\mathbf v \ge 0`}</M> for all v, θ.
            For linear regression:
          </p>
          <MB>{t`\mathbf v^\top\tfrac2N\tilde{\mathbf X}^\top\tilde{\mathbf X}\mathbf v = \tfrac2N(\tilde{\mathbf X}\mathbf v)^\top(\tilde{\mathbf X}\mathbf v) = \tfrac2N\|\tilde{\mathbf X}\mathbf v\|_2^2 \ge 0 \quad\Rightarrow\quad \text{convex } \checkmark`}</MB>
          <H>Condition number</H>
          <p>
            The eigenvalues of H measure curvature along its principal directions. The <strong>condition number</strong>
            <M>{t`\kappa = \lambda_{\max}/\lambda_{\min}`}</M> says how elongated the bowl is. Larger κ → slower GD (the narrow-valley
            problem from Lecture 2). Pre-processing the data (e.g. centering features) can reduce κ.
          </p>
          <LossBowlWidget />
          <H>Math → code</H>
          <PyCell
            code={py`
import numpy as np
X = np.array([[1., 1.], [3., 1.], [5., 1.]])
y = np.array([0.2, 1.1, 1.7])
N = len(y)

XtX = X.T @ X
theta_star = np.linalg.inv(XtX) @ X.T @ y       # (X^T X)^{-1} X^T y
theta_star2 = np.linalg.solve(XtX, X.T @ y)      # same, numerically better
print("X^T X =\n", XtX)
print("theta* =", theta_star, theta_star2)

H = 2 / N * XtX
eig = np.linalg.eigvalsh(H)
print("Hessian eigenvalues:", eig, " kappa =", eig.max() / eig.min())
`}
          />
          <Callout kind="exam">
            Be able to: invert a 2×2 matrix by hand, explain why the linear-regression Hessian is PSD (it's a squared norm), and
            say what a large condition number means for GD.
          </Callout>
          <FromLab to="labs/from-scratch" label="Linear & logistic regression from scratch">
            The labs compute θ* with <code>np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y</code>, or better{" "}
            <code>np.linalg.solve(...)</code>, and compare it with scikit-learn's <code>LinearRegression</code>. Because{" "}
            <code>X_tilde</code> already contains the column of ones, you pass <code>fit_intercept=False</code>; otherwise
            scikit-learn adds a second bias of its own.
          </FromLab>
          <Slides lecture={3} pages="19–24" />
        </>
      ),
    },
    {
      id: "logistic",
      title: "Logistic regression",
      summary: "Squash the linear output through a sigmoid to get a probability for binary classification.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            For classification we want a probability, a number between 0 and 1, not an unbounded value. Logistic regression
            takes the linear model's output <M>{t`z = \theta^\top\tilde{\mathbf x}`}</M> and squashes it with the sigmoid. (Despite the
            name, it's a <em>classifier</em>.)
          </p>
          <H>The sigmoid</H>
          <Formula
            tex={t`\sigma(z) = \frac{1}{1+e^{-z}}\in(0,1), \qquad \frac{d\sigma}{dz} = \sigma(z)\,(1-\sigma(z))`}
            symbols={[
              [t`z`, <>the <strong>logit</strong>: the raw linear output <M>{t`\theta^\top\tilde{\mathbf x}`}</M>, any real number</>],
              [t`\sigma(0)=0.5`, "z > 0 → probability above ½; z < 0 → below ½"],
              [t`\sigma(z)(1-\sigma(z))`, "the derivative is expressed through σ itself — handy for the gradient"],
            ]}
          />
          <SigmoidWidget />
          <H>Model and loss</H>
          <Formula
            tex={t`\hat y_n = \sigma(\theta^\top\tilde{\mathbf x}_n),\qquad J(\theta) = -\frac1N\sum_{n=1}^N\Big[y_n\ln\hat y_n + (1-y_n)\ln(1-\hat y_n)\Big]`}
            symbols={[
              [t`y_n\in\{0,1\}`, "label: 0 = versicolor, 1 = virginica"],
              [t`\hat y_n`, "predicted probability of class 1"],
              ["J", <><strong>binary cross-entropy (BCE)</strong>: Lecture 1's CCE with K = 2</>],
              [t`y_n\ln\hat y_n`, "active only when y = 1: penalizes a small ŷ"],
              [t`(1-y_n)\ln(1-\hat y_n)`, "active only when y = 0: penalizes a large ŷ"],
            ]}
          />
          <H>Worked example (slide 28)</H>
          <Steps
            title="A versicolor (y = 0) with x̃ = (4.0, 1.3, 1), θ = (1, 1, −5)"
            steps={[
              { title: "Logit", body: <MB>{t`z = \theta^\top\tilde{\mathbf x} = 1\cdot4.0 + 1\cdot1.3 + (-5)\cdot1 = 0.3`}</MB> },
              { title: "Probability", body: <MB>{t`\hat y = \sigma(0.3) = \frac1{1+e^{-0.3}} = \frac1{1+0.741} \approx 0.57`}</MB> },
              { title: "Prediction", body: <p>0.57 &gt; 0.5 → predicts virginica. Wrong! (label is versicolor)</p> },
              { title: "Loss (y = 0, so only the second term survives)", body: <MB>{t`J_n = -0\cdot\ln(0.57) - (1-0)\ln(1-0.57) = -\ln(0.43)\approx 0.85`}</MB> },
            ]}
          />
          <H>The decision boundary</H>
          <p>
            Predict class 1 when <M>{t`\hat y > 0.5`}</M>. Since σ is increasing and <M>{t`\sigma(0)=0.5`}</M>:
            <M>{t`\ \sigma(\theta^\top\tilde{\mathbf x}) > 0.5 \iff \theta^\top\tilde{\mathbf x} > 0`}</M>. So the boundary is the
            hyperplane <M>{t`\theta^\top\tilde{\mathbf x} = 0`}</M>, with normal vector <M>{t`\mathbf w`}</M>. For θ = (1, 1, −5):
            the line <M>{t`x_1 + x_2 = 5`}</M>.
          </p>
          <LogisticWidget />
          <Callout kind="intuition">
            In linear regression the hyperplane <strong>fits</strong> the data; in logistic regression it <strong>separates</strong> it.
          </Callout>
          <PyCell
            code={py`
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
x_tilde = np.array([4.0, 1.3, 1.0])
theta = np.array([1.0, 1.0, -5.0])
y = 0

z = theta @ x_tilde
y_hat = sigmoid(z)
J_n = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
print(f"z = {z:.2f}, y_hat = {y_hat:.3f}, J_n = {J_n:.3f}")
`}
          />
          <Slides lecture={3} pages="27–30" />
        </>
      ),
    },
    {
      id: "logistic-train",
      title: "Training logistic regression",
      summary: "The gradient is again error × input. Convex, but no closed form, so we use gradient descent.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The beautiful result here: after the chain rule, the logistic regression gradient has the <strong>same shape</strong> as
            linear regression's — prediction minus label, times the input. Remember the structure, not just the formula.
          </p>
          <H>Deriving the gradient</H>
          <Steps
            title="Chain: θ → zₙ → ŷₙ → Jₙ"
            steps={[
              { title: "Loss w.r.t. prediction", body: <MB>{t`\frac{\partial J_n}{\partial\hat y_n} = -\frac{y_n}{\hat y_n} + \frac{1-y_n}{1-\hat y_n} = \frac{\hat y_n - y_n}{\hat y_n(1-\hat y_n)}`}</MB> },
              { title: "Prediction w.r.t. logit (sigmoid derivative)", body: <MB>{t`\frac{\partial\hat y_n}{\partial z_n} = \hat y_n(1-\hat y_n)`}</MB> },
              { title: "Logit w.r.t. parameters", body: <MB>{t`\frac{\partial z_n}{\partial\theta} = \tilde{\mathbf x}_n`}</MB> },
              { title: "Multiply: the ŷ(1−ŷ) terms cancel!", body: <MB>{t`\nabla_\theta J_n = \frac{\hat y_n - y_n}{\hat y_n(1-\hat y_n)}\cdot\hat y_n(1-\hat y_n)\cdot\tilde{\mathbf x}_n = (\hat y_n - y_n)\,\tilde{\mathbf x}_n`}</MB> },
              { title: "Average over all examples (vector form)", body: <MB>{t`\nabla_\theta J = \frac1N\sum_n(\hat y_n - y_n)\tilde{\mathbf x}_n = \frac1N\tilde{\mathbf X}^\top(\hat{\mathbf y} - \mathbf y)`}</MB> },
              { title: "Numbers for our flower", body: <MB>{t`\frac{\partial J_n}{\partial \hat y_n}\approx\frac{0.57}{0.57\cdot0.43}\approx2.35,\qquad \nabla_\theta J_n \approx (0.57-0)\begin{pmatrix}4.0\\1.3\\1\end{pmatrix}\approx\begin{pmatrix}2.30\\0.75\\0.57\end{pmatrix}`}</MB> },
            ]}
          />
          <H>Two gradients, one structure</H>
          <Table
            head={["", "Linear regression", "Logistic regression"]}
            rows={[
              ["prediction", <M>{t`\hat{\mathbf y} = \tilde{\mathbf X}\theta`}</M>, <M>{t`\hat{\mathbf y} = \sigma(\tilde{\mathbf X}\theta)`}</M>],
              ["gradient", <M>{t`\frac2N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)`}</M>, <M>{t`\frac1N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)`}</M>],
              ["closed form?", "yes", "no"],
              ["convex?", "yes", "yes"],
            ]}
          />
          <H>Why no closed form, and is it still convex?</H>
          <p>
            Setting <M>{t`\frac1N\tilde{\mathbf X}^\top(\sigma(\tilde{\mathbf X}\theta) - \mathbf y) = 0`}</M>: θ is trapped inside the
            non-linear σ, so we can't isolate it. → Gradient descent. The Hessian now depends on θ, but is still PSD:
          </p>
          <MB>{t`\mathbf H(\theta) = \tfrac1N\tilde{\mathbf X}^\top\mathbf S\tilde{\mathbf X},\quad \mathbf S = \mathrm{diag}\big(\hat y_n(1-\hat y_n)\big),\qquad \mathbf v^\top\mathbf H\mathbf v = \tfrac1N\sum_n\underbrace{\hat y_n(1-\hat y_n)}_{\ge0}(\tilde{\mathbf x}_n^\top\mathbf v)^2\ge0`}</MB>
          <H>Math → code (slide 35)</H>
          <MathCode
            rows={[
              [t`\mathbf z = \tilde{\mathbf X}\theta`, "z = X_train @ theta", "(N,)"],
              [t`\hat{\mathbf y} = \sigma(\mathbf z)`, "y_hat = 1 / (1 + np.exp(-z))", "element-wise"],
              [t`J_n = -y_n\ln\hat y_n - (1-y_n)\ln(1-\hat y_n)`, "J_ns = -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)", "(N,) per-example losses"],
              [t`J = \frac1N\sum_n J_n`, "loss = J_ns.mean()", ""],
              [t`\nabla J = \frac1N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)`, "grad = X_train.T @ error / N", "no factor 2 here!"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
# illustrative iris-like data: petal length, petal width, 1 ; y: 0 = versicolor, 1 = virginica
X_train = np.array([[4.0, 1.3, 1], [4.5, 1.5, 1], [4.7, 1.4, 1], [3.9, 1.1, 1], [5.0, 1.7, 1],
                    [6.0, 2.5, 1], [5.1, 1.9, 1], [5.9, 2.1, 1], [5.6, 1.8, 1], [5.0, 1.5, 1]])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

theta = np.array([1.0, 1.0, -5.0])
epochs, lr, N = 1000, 0.2, len(y_train)
for epoch in range(epochs):
    # forward pass
    z = X_train @ theta
    y_hat = 1 / (1 + np.exp(-z))
    error = y_hat - y_train
    J_ns = (-y_train * np.log(y_hat)
            - (1 - y_train) * np.log(1 - y_hat))
    loss = J_ns.mean()
    # gradient
    grad = X_train.T @ error / N
    # gradient descent update
    theta = theta - lr * grad
    if epoch in (0, 9, 99, 999):
        acc = ((y_hat > 0.5) == y_train).mean()
        print(f"epoch {epoch+1:4d}: loss {loss:.3f}, accuracy {acc:.2f}")
print("theta =", theta.round(2))
`}
          />
          <Callout kind="mistake">
            Linear regression's MSE gradient has a 2 (from the square); logistic regression's BCE gradient doesn't. Don't mix them up.
          </Callout>
          <Callout kind="note">
            On the real iris split in the slides, the test accuracy (0.96) was <em>higher</em> than training (0.93). That's chance:
            with only 25 test flowers, both numbers are noisy.
          </Callout>
          <FromLab to="labs/from-scratch" label="Linear & logistic regression from scratch">
            Three things the lab tutorials showed. First, the accuracy stops improving long before the loss does: once the
            boundary sits between the classes, the BCE keeps rewarding more confident probabilities. Second, on perfectly
            separable data the loss has no minimum, so the weights keep growing. Third, scikit-learn's{" "}
            <code>LogisticRegression()</code> only matches your gradient descent with <code>C=np.inf</code>, because its
            default adds an L2 penalty.
          </FromLab>
          <Slides lecture={3} pages="32–37" />
        </>
      ),
    },
    {
      id: "multiclass",
      title: "Multiclass logistic regression (softmax)",
      summary: "K linear models in parallel, turned into K probabilities by the softmax, trained with cross-entropy.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            With K &gt; 2 classes (setosa, versicolor, virginica) we need one score per class. We run K linear models side by
            side (a matrix multiply) and the <strong>softmax</strong> turns the K scores into a probability distribution.
          </p>
          <H>Parameters as a matrix</H>
          <Formula
            tex={t`\tilde{\mathbf W} = \begin{pmatrix}\theta_1^\top\\\vdots\\\theta_K^\top\end{pmatrix}\in\mathbb R^{K\times(D+1)},\qquad \mathbf z_n = \tilde{\mathbf W}\tilde{\mathbf x}_n = \begin{pmatrix}\theta_1^\top\tilde{\mathbf x}_n\\\vdots\\\theta_K^\top\tilde{\mathbf x}_n\end{pmatrix} = \mathbf W\mathbf x_n + \mathbf b`}
            symbols={[
              [t`\theta_k = (w_{k,1},\dots,w_{k,D},b_k)`, "the weights and bias of class k's linear model; one ROW of W̃"],
              [t`\mathbf z_n\in\mathbb R^K`, <>the K <strong>logits</strong> (pre-activations), one per class</>],
              [t`P = K(D+1)`, "total number of parameters (9 for K = 3, D = 2)"],
            ]}
          />
          <H>The softmax</H>
          <Formula
            tex={t`\hat y_{n,k} = \frac{e^{z_{n,k}}}{\sum_{k'=1}^K e^{z_{n,k'}}}\in(0,1),\qquad \sum_{k=1}^K\hat y_{n,k} = 1`}
            symbols={[
              [t`e^{z}`, "makes every score positive, and preserves the order"],
              [t`\sum_{k'} e^{z_{n,k'}}`, "normalizer: dividing by the total makes them sum to 1"],
              [t`\arg\max_k`, "the prediction: the class with the largest probability (= largest logit)"],
            ]}
          />
          <SoftmaxWidget />
          <H>Worked example (slides 39–43, 46)</H>
          <Steps
            title="Our versicolor x̃ = (4.0, 1.3, 1), y = (0, 1, 0)"
            steps={[
              { title: "Lay out θ = (1, 0, −3, 0, 1, −1, −1, 0, 2) as rows", body: <MB>{t`\tilde{\mathbf W} = \begin{pmatrix}1&0&-3\\0&1&-1\\-1&0&2\end{pmatrix}`}</MB> },
              { title: "Logits (each row · x̃)", body: <MB>{t`\mathbf z = \begin{pmatrix}4.0 + 0 - 3\\0 + 1.3 - 1\\-4.0 + 0 + 2\end{pmatrix} = \begin{pmatrix}1\\0.3\\-2\end{pmatrix}`}</MB> },
              { title: "Exponentiate", body: <MB>{t`(e^1, e^{0.3}, e^{-2}) \approx (2.72,\ 1.35,\ 0.14),\quad \text{sum} \approx 4.20`}</MB> },
              { title: "Normalize", body: <MB>{t`\hat{\mathbf y} \approx \tfrac1{4.20}(2.72, 1.35, 0.14) \approx (0.65,\ 0.32,\ 0.03)`}</MB> },
              { title: "Prediction", body: <p>argmax → setosa (0.65). Wrong: versicolor only gets 0.32.</p> },
              { title: "CCE loss: only the true class's term survives", body: <MB>{t`J_n = -\sum_k y_{n,k}\ln\hat y_{n,k} = -\ln 0.32 \approx 1.14`}</MB> },
              { title: "Gradient: error × input again (outer product)", body: <MB>{t`\nabla_{\tilde{\mathbf W}}J_n = (\hat{\mathbf y}_n - \mathbf y_n)\tilde{\mathbf x}_n^\top \approx \begin{pmatrix}0.65\\-0.68\\0.03\end{pmatrix}\begin{pmatrix}4.0&1.3&1\end{pmatrix} \approx \begin{pmatrix}2.59&0.84&0.65\\-2.72&-0.88&-0.68\\0.13&0.04&0.03\end{pmatrix}`}</MB> },
              { title: "Read the gradient", body: <p>Row 2 (versicolor) is negative → GD <em>increases</em> versicolor's weights (−η·negative = +). Row 1 (setosa) positive → GD decreases setosa's weights. Exactly what we want.</p> },
            ]}
          />
          <H>Vector form for all N examples</H>
          <MB>{t`\hat{\mathbf Y} = \mathrm{softmax}(\tilde{\mathbf X}\tilde{\mathbf W}^\top)\in\mathbb R^{N\times K}\ (\text{row-wise}),\qquad \nabla_{\tilde{\mathbf W}}J = \tfrac1N(\hat{\mathbf Y}-\mathbf Y)^\top\tilde{\mathbf X}\in\mathbb R^{K\times(D+1)}`}</MB>
          <MathCode
            rows={[
              [t`\mathbf Z = \tilde{\mathbf X}\tilde{\mathbf W}^\top`, "Z = X_train @ W_tilde.T", "(N, D+1)@(D+1, K) → (N, K)"],
              [t`e^{\mathbf Z}`, "E = np.exp(Z)", "(N, K)"],
              [t`\hat{\mathbf Y} = \text{softmax}`, "Y_hat = E / E.sum(axis=1, keepdims=True)", "divide each row by its own sum"],
              [t`J_n = -\sum_k Y_{nk}\ln\hat Y_{nk}`, "J_ns = -(Y_train * np.log(Y_hat)).sum(1)", "(N,)"],
              [t`\nabla J = \frac1N(\hat{\mathbf Y}-\mathbf Y)^\top\tilde{\mathbf X}`, "grad = error.T @ X_train / N", "(K, N)@(N, D+1) → (K, D+1) = W̃'s shape ✓"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
X_train = np.array([[4.0, 1.3, 1.0]])         # one flower, (N, D+1)
Y_train = np.array([[0, 1, 0]])               # one-hot: versicolor
W_tilde = np.array([[1, 0, -3], [0, 1, -1], [-1, 0, 2]], dtype=float)
N = len(Y_train)

Z = X_train @ W_tilde.T
E = np.exp(Z)
Y_hat = E / E.sum(axis=1, keepdims=True)
error = Y_hat - Y_train
J_ns = -(Y_train * np.log(Y_hat)).sum(1)
grad = error.T @ X_train / N
print("Z =", Z, "\nY_hat =", Y_hat.round(2), "\nloss =", J_ns.mean().round(2))
print("grad =\n", grad.round(2))
`}
          />
          <Think q="What happens if you forget keepdims=True in E.sum(axis=1, keepdims=True)?">
            <code>E.sum(axis=1)</code> has shape (N,), and broadcasting (N, K) / (N,) either fails or (when N = K) silently divides
            the wrong entries. <code>keepdims=True</code> keeps shape (N, 1) so each row is divided by its own sum.
          </Think>
          <Callout kind="intuition">
            On MNIST (28×28 images, D = 784, K = 10), each row of W is itself a 28×28 "template" image for its digit: positive
            weights are evidence for, negative against. Linear models are interpretable.
          </Callout>
          <FromLab to="labs/pytorch" label="Training an MLP in NumPy and PyTorch">
            The multiclass tutorial computes the softmax as <code>np.exp(Z - Z.max(axis=1, keepdims=True))</code>, then
            normalizes. Subtracting the row maximum avoids overflow and changes nothing, because the softmax only depends on
            differences of logits. In PyTorch (Lab 4), <code>CrossEntropyLoss</code> takes the raw logits and integer labels,
            so the model must <em>not</em> end with a softmax. Lab 4 also plotted the MNIST weight rows as the digit templates
            from this page.
          </FromLab>
          <Slides lecture={3} pages="39–49" />
        </>
      ),
    },
    {
      id: "summary",
      title: "Summary: three linear models",
      summary: "One table to remember: task, squashing, output, objective, convexity, closed form.",
      body: () => (
        <>
          <div className="table-wrap">
            <Table
              head={["", "Linear regression", "Logistic regression", "Multiclass logistic regression"]}
              rows={[
                ["Model", <M>{t`\hat y = \theta^\top\tilde{\mathbf x}`}</M>, <M>{t`\hat y = \sigma(\theta^\top\tilde{\mathbf x})`}</M>, <M>{t`\hat{\mathbf y} = \mathrm{softmax}(\tilde{\mathbf W}\tilde{\mathbf x})`}</M>],
                ["Task", "regression", "binary classification", "multiclass classification"],
                ["Squashing", "none", "sigmoid", "softmax"],
                ["Output", "a value", "a probability", "K probabilities"],
                ["Objective", "MSE", "BCE", "CCE"],
                ["Gradient", <M>{t`\tfrac2N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)`}</M>, <M>{t`\tfrac1N\tilde{\mathbf X}^\top(\hat{\mathbf y}-\mathbf y)`}</M>, <M>{t`\tfrac1N(\hat{\mathbf Y}-\mathbf Y)^\top\tilde{\mathbf X}`}</M>],
                ["Convex", "yes", "yes", "yes"],
                ["Closed form", "yes", "no", "no"],
                ["Likelihood (extra lecture)", "Gaussian", "Bernoulli", "categorical"],
              ]}
            />
          </div>
          <ul>
            <li>Sigmoid squashes <em>one</em> logit; softmax normalizes <em>K</em> logits.</li>
            <li>All three gradients have the structure <strong>error × input</strong>.</li>
            <li>All three losses are convex; only linear regression has a closed-form minimum.</li>
          </ul>
          <H>Why linear models are still used, and their limit</H>
          <p>
            Convex and fast to train, interpretable weights, great when classes are linearly separable. But if no straight
            line/plane separates the classes, a linear model can't succeed → transform the features (Lecture 4).
          </p>
          <Callout kind="exam">
            The Lecture 5 slides say <strong>Lecture 3's summary table is printed in the midterm booklet</strong>, so you don't need
            to memorize it. What you do need is to <em>use</em> it: explain why each entry is what it is, and apply the gradients
            (e.g. "which way does each weight move?").
          </Callout>
          <Slides lecture={3} pages="52–54" />
        </>
      ),
    },
  ],
};
