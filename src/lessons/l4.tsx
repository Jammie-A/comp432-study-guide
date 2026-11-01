import { Callout, FromLab, Formula, H, M, MB, MathCode, Slides, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { CapacityWidget } from "../widgets/l2";
import { ActivationWidget, BackpropWidget, NonConvexWidget, PulseWidget } from "../widgets/l4";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

export const lecture4: Lecture = {
  id: "l4",
  number: 4,
  title: "From linear models to neural networks",
  sections: [
    {
      id: "basis",
      title: "Feature transforms & basis functions",
      summary: "If the data isn't linear, transform the inputs first; then a linear model works again.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Lecture 3 ended with a problem: if no line separates the classes (or fits the curve), a linear model fails. The
            trick: map each input through a fixed non-linear function <M>{t`\phi`}</M> first, into a space where a line
            <em> does</em> work. Then run the same linear model on <M>{t`\phi(\mathbf x)`}</M>.
          </p>
          <MB>{t`\mathbf x\ \xrightarrow{\text{transform}}\ \phi(\mathbf x)\ \xrightarrow{\text{linear model}}\ \hat y\ \xrightarrow{\text{loss}}\ J(\theta)`}</MB>
          <H>Two examples from the slides</H>
          <Table
            head={["Task", "Transform", "Why it works"]}
            rows={[
              ["Classification: class inside a circle, other class around it", <M>{t`\phi(\mathbf x) = (x_1^2,\ x_2^2)^\top`}</M>, <>A circle <M>{t`x_1^2+x_2^2=r^2`}</M> becomes a straight line <M>{t`\phi_1+\phi_2=r^2`}</M> in the new space.</>],
              ["Regression: a sine-shaped curve", <M>{t`\phi(x) = \sin(2\pi x)`}</M>, <>Against <M>{t`\phi(x)`}</M>, y is a straight line.</>],
            ]}
          />
          <H>Which basis functions?</H>
          <ul>
            <li><strong>Polynomials</strong> <M>{t`1, x, x^2, x^3,\dots`}</M>: by Taylor's theorem, any smooth function is well approximated near a point by a sum of powers. Capture local shape.</li>
            <li><strong>Waves</strong> <M>{t`\sin(m\pi x)`}</M>: by Fourier series, periodic signals are sums of sines and cosines. Capture periodic structure.</li>
          </ul>
          <H>Linear basis function models</H>
          <Formula
            tex={t`\tilde\phi(\mathbf x) = \begin{pmatrix}\phi(\mathbf x)\\1\end{pmatrix}\in\mathbb R^{M+1},\qquad \phi:\mathbb R^D\to\mathbb R^M`}
            symbols={[
              [t`\phi(\mathbf x)`, "M non-linear basis functions applied to x"],
              [t`\tilde\phi`, "the same with a 1 appended (for the bias), just like x̃"],
              [t`M`, "number of basis functions (can differ from D)"],
            ]}
          />
          <Table
            head={["Problem", "Model", "Loss", "Parameters"]}
            rows={[
              ["Regression", <M>{t`\hat y = \theta^\top\tilde\phi(\mathbf x)`}</M>, "MSE", "vector, M + 1"],
              ["Binary classification", <M>{t`\hat y = \sigma(\theta^\top\tilde\phi(\mathbf x))`}</M>, "BCE", "vector, M + 1"],
              ["Multiclass", <M>{t`\hat{\mathbf y} = \mathrm{softmax}(\mathbf W\tilde\phi(\mathbf x))`}</M>, "CCE", "matrix, K × (M + 1)"],
            ]}
          />
          <Callout kind="intuition">
            Non-linear in the <strong>inputs</strong>, but still linear in the <strong>parameters</strong>. So everything from Lecture 3
            carries over: just replace <M>{t`\tilde{\mathbf x}`}</M> by <M>{t`\tilde\phi(\mathbf x)`}</M>. Same gradients, still convex.
          </Callout>
          <PyCell
            code={py`
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
X = rng.uniform(-2.5, 2.5, size=(200, 2))
y = (X[:, 0]**2 + X[:, 1]**2 < 2.5).astype(int)       # inner class = 1

Phi = X ** 2                                          # phi(x) = (x1^2, x2^2)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12); ax[0].set_title("input space: no line separates")
ax[1].scatter(Phi[:, 0], Phi[:, 1], c=y, cmap="coolwarm", s=12); ax[1].set_title("transformed: a line does")
ax[1].plot([0, 2.5], [2.5, 0], "k--")
`}
          />
          <Slides lecture={4} pages="5–13" />
        </>
      ),
    },
    {
      id: "polyfit",
      title: "Polynomial fitting & the design matrix",
      summary: "Powers of x as features, the design matrix Φ̃, and last week's closed form reused verbatim.",
      body: () => (
        <>
          <H>The setup</H>
          <Formula
            tex={t`\tilde\phi(x) = (x,\ x^2,\ \dots,\ x^M,\ 1)^\top,\qquad \hat y = \theta^\top\tilde\phi(x) = w_1x + w_2x^2 + \dots + w_Mx^M + b`}
            symbols={[
              [t`M`, "the order (degree) of the polynomial"],
              [t`\theta = (w_1,\dots,w_M,b)`, "M + 1 parameters"],
            ]}
          />
          <H>The design matrix</H>
          <MB>{t`\tilde{\boldsymbol\Phi} = \begin{pmatrix}x_1 & x_1^2 & \cdots & x_1^M & 1\\ \vdots &&&& \vdots\\ x_N & x_N^2 & \cdots & x_N^M & 1\end{pmatrix}\in\mathbb R^{N\times(M+1)},\qquad \theta^\star = \big(\tilde{\boldsymbol\Phi}^\top\tilde{\boldsymbol\Phi}\big)^{-1}\tilde{\boldsymbol\Phi}^\top\mathbf y`}</MB>
          <p>Each row is one example's features; it's X̃ from Lecture 3 with extra columns. The closed form applies verbatim with <M>{t`\tilde{\mathbf X}\to\tilde{\boldsymbol\Phi}`}</M>.</p>
          <Steps
            title="Fit a quadratic (M = 2) through (0, 1), (1, 0), (2, 3)"
            steps={[
              { title: "Design matrix: columns x, x², 1", body: <MB>{t`\tilde{\boldsymbol\Phi} = \begin{pmatrix}0&0&1\\1&1&1\\2&4&1\end{pmatrix}`}</MB> },
              { title: "N = 3 points, P = 3 parameters", body: <p>Φ̃ is square and invertible, so θ* solves <M>{t`\tilde{\boldsymbol\Phi}\theta = \mathbf y`}</M> exactly (zero training error).</p> },
              { title: "Row 1 (x = 0)", body: <MB>{t`b = 1`}</MB> },
              { title: "Rows 2 and 3", body: <MB>{t`w_1 + w_2 + 1 = 0,\quad 2w_1 + 4w_2 + 1 = 3 \;\Rightarrow\; w_1 = -3,\ w_2 = 2`}</MB> },
              { title: "Result", body: <MB>{t`\hat y = -3x + 2x^2 + 1`}</MB> },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 0.0, 3.0])
M = 2
Phi = np.column_stack([x**m for m in range(1, M + 1)] + [np.ones_like(x)])
theta = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
print(Phi)
print("theta (w1, w2, b) =", theta.round(3))
`}
          />
          <H>Choosing the order</H>
          <p>The order M is the model's <strong>capacity</strong> (Lecture 2). Too small underfits, too large overfits:</p>
          <CapacityWidget showWeights />
          <Think q="Set M = 9 with N = 10 and look at the weights. What do you notice?">
            They're enormous (thousands or more) with alternating signs: huge terms cancelling each other to produce values around
            ±1. That's the fingerprint of overfitting, and it motivates the next section: penalize large weights.
          </Think>
          <FromLab to="labs/pytorch" label="Training an MLP in NumPy and PyTorch">
            The polynomial-fitting tutorial ran this experiment for M = 0 to 9 and found something surprising: <em>gradient
            descent</em> on the degree-9 polynomial overfits much less than the closed-form solution. The huge alternating
            weights lie along very flat, badly conditioned directions that GD hasn't reached after 20,000 steps, so stopping
            early keeps the weights small, just like early stopping (Lecture 2).
          </FromLab>
          <Slides lecture={4} pages="14–18" />
        </>
      ),
    },
    {
      id: "regularization",
      title: "L2 regularization",
      summary: "Add λ/2·‖θ‖² to the loss to keep weights small: Occam's razor as a formula.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            More data cures overfitting, but data is expensive. Another cure: since overfitting shows up as huge weights,
            add a penalty for large weights to the loss. The optimizer then has to balance fitting the data against keeping
            θ small.
          </p>
          <Formula
            title="L2 regularization (weight decay, ridge regression)"
            tex={t`\tilde J(\theta) = J(\theta) + \frac\lambda2\|\theta\|_2^2,\qquad \nabla_\theta\tilde J = \nabla_\theta J + \lambda\theta`}
            symbols={[
              [t`\|\theta\|_2^2 = \sum_p\theta_p^2`, "sum of squared parameters: large weights cost a lot"],
              [t`\lambda > 0`, "regularization factor: how much we care about small weights. A hyperparameter (pick on the validation set)."],
              [t`\lambda\theta`, "the gradient of λ/2·‖θ‖²: the ½ cancels the 2 from the square"],
            ]}
          />
          <p>
            The name "weight decay": the GD update becomes <M>{t`\theta \leftarrow \theta - \eta(\nabla J + \lambda\theta) = (1-\eta\lambda)\theta - \eta\nabla J`}</M>:
            every step first shrinks θ a little.
          </p>
          <H>Closed form stays</H>
          <Steps
            steps={[
              { title: "Gradient of the MSE part (Lecture 3, with Φ̃)", body: <MB>{t`\nabla J = \tfrac2N\tilde{\boldsymbol\Phi}^\top(\tilde{\boldsymbol\Phi}\theta - \mathbf y)`}</MB> },
              { title: "Add the penalty's gradient and set to zero", body: <MB>{t`\tfrac2N\tilde{\boldsymbol\Phi}^\top(\tilde{\boldsymbol\Phi}\theta - \mathbf y) + \lambda\theta = 0`}</MB> },
              { title: "Multiply by N/2", body: <MB>{t`\tilde{\boldsymbol\Phi}^\top\tilde{\boldsymbol\Phi}\theta + \tfrac{N\lambda}2\theta = \tilde{\boldsymbol\Phi}^\top\mathbf y`}</MB> },
              { title: "Factor out θ (θ = Iθ)", body: <MB>{t`\Big(\tilde{\boldsymbol\Phi}^\top\tilde{\boldsymbol\Phi} + \tfrac{N\lambda}2\mathbf I\Big)\theta = \tilde{\boldsymbol\Phi}^\top\mathbf y\quad\Rightarrow\quad \theta^\star = \Big(\tilde{\boldsymbol\Phi}^\top\tilde{\boldsymbol\Phi} + \tfrac{N\lambda}2\mathbf I\Big)^{-1}\tilde{\boldsymbol\Phi}^\top\mathbf y`}</MB> },
            ]}
          />
          <H>Visual: the effect of λ</H>
          <CapacityWidget withLambda showWeights />
          <Table
            head={["λ", "Result"]}
            rows={[["0", "under-regularized → overfitting"], ["≈ 10⁻⁵", "proper regularization"], ["0.1", "over-regularized → underfitting (weights pushed too close to 0)"]]}
          />
          <PyCell
            code={py`
import numpy as np
rng = np.random.default_rng(0)
N, M = 10, 9
x = np.linspace(0, 1, N)
y = np.sin(2 * np.pi * x) + 0.2 * rng.normal(size=N)
Phi = np.column_stack([x**m for m in range(1, M + 1)] + [np.ones(N)])

for lam in [0, 1e-5, 0.1]:
    A = Phi.T @ Phi + N * lam / 2 * np.eye(M + 1)
    theta = np.linalg.solve(A, Phi.T @ y)
    print(f"lambda={lam:<6}  largest |weight| = {np.abs(theta).max():12.1f}")
`}
          />
          <Callout kind="note">
            Usually the bias b is left out of the penalty (scikit-learn does so); the slides keep it for simplicity. The extra
            lecture derives this penalty from a Gaussian prior on θ (MAP estimation).
          </Callout>
          <FromLab to="labs/from-scratch" label="Linear & logistic regression from scratch">
            scikit-learn calls this penalty <code>C</code>, the <em>inverse</em> strength: <M>{t`\lambda = 1/(CN)`}</M>, so
            Lab 4's <code>C=0.01</code> regularizes 100 times more than the default <code>C=1</code> (the weight images got
            smoother). In PyTorch, the same penalty is <code>SGD(..., weight_decay=1e-4)</code>, as in Lab 5.
          </FromLab>
          <Slides lecture={4} pages="17–21" />
        </>
      ),
    },
    {
      id: "mlp",
      title: "Neurons, layers & the MLP",
      summary: "A neural network learns the feature transform itself: layers of z = Wx + b followed by a non-linearity.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Hand-designing φ needs domain knowledge and is hard. Neural networks make the transform <strong>learnable</strong>:
            <M>{t`\ \hat y = \mathbf w^\top\phi_\nu(\mathbf x) + b`}</M>, where the parameters ν of the feature extractor are trained
            together with the linear "head" (w, b). This is <strong>representation learning</strong>.
          </p>
          <H>One neuron → one layer</H>
          <Formula
            tex={t`z_j = \sum_i W_{ji}x_i + b_j,\quad h_j = f(z_j)\qquad\Longrightarrow\qquad \mathbf z = \mathbf W\mathbf x + \mathbf b,\quad \mathbf h = f(\mathbf z)`}
            symbols={[
              [t`z_j`, <>neuron j's <strong>pre-activation</strong>: a weighted sum of inputs plus bias</>],
              [t`h_j = f(z_j)`, <>its <strong>activation</strong>, after a non-linear function f (e.g. ReLU)</>],
              [t`\mathbf W\in\mathbb R^{H\times D}`, "ONE ROW PER NEURON; H neurons, D inputs"],
              [t`\mathbf b\in\mathbb R^H`, "one bias per neuron"],
              [t`f(\mathbf z)`, "applied element-wise"],
            ]}
          />
          <p>Every input feeds every neuron: a <strong>fully-connected (dense) layer</strong>.</p>
          <H>The multi-layer perceptron (MLP)</H>
          <Formula
            tex={t`\hat{\mathbf y} = \mathbf W^{(2)}\underbrace{f^{(1)}\big(\mathbf W^{(1)}\mathbf x + \mathbf b^{(1)}\big)}_{\text{hidden representation }\mathbf h} + \mathbf b^{(2)}`}
            symbols={[
              [t`\mathbf W^{(1)}\in\mathbb R^{H\times D}`, "first layer: input → H hidden neurons"],
              [t`\mathbf W^{(2)}\in\mathbb R^{K\times H}`, "output layer: hidden → K outputs, NO activation (sigmoid/softmax live in the loss)"],
              [t`P = (D+1)H + (H+1)K`, "parameter count: weights + biases of both layers"],
            ]}
          />
          <p>Counted by weight matrices, this is a <strong>two-layer MLP</strong> (one hidden layer). A deep network stacks more: <M>{t`\hat{\mathbf y} = \mathbf W^{(4)}f^{(3)}(\mathbf W^{(3)}f^{(2)}(\mathbf W^{(2)}f^{(1)}(\mathbf W^{(1)}\mathbf x+\mathbf b^{(1)})+\mathbf b^{(2)})+\mathbf b^{(3)})+\mathbf b^{(4)}`}</M>.</p>
          <Steps
            title="Shapes and parameter count: D = 2 inputs, H = 3 hidden, K = 2 outputs"
            steps={[
              { title: "Layer 1", body: <MB>{t`\mathbf W^{(1)}\in\mathbb R^{3\times2},\ \mathbf b^{(1)}\in\mathbb R^3 \Rightarrow 6 + 3 = 9 = (D+1)H`}</MB> },
              { title: "Layer 2", body: <MB>{t`\mathbf W^{(2)}\in\mathbb R^{2\times3},\ \mathbf b^{(2)}\in\mathbb R^2 \Rightarrow 6 + 2 = 8 = (H+1)K`}</MB> },
              { title: "Total", body: <MB>{t`P = 9 + 8 = 17`}</MB> },
              { title: "Shapes through the network", body: <MB>{t`\mathbf x\,(2)\to\mathbf z^{(1)}\,(3)\to\mathbf h\,(3)\to\hat{\mathbf y}\,(2)`}</MB> },
            ]}
          />
          <H>Why the activation must be non-linear</H>
          <MB>{t`f(z)=z:\quad \hat{\mathbf y} = \mathbf W^{(2)}(\mathbf W^{(1)}\mathbf x+\mathbf b^{(1)})+\mathbf b^{(2)} = \underbrace{\mathbf W^{(2)}\mathbf W^{(1)}}_{\text{one matrix}}\mathbf x + \underbrace{\mathbf W^{(2)}\mathbf b^{(1)}+\mathbf b^{(2)}}_{\text{one bias}}`}</MB>
          <p>Without a non-linearity, stacking layers collapses into a single linear model: more parameters, zero extra expressivity.</p>
          <H>Math → code</H>
          <MathCode
            rows={[
              [t`\mathbf z^{(1)} = \mathbf W^{(1)}\mathbf x + \mathbf b^{(1)}`, "z1 = W1 @ x + b1", "(H, D)@(D,) → (H,)"],
              [t`\mathbf h = \mathrm{ReLU}(\mathbf z^{(1)})`, "h = np.maximum(0, z1)", "element-wise"],
              [t`\hat{\mathbf y} = \mathbf W^{(2)}\mathbf h + \mathbf b^{(2)}`, "y_hat = W2 @ h + b2", "(K, H)@(H,) → (K,)"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
rng = np.random.default_rng(0)
D, H, K = 2, 3, 2
W1, b1 = rng.normal(size=(H, D)), np.zeros(H)
W2, b2 = rng.normal(size=(K, H)), np.zeros(K)

x = np.array([1.0, 2.0])
z1 = W1 @ x + b1
h = np.maximum(0, z1)          # ReLU
y_hat = W2 @ h + b2
print("shapes:", x.shape, z1.shape, h.shape, y_hat.shape)
print("P =", W1.size + b1.size + W2.size + b2.size, "= (D+1)H + (H+1)K =", (D + 1) * H + (H + 1) * K)
`}
          />
          <Callout kind="exam">
            Common questions: shapes of each W and b, number of parameters, and "why can't we use the identity as activation?"
          </Callout>
          <FromLab to="labs/pytorch" label="Training an MLP in NumPy and PyTorch">
            In PyTorch, <code>nn.Linear(D, H)</code> stores <M>{t`\mathbf W`}</M> with shape <strong>(H, D)</strong>, one row
            per neuron, exactly as on this page. scikit-learn's <code>coefs_[0]</code> stores it the other way round: (D, H).
            Lab 4 also showed that <code>MLPClassifier(hidden_layer_sizes=())</code>, an MLP with no hidden layer, is
            multiclass logistic regression.
          </FromLab>
          <Slides lecture={4} pages="22–29, 42–44" />
        </>
      ),
    },
    {
      id: "activations",
      title: "Activation functions",
      summary: "Sigmoid, tanh and ReLU, and why ReLU is the default.",
      body: () => (
        <>
          <Table
            head={["Name", "f(z)", "f′(z)", "Notes"]}
            rows={[
              ["sigmoid", <M>{t`\sigma(z) = \frac1{1+e^{-z}}`}</M>, <M>{t`\sigma(z)(1-\sigma(z))`}</M>, "output (0, 1); saturates for large |z|"],
              ["tanh", <M>{t`\tanh(z) = 2\sigma(2z)-1`}</M>, <M>{t`1-\tanh^2(z)`}</M>, "output (−1, 1); saturates"],
              ["ReLU", <M>{t`\max(0, z)`}</M>, <M>{t`\mathbb 1[z>0]`}</M>, "the default: cheap, derivative 0 or 1"],
            ]}
          />
          <ActivationWidget />
          <Callout kind="intuition">
            <strong>Saturation:</strong> for large |z|, sigmoid and tanh are flat, so f′ ≈ 0. In backprop gradients get multiplied by
            f′, so they nearly vanish and learning stalls. ReLU's derivative is exactly 1 for every positive z. (Smooth variants like
            GELU are used in transformers.)
          </Callout>
          <PyCell
            code={py`
import numpy as np
z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
sig = 1 / (1 + np.exp(-z))
print("sigmoid   ", sig.round(3), " deriv", (sig * (1 - sig)).round(3))
print("tanh      ", np.tanh(z).round(3), " deriv", (1 - np.tanh(z)**2).round(3))
print("ReLU      ", np.maximum(0, z), "        deriv", (z > 0).astype(float))
`}
          />
          <Slides lecture={4} pages="30" />
        </>
      ),
    },
    {
      id: "nonconvex-uat",
      title: "Non-convexity & universal approximation",
      summary: "The price of the extra power: no convexity guarantee. The prize: one hidden layer can approximate anything.",
      body: () => (
        <>
          <H>Neural network losses are not convex</H>
          <p>
            Learning φ makes the loss non-convex: local minima, saddle points, plateaus; no guarantee of finding the global
            minimum. (Empirically, SGD and Adam still work well even with billions of parameters.)
          </p>
          <Steps
            title="The tiny counterexample (slides 33–34)"
            steps={[
              { title: "Setup", body: <><p>One data point (x, y) = (1, 1), two hidden ReLU units, no biases, squared error:</p><MB>{t`\hat y = w_1\,\mathrm{ReLU}(\nu_1x) + w_2\,\mathrm{ReLU}(\nu_2x),\qquad \theta = (\nu_1,\nu_2,w_1,w_2)`}</MB></> },
              { title: "Solution θ = (1, 0, 1, 0): top neuron does the work", body: <MB>{t`\hat y = 1\cdot\mathrm{ReLU}(1) + 0 = 1 \Rightarrow J(\theta) = 0`}</MB> },
              { title: "Solution θ′ = (0, 1, 0, 1): bottom neuron does the work", body: <MB>{t`\text{by symmetry } J(\theta') = 0`}</MB> },
              { title: "Midpoint t = ½", body: <MB>{t`\tfrac12\theta + \tfrac12\theta' = (\tfrac12,\tfrac12,\tfrac12,\tfrac12):\quad \hat y = \tfrac12\mathrm{ReLU}(\tfrac12) + \tfrac12\mathrm{ReLU}(\tfrac12) = \tfrac12`}</MB> },
              { title: "Check the convexity inequality", body: <MB>{t`J(\text{midpoint}) = (\tfrac12 - 1)^2 = \tfrac14 \;>\; \tfrac12J(\theta)+\tfrac12J(\theta') = 0`}</MB> },
              { title: "Conclusion", body: <p>The chord lies <em>below</em> the graph somewhere, so J is not convex. (We can't use the Hessian test here: ReLU has kinks.)</p> },
            ]}
          />
          <NonConvexWidget />
          <H>Universal approximation theorem</H>
          <p>
            A one-hidden-layer MLP with a non-polynomial activation can approximate <strong>any continuous function on a compact
            domain</strong> as accurately as we want, given enough neurons H.
          </p>
          <Steps
            title="Why it's plausible: build it from pulses"
            steps={[
              { title: "A steep sigmoid is a switch", body: <p><M>{t`\sigma(kx)`}</M> with large k jumps from 0 to 1 at x = 0. <M>{t`\sigma(k(x-a))`}</M> switches on at a.</p> },
              { title: "Two switches make a pulse", body: <MB>{t`\sigma(500x) - \sigma(500(x-0.2)) \approx \begin{cases}1 & 0<x<0.2\\0&\text{otherwise}\end{cases}`}</MB> },
              { title: "That's a 2-neuron MLP", body: <MB>{t`h_1 = \sigma(500x),\quad h_2 = \sigma(500x - 100),\quad \hat y = 0.55h_1 - 0.55h_2`}</MB> },
              { title: "Sum pulses over all intervals", body: <p>Height of each pulse = average of the target on its interval. 2 neurons per pulse; more intervals → better approximation.</p> },
            ]}
          />
          <PulseWidget />
          <Callout kind="exam">
            What the theorem does <strong>not</strong> say: that training will <em>find</em> those parameters (the loss is non-convex),
            or how many neurons you need. Representation ≠ learnability.
          </Callout>
          <Slides lecture={4} pages="31–41" />
        </>
      ),
    },
    {
      id: "backprop-scalar",
      title: "Backpropagation: the chain rule",
      summary: "Compute all gradients in one backward sweep by multiplying local derivatives and reusing them.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            To train a network with GD we need <M>{t`\partial J/\partial w`}</M> for every weight. <strong>Backpropagation</strong> is
            the chain rule applied efficiently: first a forward pass computes and stores every intermediate value, then a
            backward pass goes from the loss towards the input, multiplying local derivatives. Each intermediate derivative
            is computed once and reused.
          </p>
          <H>The chain rule on a chain</H>
          <MB>{t`x\xrightarrow{\times w_1}h_1\xrightarrow{\times w_2}h_2\xrightarrow{(\cdot-y)^2}J`}</MB>
          <MB>{t`\frac{\partial J}{\partial w_2} = \frac{dJ}{dh_2}\cdot\frac{\partial h_2}{\partial w_2},\qquad \frac{\partial J}{\partial w_1} = \underbrace{\frac{dJ}{dh_2}\cdot\frac{dh_2}{dh_1}}_{=\,dJ/dh_1}\cdot\frac{\partial h_1}{\partial w_1}`}</MB>
          <p>Notice <M>{t`dJ/dh_2`}</M> appears in both — compute it once, pass it backward.</p>
          <H>Visual (slide 47 numbers)</H>
          <BackpropWidget />
          <Table
            head={["Pass", "Computation", "Value"]}
            rows={[
              ["forward", "h₁ = w₁x", "2"],
              ["forward", "h₂ = w₂h₁", "6"],
              ["forward", "J = (h₂ − y)²", "4"],
              ["backward", "dJ/dh₂ = 2(h₂ − y)", "4"],
              ["backward", "∂J/∂w₂ = dJ/dh₂ · h₁", "8"],
              ["backward", "dJ/dh₁ = dJ/dh₂ · w₂", "12"],
              ["backward", "∂J/∂w₁ = dJ/dh₁ · x", "24"],
            ]}
          />
          <PyCell
            code={py`
x, y, w1, w2 = 2.0, 4.0, 1.0, 3.0
# forward pass (store everything)
h1 = w1 * x
h2 = w2 * h1
J = (h2 - y) ** 2
# backward pass
dJ_dh2 = 2 * (h2 - y)
dJ_dw2 = dJ_dh2 * h1          # local derivative dh2/dw2 = h1
dJ_dh1 = dJ_dh2 * w2          # local derivative dh2/dh1 = w2
dJ_dw1 = dJ_dh1 * x           # local derivative dh1/dw1 = x
print(f"J = {J}, dJ/dw2 = {dJ_dw2}, dJ/dw1 = {dJ_dw1}")

# check with finite differences
eps = 1e-6
J_of = lambda w1, w2: (w2 * w1 * x - y) ** 2
print("numerical:", (J_of(w1, w2 + eps) - J) / eps, (J_of(w1 + eps, w2) - J) / eps)
`}
          />
          <Callout kind="note">
            In PyTorch, <code>loss.backward()</code> does exactly this: it records the computation graph in the forward pass and
            runs the backward pass for every supported operation.
          </Callout>
          <Slides lecture={4} pages="47–48, 52" />
        </>
      ),
    },
    {
      id: "backprop-vector",
      title: "Backprop with vectors: a full MLP by hand",
      summary: "Jacobians and vector–Jacobian products, then the slide's 'try at home' MLP worked step by step.",
      body: () => (
        <>
          <H>The Jacobian</H>
          <Formula
            tex={t`\mathbf b = f(\mathbf a),\ \mathbf a\in\mathbb R^A,\ \mathbf b\in\mathbb R^B:\qquad \mathbf J_{\mathbf a}\mathbf b = \frac{\partial\mathbf b}{\partial\mathbf a^\top}\in\mathbb R^{B\times A},\quad [\mathbf J_{\mathbf a}\mathbf b]_{ji} = \frac{\partial b_j}{\partial a_i}`}
            symbols={[
              [t`\mathbf J_{\mathbf a}\mathbf b`, "all partial derivatives of the outputs w.r.t. the inputs: one ROW per output, one COLUMN per input"],
              [t`\nabla_{\mathbf a}J = (\mathbf J_{\mathbf a}\mathbf b)^\top\nabla_{\mathbf b}J`, <>a backward step: the <strong>vector–Jacobian product (VJP)</strong>. Jacobians are never built explicitly.</>],
            ]}
          />
          <H>The four VJP rules you need</H>
          <Table
            head={["Operation", "Backward rule", "Code"]}
            rows={[
              [<>element-wise <M>{t`\mathbf h = f(\mathbf z)`}</M></>, <M>{t`\nabla_{\mathbf z}J = f'(\mathbf z)\odot\nabla_{\mathbf h}J`}</M>, <code>dz = fprime(z) * dh</code>],
              [<>affine, w.r.t. input <M>{t`\mathbf z = \mathbf W\mathbf x+\mathbf b`}</M></>, <M>{t`\nabla_{\mathbf x}J = \mathbf W^\top\nabla_{\mathbf z}J`}</M>, <code>dx = W.T @ dz</code>],
              ["affine, w.r.t. weights", <M>{t`\nabla_{\mathbf W}J = (\nabla_{\mathbf z}J)\,\mathbf x^\top`}</M>, <code>dW = np.outer(dz, x)</code>],
              ["affine, w.r.t. bias", <M>{t`\nabla_{\mathbf b}J = \nabla_{\mathbf z}J`}</M>, <code>db = dz</code>],
            ]}
          />
          <Callout kind="intuition">
            Same pattern as Lecture 3: <strong>weight gradient = (error at the output of the layer) × (input to the layer)ᵀ</strong>.
            And passing the error back through a layer multiplies by <M>{t`\mathbf W^\top`}</M>.
          </Callout>
          <H>Try at home (slide 50), fully worked</H>
          <p>
            Graph: <M>{t`\mathbf x\to\text{affine}\to\text{ReLU}\to\text{affine}\to\sigma\to\text{BCE}`}</M> with
            <M>{t`\ \mathbf W^{(1)} = \begin{pmatrix}1&1\\1&-1\end{pmatrix},\ \mathbf b^{(1)} = (1,0),\ \mathbf W^{(2)} = (1,\ 2),\ b^{(2)} = -4,\ \mathbf x = (1,2),\ y = 1`}</M>.
          </p>
          <Steps
            steps={[
              { title: "Forward: first affine layer", body: <MB>{t`\mathbf z^{(1)} = \begin{pmatrix}1&1\\1&-1\end{pmatrix}\begin{pmatrix}1\\2\end{pmatrix}+\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}3+1\\-1+0\end{pmatrix} = \begin{pmatrix}4\\-1\end{pmatrix}`}</MB> },
              { title: "Forward: ReLU", body: <MB>{t`\mathbf h^{(1)} = (\max(0,4),\ \max(0,-1)) = (4,\ 0)`}</MB> },
              { title: "Forward: second affine layer", body: <MB>{t`z^{(2)} = 1\cdot4 + 2\cdot0 - 4 = 0`}</MB> },
              { title: "Forward: sigmoid and BCE", body: <MB>{t`\hat y = \sigma(0) = 0.5,\qquad J = -\ln(0.5) = \ln 2`}</MB> },
              { title: "Backward: through BCE", body: <MB>{t`\frac{dJ}{d\hat y} = -\frac y{\hat y} + \frac{1-y}{1-\hat y} = -\frac1{0.5} + 0 = -2`}</MB> },
              { title: "Backward: through σ", body: <MB>{t`\frac{dJ}{dz^{(2)}} = \frac{dJ}{d\hat y}\cdot\hat y(1-\hat y) = -2\cdot0.25 = -0.5\quad(=\hat y - y\ \checkmark)`}</MB> },
              { title: "Backward: second layer's parameters", body: <MB>{t`\nabla_{\mathbf W^{(2)}}J = -0.5\cdot\mathbf h^{(1)\top} = (-2,\ 0),\qquad \frac{\partial J}{\partial b^{(2)}} = -0.5`}</MB> },
              { title: "Backward: into the hidden layer (multiply by Wᵀ)", body: <MB>{t`\nabla_{\mathbf h^{(1)}}J = \mathbf W^{(2)\top}(-0.5) = (-0.5,\ -1)`}</MB> },
              { title: "Backward: through ReLU (multiply by 1[z > 0] = (1, 0))", body: <MB>{t`\nabla_{\mathbf z^{(1)}}J = (1,\ 0)\odot(-0.5,\ -1) = (-0.5,\ 0)`}</MB> },
              { title: "Backward: first layer's parameters (outer product with x)", body: <MB>{t`\nabla_{\mathbf W^{(1)}}J = \begin{pmatrix}-0.5\\0\end{pmatrix}\begin{pmatrix}1&2\end{pmatrix} = \begin{pmatrix}-0.5&-1\\0&0\end{pmatrix},\qquad \nabla_{\mathbf b^{(1)}}J = (-0.5,\ 0)`}</MB> },
              { title: "Interpret", body: <p>The second hidden neuron was "off" (z = −1 &lt; 0), so ReLU blocks its gradient: its weights get zero gradient this step.</p> },
            ]}
          />
          <PyCell
            title="Verify with NumPy (manual backprop)"
            code={py`
import numpy as np
W1 = np.array([[1., 1.], [1., -1.]]); b1 = np.array([1., 0.])
W2 = np.array([1., 2.]);              b2 = -4.0
x = np.array([1., 2.]);               y = 1.0

# forward
z1 = W1 @ x + b1
h1 = np.maximum(0, z1)
z2 = W2 @ h1 + b2
y_hat = 1 / (1 + np.exp(-z2))
J = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# backward
dJ_dyhat = -y / y_hat + (1 - y) / (1 - y_hat)
dJ_dz2 = dJ_dyhat * y_hat * (1 - y_hat)
dW2 = dJ_dz2 * h1
db2 = dJ_dz2
dh1 = W2 * dJ_dz2               # W2^T @ dz2 (W2 is a single row here)
dz1 = (z1 > 0) * dh1            # ReLU' ⊙ dh1
dW1 = np.outer(dz1, x)
db1 = dz1

print("z1", z1, " h1", h1, " z2", z2, " y_hat", y_hat, " J", round(J, 4))
print("dJ/dyhat", dJ_dyhat, " dJ/dz2", dJ_dz2)
print("dW2", dW2, " db2", db2, " dh1", dh1, " dz1", dz1)
print("dW1\n", dW1, "\ndb1", db1)
`}
          />
          <PyCell
            readOnly
            title="Same check with PyTorch autograd (read-only)"
            code={py`
import torch
W1 = torch.tensor([[1., 1.], [1., -1.]], requires_grad=True)
b1 = torch.tensor([1., 0.], requires_grad=True)
W2 = torch.tensor([[1., 2.]], requires_grad=True)
b2 = torch.tensor([-4.], requires_grad=True)
x, y = torch.tensor([1., 2.]), torch.tensor([1.])

y_hat = torch.sigmoid(W2 @ torch.relu(W1 @ x + b1) + b2)
loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
loss.backward()
print(W1.grad, b1.grad, W2.grad, b2.grad)
`}
          />
          <H>Summary: three model families</H>
          <Table
            head={["", "Linear model", "Linear basis function model", "Neural network (MLP)"]}
            rows={[
              ["Features", "x̃, raw", "φ̃(x), fixed by hand", "φ_ν(x), learned"],
              ["Parameters", "D + 1", "M + 1", "(D+1)H + (H+1)K"],
              ["Objective", "convex", "convex", "non-convex"],
              ["Gradient", "error × input", "error × features", "backpropagation"],
              ["Closed form", "regression only", "regression only", "no"],
            ]}
          />
          <FromLab to="labs/pytorch" label="Training an MLP in NumPy and PyTorch">
            The MLP tutorial writes these VJP rules for a whole minibatch in NumPy:{" "}
            <code>z2_grad = (y_hat - y) / N</code>, <code>w2_grad = H.T @ z2_grad</code>,{" "}
            <code>Z1_grad = np.outer(z2_grad, w2) * relu_derivative(Z1)</code>, <code>W1_grad = Z1_grad.T @ X</code>. Each
            weight gradient is still "error × input", summed over the examples by the matrix product.
          </FromLab>
          <Slides lecture={4} pages="49–51, 55" />
        </>
      ),
    },
  ],
};
