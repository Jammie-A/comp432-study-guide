import { Callout, H, M, MB, MathCode, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { Lecture } from "./types";
import { Private } from "../private";

const t = String.raw;
const py = String.raw;

function LabRef({ children }: { children: string }) {
  return <p className="slides">Lab material: {children}</p>;
}

export const labs: Lecture = {
  id: "labs",
  number: 6,
  title: "Code you must be able to read",
  label: "Labs 1–5",
  short: "Labs",
  sections: [
    {
      id: "numpy",
      title: "NumPy: shapes, views, vectorization",
      summary: "The NumPy behaviour the labs rely on: views vs copies, axis/keepdims, broadcasting, and paired shuffles.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The midterm shows you code and asks what it does. Most traps are NumPy details: does this line copy or share
            memory, which axis does it reduce, which shape comes out. Lab 1 drilled exactly these.
          </p>
          <H>Views vs copies</H>
          <Table
            head={["Operation", "Result", "Shares memory with x?"]}
            rows={[
              [<code>x[:2, :2]</code>, "slice", "yes (a view)"],
              [<code>x.T</code>, "transpose", "yes (a view, only the strides change)"],
              [<code>x.reshape(...)</code>, "reshape", "usually yes"],
              [<code>x.copy()</code>, "copy", "no"],
              [<code>x[[0, 2]]</code>, "fancy (list/array) indexing", "no (a copy)"],
              [<code>x + 1</code>, "arithmetic", "no (a new array)"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
x = np.arange(6.).reshape(2, 3)
y = x[:, :2]          # a view
y[0, 0] = 100         # ... so this also changes x
z = x[:, :2].copy()   # a copy
z[1, 1] = -1          # ... x is untouched
print(x)
print("y shares memory with x:", np.may_share_memory(x, y))
print("z shares memory with x:", np.may_share_memory(x, z))
`}
          />
          <H>Axis and keepdims</H>
          <MathCode
            rows={[
              ["\\text{mean of each column}", "X.mean(axis=0)", "(N, D) → (D,): reduces over the rows (examples)"],
              ["\\text{mean of each row}", "X.mean(axis=1)", "(N, D) → (N,): reduces over the columns (features)"],
              ["\\text{row sums kept as a column}", "E.sum(axis=1, keepdims=True)", "(N, K) → (N, 1): broadcasts back against (N, K)"],
            ]}
          />
          <Callout kind="intuition">
            Rule of thumb: <code>axis=k</code> makes dimension k <em>disappear</em>. In this course rows are examples, so
            <code> axis=0</code> averages over examples (batch norm, standardization) and <code>axis=1</code> over features or
            classes (layer norm, softmax normalizer, CCE's sum over K).
          </Callout>
          <H>Vectorization</H>
          <p>
            Replace Python loops by array operations. Lab 1's example: subtract the minimum from every entry. The double loop
            becomes <code>x - x.min()</code>; Lab 3's grid evaluation becomes <code>np.meshgrid</code>; every model in the course
            becomes one matrix product (<code>X_tilde @ theta</code>).
          </p>
          <H>Shuffling inputs and labels together</H>
          <PyCell
            code={py`
import numpy as np
X = np.array([[0, 0], [0, 1], [1, 1]])
Y = np.array([10, 20, 30])
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))      # ONE permutation...
print(X[perm], Y[perm])             # ...applied to both, so rows still match

bad_X, bad_Y = X[rng.permutation(3)], Y[rng.permutation(3)]   # two different permutations: BUG
print("mismatched:", bad_X.tolist(), bad_Y.tolist())
`}
          />
          <Private id="labs/numpy/strides" quiet />
          <LabRef>assignment 1 (exercises 2.1–2.4), the NumPy tutorial</LabRef>
        </>
      ),
    },
    {
      id: "finite-differences",
      title: "Finite differences: checking gradients numerically",
      summary: "Approximate a derivative with two function evaluations — and why ε can be too large or too small.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            A finite difference turns the derivative's definition into code. It's the standard way to check a hand-written
            gradient (Lab 1 exercise 1.4, Lab 2 exercise 1.2, and every "verify with autograd" exercise).
          </p>
          <MB>{t`f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}\qquad\text{(central difference)}`}</MB>
          <PyCell
            code={py`
def approx_derivative_of(f, epsilon=1e-6):
    return lambda x: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)   # a closure

f = lambda x: x**3          # exact derivative: 3x^2 = 12 at x = 2
for eps in [1, 1e-5, 1e-15]:
    print(f"eps = {eps:<6g}: {approx_derivative_of(f, eps)(2.0):.10f}")
`}
          />
          <Table
            head={["ε", "Result at x = 2", "Why"]}
            rows={[
              ["1", "13 (wrong)", "too large: the approximation's own error (here exactly ε² = 1 for x³)"],
              ["10⁻⁵", "12.0000000002 (great)", "the sweet spot"],
              ["10⁻¹⁵", "≈ 11.99 (wrong)", "too small: f(x+ε) − f(x−ε) loses almost all digits to floating-point rounding"],
            ]}
          />
          <Callout kind="exam">
            Expect "which ε gives the best estimate" or "why is the estimate with ε = 10⁻¹⁵ wrong?" — floating-point
            cancellation, not a math error.
          </Callout>
          <LabRef>assignment 1 (exercise 1.4), assignment 2 (exercise 1.2)</LabRef>
        </>
      ),
    },
    {
      id: "from-scratch",
      title: "Linear & logistic regression from scratch (+ scikit-learn)",
      summary: "The exact NumPy lines from Lab 3, and the scikit-learn settings that make it agree with you.",
      body: () => (
        <>
          <H>The tutorial implementations</H>
          <p>
            The lab 3 tutorials (handed out with their code) take one gradient-descent step like this. Assignment 3 then asks
            you to write the same ideas yourself, in vectorized form.
          </p>
          <PyCell
            title="One gradient-descent step: linear and logistic regression (lab 3 tutorials)"
            code={py`
import numpy as np

def gd_step_linear(X, Y, w, b, lr):           # linear regression tutorial: y_hat = w x + b
    Y_hat = w * X + b
    grad_w = 2 * ((Y_hat - Y) * X).mean()
    grad_b = 2 * (Y_hat - Y).mean()
    return w - lr * grad_w, b - lr * grad_b

def gd_step_logistic(X_tilde, y, theta, lr):  # logistic regression tutorial
    y_hat = 1 / (1 + np.exp(-(X_tilde @ theta)))
    grad = X_tilde.T @ (y_hat - y) / y.size  # error times input, no factor 2
    return theta - lr * grad

# Lecture 3's mini iris data: GD reaches the closed-form solution (0.375, -0.125)
X, Y = np.array([[1.0], [3.0], [5.0]]), np.array([[0.2], [1.1], [1.7]])
w, b = 0.5, 1.0
for _ in range(3000):
    w, b = gd_step_linear(X, Y, w, b, 0.025)
print("linear  :", round(w, 3), round(b, 3))

X_tilde = np.array([[0.0, 1], [1.0, 1], [3.0, 1], [4.0, 1]])
y = np.array([0.0, 0.0, 1.0, 1.0])
theta = np.zeros(2)
for _ in range(200):
    theta = gd_step_logistic(X_tilde, y, theta, 0.5)
print("logistic:", theta.round(3), "-> boundary at x =", round(-theta[1] / theta[0], 2))
`}
          />
          <Private id="labs/from-scratch/a3" quiet />
          <H>scikit-learn: the settings that trip people up</H>
          <Table
            head={["Setting", "Meaning"]}
            rows={[
              [<code>LinearRegression(fit_intercept=False)</code>, "use it when X_tilde already has the column of ones; otherwise sklearn adds its own intercept"],
              [<code>LogisticRegression()</code>, <>default <code>C=1</code> <strong>regularizes</strong> (L2 on the weights, not the bias). It does not minimize plain BCE.</>],
              [<><code>C</code></>, <>the <strong>inverse</strong> regularization strength: smaller C = stronger penalty. Equivalent to <M>{t`\lambda = 1/(CN)`}</M>. <code>C=np.inf</code> = no penalty.</>],
              [<code>model.score(X, y)</code>, "accuracy; the error rate in percent is (1 − score) · 100"],
              [<code>coef_</code>, "LogisticRegression: shape (K, D), one row per class, like our W"],
            ]}
          />
          <Private id="labs/from-scratch/a3-sklearn" />
          <Callout kind="note">
            Lab 3's last remark: on <strong>linearly separable</strong> data, unregularized logistic regression has no minimum —
            the loss keeps decreasing as the weights grow, so solvers stop only at their tolerance (sklearn's got weights ≈ 10 and 17).
            That's one reason regularization exists.
          </Callout>
          <Callout kind="intuition">
            From the multiclass tutorial: the softmax only depends on <em>differences</em> of logits, so code subtracts the row max
            before <code>np.exp</code> (<code>np.exp(Z - Z.max(axis=1, keepdims=True))</code>) to avoid overflow without changing the
            result. Also: accuracy saturates long before the loss stops decreasing.
          </Callout>
          <LabRef>assignment 3, tutorials on linear, logistic and multiclass logistic regression</LabRef>
        </>
      ),
    },
    {
      id: "pytorch",
      title: "PyTorch & MLPs on MNIST",
      summary: "Tensors, autograd, nn.Sequential, the training loop, and the scikit-learn ↔ PyTorch correspondence of Lab 4.",
      body: () => (
        <>
          <H>Autograd in five lines (PyTorch tutorial)</H>
          <PyCell
            readOnly
            title="PyTorch (read-only)"
            code={py`
x = torch.arange(3, dtype=torch.float32, requires_grad=True)   # x = [0, 1, 2]
y = ((x + 2) ** 2 + 3).mean()
y.backward()
print(x.grad)          # tensor([1.3333, 2.0000, 2.6667])
`}
          />
          <Steps
            title="Check it by hand (chain rule)"
            steps={[
              { title: "Write y per entry", body: <MB>{t`y = \frac1D\sum_i \big[(x_i+2)^2 + 3\big],\quad D = 3`}</MB> },
              { title: "Differentiate", body: <MB>{t`\frac{\partial y}{\partial x_i} = \frac1D\cdot 2(x_i + 2)`}</MB> },
              { title: "Plug in x = (0, 1, 2)", body: <MB>{t`\tfrac23(2, 3, 4) = (\tfrac43,\ 2,\ \tfrac83)`}</MB> },
            ]}
          />
          <H>The training loop (MLP tutorial)</H>
          <PyCell
            readOnly
            title="PyTorch MLP training loop from the lab 4 MLP tutorial (read-only)"
            code={py`
model = torch.nn.Sequential(
    torch.nn.Linear(D, H),           # W1 of shape (H, D), b1 of shape (H,)
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),           # w2 of shape (1, H), b2 of shape (1,)
    torch.nn.Sigmoid(),
    torch.nn.Flatten(start_dim=0),   # (N, 1) -> (N,)
)
loss_function = torch.nn.BCELoss()                    # takes probabilities (the model ends with Sigmoid)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(1, N_epochs + 1):
    perm = torch.randperm(X_train_pt.shape[0])        # new random order every epoch
    for start in range(0, X_train_pt.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        y_hat = model(X_train_pt[idx])                # forward pass
        loss = loss_function(y_hat, y_train_pt[idx])  # objective on the minibatch
        optimizer.zero_grad()                         # reset accumulated gradients
        loss.backward()                               # backpropagation
        optimizer.step()                              # gradient descent step
    with torch.no_grad():                             # monitoring: no graph needed
        train_loss_pt.append(loss_function(model(X_train_pt), y_train_pt).item())
`}
          />
          <Private id="labs/pytorch/a4" />
          <Table
            head={["Thing", "What to remember"]}
            rows={[
              [<code>nn.Linear(in, out)</code>, <>weight shape <strong>(out, in)</strong>, one row per output unit; bias (out,)</>],
              [<code>CrossEntropyLoss</code>, "input: logits (N, K); target: class indices, int64 (N,). Don't add a softmax to the model."],
              [<code>BCEWithLogitsLoss</code>, "binary: takes logits, applies the sigmoid itself; targets are floats"],
              [<code>BCELoss</code>, "binary: takes probabilities (model must end with Sigmoid)"],
              [<code>dtype</code>, "inputs float32, class labels int64"],
              [<code>model.train() / model.eval()</code>, "switch dropout and batch norm between training and inference behaviour"],
              [<code>torch.no_grad()</code>, "no computation graph: use for evaluation"],
              [<code>x.detach().numpy()</code>, "tensor → NumPy (and .cpu() first if it lives on a GPU)"],
            ]}
          />
          <H>scikit-learn vs PyTorch (Lab 4)</H>
          <Table
            head={["", "scikit-learn", "PyTorch"]}
            rows={[
              ["Multiclass logistic regression", <><code>LogisticRegression()</code> or <code>MLPClassifier(hidden_layer_sizes=())</code></>, <code>Sequential(Linear(784, 10))</code>],
              ["One hidden layer of 32 ReLU", <code>MLPClassifier(hidden_layer_sizes=(32,))</code>, <code>Linear(784, 32), ReLU(), Linear(32, 10)</code>],
              ["First-layer weights", <><code>coefs_[0]</code>, shape <strong>(784, 32)</strong> (transposed!)</>, <><code>model[0].weight</code>, shape <strong>(32, 784)</strong></>],
              ["Biases", <code>intercepts_</code>, <code>model[0].bias</code>],
              ["Autodiff / GPU", "no / no", "yes / yes"],
            ]}
          />
          <Callout kind="mistake">
            <code>hidden_layer_sizes=(32)</code> is just the integer 32 in parentheses; a one-element tuple needs the comma:
            <code> (32,)</code>. And fit <code>MinMaxScaler</code> on the training set only, then <code>transform</code> the test set.
          </Callout>
          <H>Counting parameters</H>
          <PyCell
            code={py`
# P = sum over layers of (inputs + 1) * outputs
layers = [784, 128, 64, 10]
P = sum((a + 1) * b for a, b in zip(layers[:-1], layers[1:]))
print("784-128-64-10 MLP:", P)              # 109386
print("784-10 (logistic regression):", 785 * 10)
`}
          />
          <Callout kind="note">
            From the polynomial tutorial: gradient descent on a degree-9 polynomial overfits <em>less</em> than the direct solution,
            because the huge weights lie along flat, badly conditioned directions GD hasn't reached after 20,000 steps — an implicit
            early stopping.
          </Callout>
          <LabRef>assignment 4, tutorials on PyTorch, MLPs and polynomial fitting</LabRef>
        </>
      ),
    },
    {
      id: "conv-code",
      title: "Convolutions in code",
      summary: "(N, C, H, W) tensors, Conv2d weight shapes, output sizes, and the in_features of the linear head.",
      body: () => (
        <>
          <H>The 2-D convolution as loops (convolution tutorial)</H>
          <MB>{t`a_{i,j} = \sum_{u=0}^{F-1}\sum_{v=0}^{F-1} w_{u,v}\,x_{i+u,\,j+v},\qquad \text{output } (H - F + 1)\times(W - F + 1)\ \text{without padding}`}</MB>
          <PyCell
            code={py`
import numpy as np

def conv2d_python(image, w):
    H, W = image.shape
    F, _ = w.shape
    out = np.zeros((H - F + 1, W - F + 1))
    for i in range(H - F + 1):
        for j in range(W - F + 1):
            out[i, j] = (w * image[i:i + F, j:j + F]).sum()   # dot product with the window
    return out

image = np.zeros((5, 6)); image[:, 3:] = 1                   # dark left half, bright right half
w_vertical = np.array([[-1, 1], [-1, 1]])                     # right column minus left column
print(conv2d_python(image, w_vertical))                       # lights up at the vertical edge
`}
          />
          <H>Shapes in PyTorch</H>
          <Table
            head={["Object", "Shape"]}
            rows={[
              ["batch of images", <><strong>(N, C, H, W)</strong>: images, channels, height, width</>],
              [<code>Conv2d(C_in, C_out, F)</code>, <>weight <strong>(C_out, C_in, F, F)</strong>, bias (C_out,)</>],
              [<code>Conv1d(C_in, C_out, F)</code>, "weight (C_out, C_in, F); input (N, C_in, W)"],
              [<code>MaxPool2d(2)</code>, "halves H and W (rounding down); no parameters"],
              [<code>Flatten()</code>, "(N, C, H, W) → (N, C·H·W); the batch axis stays"],
            ]}
          />
          <Steps
            title="Practice: track the shapes of a small CNN on 32×32 RGB images"
            steps={[
              { title: "Input", body: <MB>{t`(N, 3, 32, 32)`}</MB> },
              { title: "Conv2d(3, 8, 5), no padding: 32 − 5 + 1 = 28", body: <MB>{t`(N, 8, 28, 28),\quad 5\cdot5\cdot3\cdot8 + 8 = 608\ \text{parameters}`}</MB> },
              { title: "ReLU, then MaxPool2d(2): halve the size", body: <MB>{t`(N, 8, 14, 14)`}</MB> },
              { title: "Flatten", body: <MB>{t`(N,\ 8\cdot14\cdot14) = (N,\ 1568)`}</MB> },
              { title: "Linear(1568, 10)", body: <MB>{t`1568\cdot10 + 10 = 15\,690\ \text{parameters, total } 16\,298`}</MB> },
            ]}
          />
          <Private id="labs/conv-code/a5-mnist" quiet />
          <Callout kind="exam">
            "What is <code>in_features</code> of the linear layer?" is the classic CNN code question. Track the shape layer by
            layer: conv (with or without padding), pooling halves, flatten multiplies channels × height × width.
          </Callout>
          <Private id="labs/conv-code/a5-1d" />
          <LabRef>assignment 5, the convolution layer tutorial</LabRef>
        </>
      ),
    },
    {
      id: "debugging",
      title: "Debugging neural networks",
      summary: "The step-by-step strategy and the five bugs of the Lab 5 debugging tutorial.",
      body: () => (
        <>
          <H>The strategy</H>
          <ol>
            <li><strong>Check the data</strong>: inputs and labels match (also after shuffling), shapes and value ranges as expected.</li>
            <li><strong>Start simple</strong> (e.g. a linear model): the loss must decrease at least a little. If not: check minibatches, tensor shapes, whether parameters change at all (forgotten <code>backward()</code>/<code>step()</code>), softmax axis, a few learning rates.</li>
            <li><strong>Implement the final model</strong>; fix shape mismatches; <code>train()</code> for training, <code>eval()</code> for evaluation.</li>
            <li><strong>Overfitting test</strong>: 5–10 examples, many epochs, large learning rate. A correct model drives the training loss to ≈ 0. If it can't even memorize, there's a bug.</li>
            <li><strong>Run the real experiment</strong>; NaNs → smaller learning rate or gradient clipping; tune on validation.</li>
          </ol>
          <H>The five bugs</H>
          <Table
            head={["Symptom", "Cause", "Fix"]}
            rows={[
              [<>RuntimeError: tensors on different devices (cpu vs cuda)</>, "model on GPU, minibatch still on CPU", <code>images, labels = images.to(device), labels.to(device)</code>],
              [<>RuntimeError: shapes cannot be multiplied (28x28 and 784x500)</>, "images are (N, 1, 28, 28); the MLP wants (N, 784)", <code>torch.flatten(images, start_dim=1)</code>],
              ["Runs, but the loss stays large and erratic", <><code>optimizer.zero_grad()</code> missing: gradients accumulate</>, "reset gradients before every backward()"],
              ["Memory grows during evaluation; warning about requires_grad", <>no <code>torch.no_grad()</code>, and loss <em>tensors</em> stored in a list keep every graph alive</>, <><code>with torch.no_grad():</code> and store <code>loss.item()</code></>],
              ["Loss spike at the end of every epoch", <>60000 = 458·131 + 2: the last minibatch has 2 examples; batch norm with B = 2 maps every feature to ±1</>, <><code>drop_last=True</code> for the <em>training</em> loader (never the test loader)</>],
            ]}
          />
          <Think q="Why does batch norm turn every feature into ±1 when the minibatch has only 2 examples?">
            With two values a and b, the mean is (a + b)/2 and the standard deviation is |a − b|/2, so (a − μ)/σ = ±1 whatever a and b
            are (ignoring ε). The normalized features carry no information, so the forward pass is nonsense and the running
            statistics get polluted.
          </Think>
          <PyCell
            code={py`
import numpy as np
z = np.array([[3.0, -7.0, 0.2],
              [5.0, 10.0, 0.1]])          # a minibatch of B = 2 samples, 3 units
print((z - z.mean(axis=0)) / z.std(axis=0))   # every entry is +-1
`}
          />
          <Callout kind="exam">
            The midterm's sample question 3 (CCE summed instead of averaged → 230.26) is exactly this kind of reasoning: read the
            code, compute what it should print, explain the discrepancy. Silent bugs are the dangerous ones.
          </Callout>
          <LabRef>the debugging tutorial (lab 5)</LabRef>
        </>
      ),
    },
  ],
};
