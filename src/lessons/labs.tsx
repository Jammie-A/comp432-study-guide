import { Callout, CodeWalk, FnRef, H, M, MB, MathCode, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { Lecture } from "./types";
import { Private } from "../private";

const t = String.raw;
const py = String.raw;

function LabRef({ children }: { children: string }) {
  return <p className="slides">Lab material: {children}</p>;
}

/** A short code snippet inside a prompt or table. */
function Pre({ children }: { children: string }) {
  return <pre className="snippet">{children.replace(/^\n/, "").replace(/\s+$/, "")}</pre>;
}

export const labs: Lecture = {
  id: "labs",
  number: 6,
  title: "Code you must be able to read",
  label: "Labs 1–5",
  short: "Labs",
  sections: [
    // ------------------------------------------------------------------ Python
    {
      id: "python",
      title: "Python you need to read the lab code",
      summary: "Functions, lambdas, tuple unpacking, loops, slices, f-strings, with-blocks and classes: every piece of Python the lab code uses.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            The lab notebooks use a handful of Python features over and over. If one of them is unfamiliar, a whole line of
            code becomes unreadable, even when you understand the math behind it. This page explains each feature once, with
            an example you can run, so the later lab pages can focus on the machine learning.
          </p>
          <H>Every piece of syntax, explained</H>
          <FnRef
            rows={[
              ["def f(a, b=2):", <>Defines a function named <code>f</code>. Nothing runs yet: the indented body runs each time you <em>call</em> <code>f(...)</code>. <code>b=2</code> is a default value, so callers may leave <code>b</code> out.</>, "—"],
              ["return a, b", <>Sends values back to the caller. Two values separated by a comma are returned together as a <strong>tuple</strong>.</>, "—"],
              ["w, b = f(...)", <><strong>Tuple unpacking</strong>: the first returned value goes into <code>w</code>, the second into <code>b</code>. The labs use this for every gradient step: <code>w, b = gd_step(...)</code>.</>, "—", "w, b = 1.0, 0.5"],
              ["lambda x: x ** 2", <>A one-line function without a name. Everything after the colon is returned. <code>sigmoid = lambda z: 1 / (1 + np.exp(-z))</code> is the same as a <code>def</code>.</>, "—", "(lambda x: x ** 2)(3)", "9"],
              ["**   //   %", <>Power, integer division (rounds down), and remainder. <code>epoch % 100 == 0</code> means "every 100th epoch".</>, "numbers", "(2 ** 3, 7 // 2, 7 % 2)", "(8, 3, 1)"],
              ["range(start, stop, step)", <>The integers from <code>start</code> up to <strong>but not including</strong> <code>stop</code>. <code>range(3)</code> is 0, 1, 2.</>, "—", "list(range(0, 10, 4))", "[0, 4, 8]"],
              ["a[-1]   a[i:j]", <>The last element; the elements <code>i</code> to <code>j − 1</code>. A slice that runs past the end is simply cut short (no error), which is why the last minibatch can be smaller.</>, "—", "[5, 6, 7][1:10]", "[6, 7]"],
              ["zip(a, b)", "Walks through two lists side by side, giving pairs.", "—", "list(zip([1, 2], [3, 4]))", "[(1, 3), (2, 4)]"],
              ["enumerate(xs)", "Gives (index, item) pairs.", "—", "list(enumerate('ab'))", "[(0, 'a'), (1, 'b')]"],
              ["[f(x) for x in xs]", <>A <strong>list comprehension</strong>: builds a list by applying <code>f</code> to every element.</>, "—", "[x * x for x in range(4)]", "[0, 1, 4, 9]"],
              ["sum(... for a, b in ...)", "Adds up the values a loop generates. Used to count the parameters of a network layer by layer.", "—", "sum((a + 1) * b for a, b in zip([784, 32], [32, 10]))", "25450"],
              ['f"{x:.4f} {n:4d}"', <>An <strong>f-string</strong>: the expressions in braces are filled in. <code>:.4f</code> = 4 decimals, <code>:4d</code> = an integer padded to width 4.</>, "string", 'f"{2/3:.4f} {7:4d}"', "0.6667    7"],
              ["for _ in range(n):", <><code>_</code> is a normal variable name that by convention means "I don't use this value".</>, "—"],
              ["f(*pair)", <>The <code>*</code> unpacks a tuple into separate arguments: <code>np.linspace(*xlim, 200)</code> is <code>np.linspace(xlim[0], xlim[1], 200)</code>.</>, "—", "np.linspace(*(0, 1), 3)", "[0.  0.5 1. ]"],
              ["xs.append(v)", <>Adds <code>v</code> to the end of the list <code>xs</code>, in place. The labs collect one loss per epoch this way.</>, "—"],
              ["with ctx:", <>Runs the indented block inside a context, and undoes it afterwards. <code>with torch.no_grad():</code> turns gradient tracking off only for that block.</>, "—"],
              ["@torch.no_grad()", <>A <strong>decorator</strong> on the line above a <code>def</code>: the whole function behaves as if its body were inside <code>with torch.no_grad():</code>.</>, "—"],
              ["assert cond, \"msg\"", <>Stops with an error showing <code>msg</code> if <code>cond</code> is false. The labs' "check" cells are asserts.</>, "—"],
              ['"""docstring"""', "Documentation for the function below the def. It does nothing when the code runs.", "—"],
              ["x.shape   x.mean()", <>Something attached to an object: an <strong>attribute</strong> (data, no parentheses) or a <strong>method</strong> (a function, called with parentheses).</>, "—"],
            ]}
          />
          <H>Classes: how <code>model(x)</code> ends up calling <code>forward</code></H>
          <p>
            PyTorch models are <strong>classes</strong>. A class is a recipe for objects that bundle data (here, the
            parameters) with functions that use it (here, <code>forward</code>). The walkthrough below builds a two-line
            stand-in for <code>torch.nn.Module</code>, so you can see the mechanism with nothing hidden.
          </p>
          <CodeWalk
            title="A tiny model class, line by line"
            code={py`
class Module:                      # a tiny stand-in for torch.nn.Module
    def __call__(self, x):         # model(x) runs this...
        return self.forward(x)     # ...which calls forward

class Line(Module):                # Line inherits __call__ from Module
    def __init__(self, w, b):      # runs once, when the object is created
        self.w, self.b = w, b      # store the parameters on the object

    def forward(self, x):
        return self.w * x + self.b

model = Line(2.0, 1.0)
print(model(3.0))
print(model.w, model.b)
`}
            steps={[
              { line: [1, 3], say: <>Defines a class <code>Module</code>. Nothing runs yet. <code>__call__</code> is a special name: Python runs it whenever you write <code>obj(...)</code> with an object of this class.</> },
              { line: 5, say: <><code>Line(Module)</code> means Line <strong>inherits</strong> from Module: it gets <code>__call__</code> for free. In PyTorch you write <code>class SimpleClassifier(nn.Module)</code> for the same reason.</> },
              { line: [6, 7], say: <><code>__init__</code> is the constructor. <code>self</code> is the object being built; <code>self.w = w</code> stores a value <em>on</em> it so other methods can use it later. (In PyTorch, <code>__init__</code> starts with <code>super().__init__()</code> and creates the layers.)</> },
              { line: [9, 10], say: <><code>forward</code> computes the output. Every method receives the object as its first argument, <code>self</code>.</> },
              { line: 12, say: <>Creating the object: Python makes an empty object and calls <code>__init__(self=that object, w=2.0, b=1.0)</code>.</>, vars: [["model.w", "2.0"], ["model.b", "1.0"]] },
              { line: 13, say: <><code>model(3.0)</code> → <code>Module.__call__(model, 3.0)</code> → <code>model.forward(3.0)</code> → <M>{t`2.0\cdot3.0+1.0`}</M>. This is exactly why the labs write <code>model(X_batch)</code> and never call <code>forward</code> directly.</>, vars: [["printed", "7.0"]] },
              { line: 14, say: "Attributes are read with a dot and no parentheses.", vars: [["printed", "2.0 1.0"]] },
            ]}
          />
          <PyCell
            title="Run it, then add a method"
            code={py`
class Module:
    def __call__(self, x):
        return self.forward(x)

class Line(Module):
    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, x):
        return self.w * x + self.b

model = Line(2.0, 1.0)
print(model(3.0))
print(model.w, model.b)
`}
          />
          <p className="muted">Try it: add a method <code>def loss(self, x, y): return (self(x) - y) ** 2</code> and predict what <code>model.loss(3.0, 5.0)</code> prints before you run it (answer: 4.0).</p>
          <H>A training loop, line by line</H>
          <p>
            Here is the loop of the linear regression tutorial, shrunk to 3 data points and 3 epochs so every value fits on the
            screen. Focus on the <em>Python</em> here (the math of the gradient is explained in the "from scratch" lab page).
          </p>
          <CodeWalk
            code={py`
import numpy as np
X = np.array([[1.0], [3.0], [5.0]])     # (N, D) = (3, 1)
Y = np.array([[0.2], [1.1], [1.7]])     # (N, K) = (3, 1)

def gd_step(X, Y, w, b, lr):
    Y_hat = w * X + b
    grad_w = 2 * ((Y_hat - Y) * X).mean()
    grad_b = 2 * (Y_hat - Y).mean()
    return w - lr * grad_w, b - lr * grad_b

w, b = 1.0, 0.5
train_loss = []
for epoch in range(3):
    w, b = gd_step(X, Y, w, b, lr=0.025)
    train_loss.append(((w * X + b - Y) ** 2).mean())
    print(f"epoch {epoch}: loss {train_loss[-1]:.4f}, w = {w:.4f}, b = {b:.4f}")
`}
            steps={[
              { line: 1, say: <>Loads NumPy under the short name <code>np</code>, so <code>np.array</code> means "NumPy's array function".</> },
              { line: [2, 3], say: <>Nested lists become 2-D arrays. Each inner list is a row: one example per row, as in the lecture notation. Note the shape is <strong>(3, 1)</strong>, a column, not (3,).</>, vars: [["X.shape", "(3, 1)"], ["Y.shape", "(3, 1)"]] },
              { line: [5, 9], say: <>Defines <code>gd_step</code>. Nothing is computed yet. The <code>X</code> and <code>Y</code> inside the function are its own parameters: they get whatever the caller passes in.</> },
              { line: 11, say: "Tuple unpacking: two assignments in one line.", vars: [["w", "1.0"], ["b", "0.5"]] },
              { line: 12, say: "An empty list that will collect one loss per epoch.", vars: [["train_loss", "[]"]] },
              { line: 13, say: <><code>range(3)</code> gives 0, 1, 2. The indented lines 14–16 run once for each value, with <code>epoch</code> set to it.</>, vars: [["epoch", "0"]] },
              { line: 14, say: <>Calls <code>gd_step</code> (passing <code>lr</code> by name, a <em>keyword argument</em>). Inside: <code>Y_hat</code> = [[1.5], [3.5], [5.5]], <code>grad_w</code> ≈ 18.333, <code>grad_b</code> = 5.0. It returns a pair, and unpacking it <strong>overwrites</strong> the old <code>w</code> and <code>b</code>.</>, vars: [["w", "0.5417 (= 1.0 − 0.025 · 18.333)"], ["b", "0.375 (= 0.5 − 0.025 · 5.0)"]] },
              { line: 15, say: <>Computes the MSE with the new parameters and appends it. <code>(...) ** 2</code> squares every entry, <code>.mean()</code> averages them.</>, vars: [["train_loss", "[1.0791]"]] },
              { line: 16, say: <>The f-string fills in the values. <code>train_loss[-1]</code> is the last entry; <code>:.4f</code> rounds to 4 decimals for display only.</>, vars: [["printed", "epoch 0: loss 1.0791, w = 0.5417, b = 0.3750"]] },
              { line: [13, 16], say: "The loop runs again for epoch = 1 and epoch = 2. The loss falls each time.", vars: [["printed", "epoch 1: loss 0.1929, w = 0.3694, b = 0.3250"], ["", "epoch 2: loss 0.0659, w = 0.3052, b = 0.3033"], ["train_loss", "[1.0791, 0.1929, 0.0659]"]] },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
X = np.array([[1.0], [3.0], [5.0]])
Y = np.array([[0.2], [1.1], [1.7]])

def gd_step(X, Y, w, b, lr):
    Y_hat = w * X + b
    grad_w = 2 * ((Y_hat - Y) * X).mean()
    grad_b = 2 * (Y_hat - Y).mean()
    return w - lr * grad_w, b - lr * grad_b

w, b = 1.0, 0.5
train_loss = []
for epoch in range(3):
    w, b = gd_step(X, Y, w, b, lr=0.025)
    train_loss.append(((w * X + b - Y) ** 2).mean())
    print(f"epoch {epoch}: loss {train_loss[-1]:.4f}, w = {w:.4f}, b = {b:.4f}")
`}
          />
          <p className="muted">Try it: change <code>lr=0.025</code> to <code>lr=0.1</code>. Predict first: does the loss still go down? (It doesn't: it blows up, because the step is too large for this data. That's Lecture 2's divergence.)</p>
          <H>Predict the output</H>
          <Think q={<>What does this print?<Pre>{py`
perm = [3, 0, 4, 1, 2]
for start in range(0, 5, 2):
    print(perm[start:start + 2])
`}</Pre></>}>
            <code>[3, 0]</code>, then <code>[4, 1]</code>, then <code>[2]</code>. <code>start</code> takes 0, 2, 4, and the last slice
            <code> perm[4:6]</code> runs past the end, so it's cut short. This is how the labs form minibatches, and why the last
            minibatch can be smaller (the cause of the debugging tutorial's fifth bug).
          </Think>
          <Think q={<>What does this print?<Pre>{py`
w, b = 1.0, 0.5
w, b = b, w
print(w, b)
`}</Pre></>}>
            <code>0.5 1.0</code>. Python evaluates the whole right-hand side first (the tuple (0.5, 1.0)), then assigns. So this
            swaps the two values.
          </Think>
          <Think q={<>What does this print?<Pre>{py`
f = lambda x: x ** 2
g = lambda x: f(x) + 1
print(g(3))
`}</Pre></>}>
            <code>10</code>: <code>g(3) = f(3) + 1 = 9 + 1</code>. Lambdas are ordinary functions and can call each other.
          </Think>
          <Think q="Why is it X.shape but X.mean() — with parentheses only on the second?">
            <code>shape</code> is an attribute (stored data), <code>mean</code> is a method (a function that computes something).
            Writing <code>X.mean</code> without parentheses doesn't compute anything; it just refers to the function.
          </Think>
          <LabRef>assignment 1 (Python warm-up and closures), the linear regression tutorial, the PyTorch tutorial (nn.Module)</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ NumPy
    {
      id: "numpy",
      title: "NumPy from zero: arrays, shapes, axis, broadcasting",
      summary: "What an array and its shape are, indexing, every NumPy function the labs call, broadcasting rules, views vs copies, and paired shuffles.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            Every model in the course is a few NumPy lines. The midterm shows you such lines and asks what they compute or
            what shape comes out. Most traps are NumPy details, not math: which axis is reduced, whether two shapes combine the
            way you think, whether a line copies or shares memory.
          </p>
          <H>An array and its shape</H>
          <p>
            A NumPy <strong>array</strong> is a grid of numbers that all have the same type. Its <strong>shape</strong> is a
            tuple with one entry per dimension (per <em>axis</em>):
          </p>
          <Table
            head={["Array", "Shape", "Think of it as"]}
            rows={[
              [<code>np.array(3.0)</code>, <code>()</code>, "a single number (a scalar)"],
              [<code>np.array([0.2, 1.1, 1.7])</code>, <code>(3,)</code>, "a vector: 3 numbers in a row, one axis"],
              [<code>np.array([[0.2], [1.1], [1.7]])</code>, <code>(3, 1)</code>, "a matrix with 3 rows and 1 column"],
              [<code>np.zeros((150, 4))</code>, <code>(150, 4)</code>, "the iris data matrix: N = 150 rows (examples), D = 4 columns (features)"],
              [<code>np.zeros((60000, 1, 28, 28))</code>, <code>(60000, 1, 28, 28)</code>, "MNIST as PyTorch wants it: (N, C, H, W)"],
            ]}
          />
          <Callout kind="mistake">
            <code>(3,)</code> and <code>(3, 1)</code> hold the same numbers but are <strong>not</strong> interchangeable. The
            linear regression tutorial keeps labels as <code>(N, K)</code> columns; the logistic tutorial keeps them as
            <code> (N,)</code> vectors. Mixing the two silently produces an (N, N) matrix (see broadcasting below).
          </Callout>
          <H>Indexing and slicing</H>
          <p>For <code>x = np.arange(12).reshape(3, 4)</code>, i.e. rows [0 1 2 3], [4 5 6 7], [8 9 10 11]:</p>
          <Table
            head={["Code", "Result", "Meaning"]}
            rows={[
              [<code>x[0, 1]</code>, <code>1</code>, "row 0, column 1 (Python counts from 0)"],
              [<code>x[0]</code>, <code>[0 1 2 3]</code>, "the whole first row (example 0)"],
              [<code>x[:, 1]</code>, <code>[1 5 9]</code>, <><code>:</code> means "all rows": column 1 (feature 1 of every example)</>],
              [<code>x[1:3]</code>, "rows 1 and 2", "a slice: start included, stop excluded"],
              [<code>x[-1, -1]</code>, <code>11</code>, "negative indices count from the end"],
              [<code>x[[0, 2]]</code>, "rows 0 and 2", <>a list of indices ("fancy indexing", used with a permutation: <code>X[perm]</code>)</>],
              [<code>x[x &gt; 8]</code>, <code>[ 9 10 11]</code>, "a boolean mask: keeps the entries where the condition is True"],
            ]}
          />
          <H>Every NumPy function the labs use</H>
          <p><strong>Making arrays and changing their shape</strong></p>
          <FnRef
            rows={[
              ["np.array(nested_list)", "Turns (nested) Python lists into an array.", "list → array", "np.array([[1, 2], [3, 4]]).shape", "(2, 2)"],
              ["np.arange(n)", <>The numbers 0, 1, …, n − 1 (like <code>range</code>). <code>np.arange(6.)</code> gives floats.</>, "→ (n,)", "np.arange(5)", "[0 1 2 3 4]"],
              ["np.linspace(a, b, n)", <><code>n</code> evenly spaced numbers from <code>a</code> to <code>b</code>, both included. Used for plotting grids.</>, "→ (n,)", "np.linspace(0, 1, 5)", "[0.   0.25 0.5  0.75 1.  ]"],
              ["np.zeros(shape), np.ones(shape)", "Arrays filled with 0 or 1. The shape is a tuple: note the double parentheses.", "→ shape", "np.zeros((2, 3)).shape", "(2, 3)"],
              ["np.ones_like(x)", <>Ones with the <em>same shape</em> as <code>x</code>. The tutorials use it to build the column of 1s for <M>{t`\tilde{\mathbf X}`}</M>.</>, "same as x", "np.ones_like(np.zeros((2, 1)))", "[[1.] [1.]]"],
              ["np.eye(K)", <>The K × K identity matrix. Row <code>k</code> is the one-hot vector of class k, so <code>np.eye(K)[labels]</code> one-hot encodes labels.</>, "→ (K, K)", "np.eye(3)[1]", "[0. 1. 0.]"],
              ["x.reshape(r, c)", <>Same numbers, new shape (the total count must match). <code>-1</code> means "work this one out".</>, "(6,) → (3, 2)", "np.arange(6).reshape(-1, 2).shape", "(3, 2)"],
              ["x.T", "Transpose: rows become columns. No copying, just a different view.", "(N, D) → (D, N)", "np.zeros((3, 2)).T.shape", "(2, 3)"],
              ["x.squeeze(axis=1)", "Removes an axis of length 1.", "(N, 1) → (N,)", "np.zeros((3, 1)).squeeze(axis=1).shape", "(3,)"],
              ["x.ravel()", "Flattens to one long vector, row by row.", "(r, c) → (r·c,)", "np.arange(6).reshape(2, 3).ravel()", "[0 1 2 3 4 5]"],
              ["np.concatenate([A, B], axis=1)", <>Glues arrays together <em>along an existing axis</em>. <code>axis=1</code>: side by side (adds columns). <code>axis=0</code>: on top of each other (adds rows).</>, "(N, 1), (N, 1) → (N, 2)", "np.concatenate([np.zeros((3, 1)), np.ones((3, 1))], axis=1).shape", "(3, 2)"],
              ["np.column_stack([a, b])", "Puts 1-D vectors side by side as columns.", "(N,), (N,) → (N, 2)", "np.column_stack([np.array([1, 3]), np.array([1, 1])])", "[[1 1] [3 1]]"],
              ["np.stack([a, b], axis=1)", <>Like <code>column_stack</code> but creates a <em>new</em> axis at the position you choose.</>, "(N,), (N,) → (N, 2)", "np.stack([np.zeros(4), np.ones(4)], axis=1).shape", "(4, 2)"],
              ["np.meshgrid(xs, ys)", <>All combinations of grid coordinates, as two matrices. The tutorials use it to colour the plane by the model's prediction.</>, "(c,), (r,) → two (r, c)", "np.meshgrid(np.arange(3), np.arange(2))[0].shape", "(2, 3)"],
              ["x.copy()", "A real copy that no longer shares memory with x.", "same as x"],
            ]}
          />
          <p><strong>Math on arrays</strong></p>
          <FnRef
            rows={[
              ["A * B,  A + B,  A - B,  A / B", <>Entry by entry (<strong>element-wise</strong>). <code>*</code> is <em>not</em> the matrix product.</>, "same shapes (or broadcast)", "np.array([1, 2]) * np.array([3, 4])", "[3 8]"],
              ["A @ B", <>The <strong>matrix product</strong> (dot products of rows of A with columns of B). The inner sizes must match.</>, "(N, D) @ (D,) → (N,)", "(np.ones((4, 3)) @ np.ones(3)).shape", "(4,)"],
              ["x.sum(), x.mean()", <>Add up or average <em>all</em> entries, giving one number.</>, "→ scalar", "np.array([[1, 2], [3, 4]]).mean()", "2.5"],
              ["x.sum(axis=a, keepdims=True)", <>Reduce along one axis only: that axis disappears, or stays with length 1 if <code>keepdims=True</code> (so the result broadcasts back).</>, "(N, K) → (N, 1)", "np.ones((4, 3)).sum(axis=1, keepdims=True).shape", "(4, 1)"],
              ["x.max(axis=1), x.argmax(axis=1)", <>The largest value in each row, or <em>its position</em>. <code>argmax</code> turns probabilities into predicted classes.</>, "(N, K) → (N,)", "np.array([[1, 5, 2], [7, 0, 3]]).argmax(axis=1)", "[1 0]"],
              ["np.exp(x), np.log(x)", <><M>{t`e^x`}</M> and the <em>natural</em> logarithm <M>{t`\ln x`}</M>, entry by entry.</>, "same as x", "np.log(np.exp(2.0))", "2.0"],
              ["np.sqrt(x), np.abs(x)", "Square root and absolute value, entry by entry.", "same as x", "np.abs(np.array([-2, 3]))", "[2 3]"],
              ["np.maximum(a, b)", <>The larger of two arrays <em>entry by entry</em>: <code>np.maximum(z, 0)</code> is the ReLU. Don't confuse with <code>np.max</code>, which finds the single largest value.</>, "same as a", "np.maximum(np.array([-1., 2.]), 0)", "[0. 2.]"],
              ["np.heaviside(z, 0)", "1 where z > 0, 0 where z < 0, and the second argument (0) where z = 0: the ReLU's derivative.", "same as z", "np.heaviside(np.array([-2., 0., 3.]), 0)", "[0. 0. 1.]"],
              ["np.outer(a, b)", <>The matrix of all products <M>{t`a_i b_j`}</M>. Used in the MLP tutorial's backward pass.</>, "(N,), (H,) → (N, H)", "np.outer(np.array([1, 2]), np.array([3, 4]))", "[[3 4] [6 8]]"],
              ["np.where(cond, a, b)", <><code>a</code> where the condition is True, <code>b</code> elsewhere.</>, "same as cond", "np.where(np.array([-1, 2]) > 0, 1, 0)", "[0 1]"],
              ["(x > 0.5).astype(float)", <>A comparison gives True/False entries; <code>astype(float)</code> turns them into 1.0/0.0. The logistic tutorial predicts classes this way.</>, "same as x", "(np.array([0.2, 0.7]) > 0.5).astype(float)", "[0. 1.]"],
              ["(a == b).mean()", "The fraction of entries that are equal (True counts as 1): this is accuracy.", "→ scalar", "(np.array([1, 0, 1, 1]) == np.array([1, 1, 1, 1])).mean()", "0.75"],
              ["np.linalg.inv(A)", <>The inverse matrix <M>{t`\mathbf A^{-1}`}</M>. The linear regression tutorial uses it for the closed form.</>, "(P, P) → (P, P)", "np.linalg.inv(np.array([[2., 0.], [0., 4.]]))", "[[0.5  0.  ] [0.   0.25]]"],
              ["np.linalg.solve(A, b)", <>Solves <M>{t`\mathbf A\boldsymbol\theta = \mathbf b`}</M> for θ without forming the inverse: more accurate. The polynomial tutorial uses it.</>, "(P, P), (P,) → (P,)", "np.linalg.solve(np.array([[2., 0.], [0., 4.]]), np.array([2., 8.]))", "[1. 2.]"],
            ]}
          />
          <p><strong>Random numbers and checks</strong></p>
          <FnRef
            rows={[
              ["rng = np.random.default_rng(42)", <>A random number generator with a fixed <strong>seed</strong> (42): every run produces the same "random" numbers, so results are reproducible.</>, "—"],
              ["rng.permutation(n)", <>The numbers 0 … n − 1 in random order. Index with it to shuffle: <code>X[perm]</code>.</>, "→ (n,)", "np.sort(np.random.default_rng(0).permutation(4))", "[0 1 2 3]"],
              ["rng.uniform(low, high, size)", "Random numbers spread evenly between low and high.", "→ size"],
              ["rng.standard_normal(shape)", "Random numbers from a Gaussian with mean 0 and standard deviation 1 (the tutorials' label noise).", "→ shape"],
              ["np.allclose(a, b)", <>True if all entries agree up to tiny rounding errors. Use it instead of <code>==</code> for floats.</>, "→ bool", "np.allclose(0.1 + 0.2, 0.3)", "True"],
            ]}
          />
          <H>Broadcasting: when shapes don't match</H>
          <p>
            For element-wise operations, NumPy can stretch an array along an axis of length 1, so <code>X - X.mean(axis=0)</code>{" "}
            subtracts each column's mean from every row. The rule:
          </p>
          <Steps
            title="How NumPy decides whether two shapes combine"
            steps={[
              { title: "Line the shapes up from the right", body: <p>(3, 2) and (2,) become (3, 2) and (_, 2). A missing axis counts as length 1.</p> },
              { title: "Compare axis by axis", body: <p>Each pair must be equal, or one of them must be 1. (3 vs 1) ✓, (2 vs 2) ✓.</p> },
              { title: "Stretch the 1s", body: <p>The length-1 axis is repeated to match: the (2,) vector is used for each of the 3 rows. Result: (3, 2).</p> },
              { title: "The trap: (3, 1) − (3,)", body: <p>Right-aligned: (3, 1) and (_, 3). Both pairs are compatible (3 vs 1, 1 vs 3), so <strong>both</strong> axes stretch and you get a <strong>(3, 3)</strong> matrix of every prediction minus every label. No error, just a wrong loss.</p> },
            ]}
          />
          <CodeWalk
            title="Building X̃, predicting, and the broadcasting trap"
            code={py`
import numpy as np
x = np.array([1.0, 3.0, 5.0])        # 3 petal lengths
X = x.reshape(-1, 1)                 # a data matrix: one row per example
X_tilde = np.concatenate([X, np.ones_like(X)], axis=1)
theta = np.array([0.3, 0.1])         # (w, b)
y_hat = X_tilde @ theta
y = np.array([0.2, 1.1, 1.7])
r = y_hat - y
bad = X_tilde @ theta.reshape(-1, 1) - y
print(y_hat, r, bad.shape)
`}
            steps={[
              { line: 2, say: "A plain vector of 3 numbers.", vars: [["x.shape", "(3,)"]] },
              { line: 3, say: <><code>reshape(-1, 1)</code>: "one column, as many rows as needed". Now it's a data matrix with N = 3, D = 1.</>, vars: [["X", "[[1.], [3.], [5.]]"], ["X.shape", "(3, 1)"]] },
              { line: 4, say: <><code>np.ones_like(X)</code> is a (3, 1) column of ones; <code>axis=1</code> glues it on as a second column. This is <M>{t`\tilde{\mathbf X}`}</M>.</>, vars: [["X_tilde", "[[1., 1.], [3., 1.], [5., 1.]]"], ["X_tilde.shape", "(3, 2)"]] },
              { line: 5, say: "The parameter vector: slope first, bias last (it multiplies the column of ones).", vars: [["theta.shape", "(2,)"]] },
              { line: 6, say: <>Matrix product (3, 2) @ (2,) → (3,). Row n gives <M>{t`w x_n + b\cdot1`}</M>: 0.3·1 + 0.1, 0.3·3 + 0.1, 0.3·5 + 0.1.</>, vars: [["y_hat", "[0.4, 1.0, 1.6]"]] },
              { line: [7, 8], say: "Labels as a (3,) vector too, so the subtraction is entry by entry: the residuals.", vars: [["r", "[0.2, -0.1, -0.1]"], ["r.shape", "(3,)"]] },
              { line: 9, say: <>Same idea with θ as a (2, 1) column: the prediction is now (3, 1), and (3, 1) − (3,) broadcasts to (3, 3). Squaring and averaging that would give a meaningless loss, with no error message.</>, vars: [["bad.shape", "(3, 3)"]] },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
x = np.array([1.0, 3.0, 5.0])
X = x.reshape(-1, 1)
X_tilde = np.concatenate([X, np.ones_like(X)], axis=1)
theta = np.array([0.3, 0.1])
y_hat = X_tilde @ theta
y = np.array([0.2, 1.1, 1.7])
r = y_hat - y
bad = X_tilde @ theta.reshape(-1, 1) - y
print(y_hat, r, bad.shape)
print(bad)
`}
          />
          <H>Views vs copies</H>
          <p>
            Some operations give a <strong>view</strong>: a new array object that looks at the <em>same memory</em>. Writing into a
            view changes the original.
          </p>
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
            Replace Python loops by array operations: they run much faster and are shorter. Lab 1's example: subtract the
            minimum from every entry. The double loop becomes <code>x - x.min()</code>; the tutorials' grid evaluation becomes{" "}
            <code>np.meshgrid</code>; every model in the course becomes one matrix product (<code>X_tilde @ theta</code>).
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
          <H>Predict the output</H>
          <Think q={<>What's the shape?<Pre>{py`
X = np.ones((5, 3))
print(X.mean(axis=1, keepdims=True).shape)
`}</Pre></>}>
            <code>(5, 1)</code>: axis 1 (the 3 columns) is reduced, and <code>keepdims</code> keeps it as length 1.
          </Think>
          <Think q={<>What does this print?<Pre>{py`
x = np.arange(6).reshape(2, 3)
print(x[:, -1])
`}</Pre></>}>
            <code>[2 5]</code>: the rows are [0 1 2] and [3 4 5]; <code>[:, -1]</code> takes the last column of every row.
          </Think>
          <Think q={<>What does this print?<Pre>{py`
print(np.ones((2, 3)) @ np.ones(3))
`}</Pre></>}>
            <code>[3. 3.]</code>: each row of ones dotted with three ones gives 3; (2, 3) @ (3,) → (2,).
          </Think>
          <Think q={<><code>y_hat</code> has shape (4, 1) and <code>y</code> has shape (4,). What is the shape of <code>((y_hat - y) ** 2).mean()</code>'s input, and is the loss right?</>}>
            <code>y_hat - y</code> broadcasts to <strong>(4, 4)</strong>, so the "MSE" averages 16 differences of every prediction
            with every label. It's a number, it's wrong, and there's no error. Fix: <code>y_hat.squeeze(axis=1)</code> or
            <code> y.reshape(-1, 1)</code>.
          </Think>
          <LabRef>assignment 1 (exercises 2.1–2.4), the NumPy tutorial, the linear and logistic regression tutorials</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ Finite differences
    {
      id: "finite-differences",
      title: "Finite differences: checking gradients numerically",
      summary: "Approximate a derivative with two function evaluations: the closure that does it, why the error is ε², and why ε can be too large or too small.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            A finite difference turns the definition of the derivative into two function calls. It's the standard way to check
            a gradient you derived by hand (Lab 1 exercise 1.4, Lab 2 exercise 1.2): if your formula and the numerical estimate
            disagree, the formula (or its code) is wrong.
          </p>
          <MB>{t`f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}\qquad\text{(central difference)}`}</MB>
          <p>
            In words: step a tiny amount ε to the right and to the left, see how much f changes, and divide by the distance
            between the two points (2ε). That's the slope of the line through two nearby points of the curve.
          </p>
          <H>The code, line by line</H>
          <CodeWalk
            code={py`
def approx_derivative_of(f, epsilon=1e-6):
    return lambda x: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

f = lambda x: x**3
df = approx_derivative_of(f, 1.0)
print(df(2.0))
`}
            steps={[
              { line: 1, say: <>A function that takes a <em>function</em> <code>f</code> as its argument. In Python, functions are values you can pass around like numbers.</> },
              { line: 2, say: <>It doesn't compute a number: it <strong>returns a new function</strong> (a lambda of x). That lambda remembers <code>f</code> and <code>epsilon</code> from the moment it was made. A function that remembers variables like this is called a <strong>closure</strong>.</> },
              { line: 4, say: <>The function to differentiate: <M>{t`f(x) = x^3`}</M>, whose exact derivative is <M>{t`3x^2`}</M>.</> },
              { line: 5, say: <>Calls <code>approx_derivative_of</code> with ε = 1.0. <code>df</code> is now a function, not a number.</>, vars: [["df", "a function of x (with f = x³ and epsilon = 1.0 baked in)"]] },
              { line: 6, say: <>Now the lambda runs with x = 2: <M>{t`\frac{f(3) - f(1)}{2} = \frac{27 - 1}{2} = 13`}</M>. The exact answer is <M>{t`3\cdot2^2 = 12`}</M>, so ε = 1 is far too large.</>, vars: [["printed", "13.0"]] },
            ]}
          />
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
          <H>Where the error ε² comes from</H>
          <Steps
            title="Central difference of f(x) = x³, worked out exactly"
            steps={[
              { title: "Expand f(x + ε)", body: <MB>{t`(x+\epsilon)^3 = x^3 + 3x^2\epsilon + 3x\epsilon^2 + \epsilon^3`}</MB> },
              { title: "Expand f(x − ε)", body: <MB>{t`(x-\epsilon)^3 = x^3 - 3x^2\epsilon + 3x\epsilon^2 - \epsilon^3`}</MB> },
              { title: "Subtract: the even powers cancel", body: <MB>{t`f(x+\epsilon) - f(x-\epsilon) = 6x^2\epsilon + 2\epsilon^3`}</MB> },
              { title: "Divide by 2ε", body: <MB>{t`\frac{6x^2\epsilon + 2\epsilon^3}{2\epsilon} = 3x^2 + \epsilon^2`}</MB> },
              { title: "Read it off", body: <p>The estimate is the true derivative <M>{t`3x^2`}</M> plus an error of exactly <M>{t`\epsilon^2`}</M>. With ε = 1 that's 12 + 1 = 13 ✓. Halving ε divides the error by 4.</p> },
            ]}
          />
          <Callout kind="note">
            The <strong>forward</strong> difference <M>{t`\frac{f(x+\epsilon) - f(x)}{\epsilon}`}</M> uses one step to the right only.
            Its error shrinks only like ε (for x³ it's <M>{t`3x\epsilon + \epsilon^2`}</M>), so the central difference is more
            accurate for the same ε.
          </Callout>
          <H>Why a tiny ε fails</H>
          <p>
            A float64 stores about 16 significant digits. With ε = 10⁻¹⁵, <M>{t`f(2+\epsilon)`}</M> and <M>{t`f(2-\epsilon)`}</M> are
            both ≈ 8.000000000000000 and differ only in the last one or two digits, which are mostly rounding noise. Subtracting
            them leaves almost nothing correct (<strong>cancellation</strong>), and dividing by the tiny <M>{t`2\epsilon`}</M> blows
            that noise up. The total error is the approximation error (shrinks with ε) plus the rounding error (grows as ε
            shrinks), so there is a best ε in between, around 10⁻⁵ here.
          </p>
          <Callout kind="exam">
            Expect "which ε gives the best estimate" or "why is the estimate with ε = 10⁻¹⁵ wrong?" — floating-point
            cancellation, not a math error.
          </Callout>
          <H>Predict the output</H>
          <Think q={<>What does this print?<Pre>{py`
d = approx_derivative_of(lambda x: x**2, 0.5)
print(d(1.0))
`}</Pre></>}>
            <code>2.0</code>: <M>{t`\frac{1.5^2 - 0.5^2}{1} = 2.25 - 0.25 = 2`}</M>, the exact derivative. For a quadratic the
            central difference has no approximation error at all (the error term involves the third derivative, which is 0).
          </Think>
          <Think q="Your hand-derived gradient gives 0.37 at some point, the central difference with ε = 1e-6 gives 0.74. What's the most likely explanation?">
            A factor 2 is missing in your formula (or its code): the estimates differ by exactly ×2, far more than finite-difference
            error at ε = 10⁻⁶. The classic case is forgetting the 2 in the MSE gradient <M>{t`\frac2N\tilde{\mathbf X}^\top\mathbf r`}</M>.
          </Think>
          <LabRef>assignment 1 (exercise 1.4), assignment 2 (exercise 1.2)</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ From scratch
    {
      id: "from-scratch",
      title: "Linear & logistic regression from scratch (+ scikit-learn)",
      summary: "The tutorials' gd_step, closed form, sigmoid, softmax, BCE/CCE and accuracy functions line by line, and what each scikit-learn call does.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            The Lab 3 tutorials implement Lecture 3's models in a few NumPy lines each. Every line is a formula from the
            slides. Here you go through each function with numbers small enough to check by hand, then see the scikit-learn
            calls that do the same thing for you.
          </p>
          <H>Linear regression: one gradient step</H>
          <p>
            The tutorial's model is <M>{t`\hat y = wx + b`}</M>, and the gradient of the MSE is
            <M>{t`\ \frac{\partial J}{\partial w} = \frac2N\sum_n(\hat y_n - y_n)x_n`}</M>, <M>{t`\frac{\partial J}{\partial b} = \frac2N\sum_n(\hat y_n - y_n)`}</M>.
            A sum divided by N is a mean, which is how the code writes it.
          </p>
          <CodeWalk
            title="gd_step from the linear regression tutorial, on Lecture 3's mini iris data"
            code={py`
def gd_step(X, Y, w, b, lr):
    Y_hat = w * X + b
    grad_w = 2 * ((Y_hat - Y) * X).mean()
    grad_b = 2 * (Y_hat - Y).mean()
    return w - lr * grad_w, b - lr * grad_b

X = np.array([[1.0], [3.0], [5.0]])
Y = np.array([[0.2], [1.1], [1.7]])
w, b = gd_step(X, Y, 1.0, 0.5, lr=0.025)
`}
            steps={[
              { line: 9, say: "We call gd_step with w = 1.0, b = 0.5 and learning rate 0.025. Follow the values inside the function." },
              { line: 2, say: <>Prediction for all examples at once: <code>w * X</code> multiplies every entry by w, <code>+ b</code> adds b to every entry.</>, vars: [["Y_hat", "[[1.5], [3.5], [5.5]]"]] },
              { line: 3, say: <><code>Y_hat - Y</code> is the error of each example; <code>* X</code> multiplies each error by its input, entry by entry; <code>.mean()</code> divides the sum by N = 3. This is <M>{t`\frac2N\sum_n(\hat y_n - y_n)x_n`}</M>.</>, vars: [["Y_hat - Y", "[[1.3], [2.4], [3.8]]"], ["(Y_hat - Y) * X", "[[1.3], [7.2], [19.0]]"], ["grad_w", "2 · 27.5 / 3 ≈ 18.333"]] },
              { line: 4, say: <>Same without the input: <M>{t`\frac2N\sum_n(\hat y_n - y_n)`}</M>. Both gradients are positive: all predictions are too high, so w and b must decrease.</>, vars: [["grad_b", "2 · 7.5 / 3 = 5.0"]] },
              { line: 5, say: <>The gradient-descent update <M>{t`\theta \leftarrow \theta - \eta\nabla J`}</M> for both parameters, returned as a tuple.</>, vars: [["new w", "1.0 − 0.025 · 18.333 ≈ 0.5417"], ["new b", "0.5 − 0.025 · 5.0 = 0.375"]] },
            ]}
          />
          <H>Linear regression: the closed form</H>
          <p>
            The tutorial then computes the exact minimizer <M>{t`\theta^* = (\tilde{\mathbf X}^\top\tilde{\mathbf X})^{-1}\tilde{\mathbf X}^\top\mathbf y`}</M> (Lecture 3) in one line:
          </p>
          <CodeWalk
            code={py`
X_tilde = np.concatenate([X, np.ones_like(X)], axis=1)
A = X_tilde.T @ X_tilde
theta_star = (np.linalg.inv(A) @ X_tilde.T @ Y).squeeze(axis=1)
`}
            steps={[
              { line: 1, say: <>Append the column of ones, so the bias becomes the last entry of θ.</>, vars: [["X_tilde", "[[1, 1], [3, 1], [5, 1]]  (3, 2)"]] },
              { line: 2, say: <>(2, 3) @ (3, 2) → (2, 2). Entry [0, 0] is <M>{t`1^2+3^2+5^2 = 35`}</M>, the off-diagonal is <M>{t`1+3+5 = 9`}</M>, entry [1, 1] counts the examples: 3.</>, vars: [["A", "[[35, 9], [9, 3]]"]] },
              { line: 3, say: <>Read right to left: <code>X_tilde.T @ Y</code> is (2, 1) = [[1·0.2 + 3·1.1 + 5·1.7], [0.2 + 1.1 + 1.7]] = [[12], [3]]. Multiply by the inverse of A. <code>squeeze(axis=1)</code> turns the (2, 1) result into a (2,) vector.</>, vars: [["X_tilde.T @ Y", "[[12.], [3.]]"], ["np.linalg.inv(A)", "[[0.125, -0.375], [-0.375, 1.4583]]"], ["theta_star", "[0.375, -0.125]   (w, b)"]] },
            ]}
          />
          <Callout kind="note">
            Gradient descent gets close to <code>theta_star</code> but never lands exactly on it, so the tutorial compares the
            two with <code>np.allclose(..., rtol=1e-4)</code>, not <code>==</code>. Logistic regression has no closed form, so
            there gradient descent is the only option.
          </Callout>
          <H>Logistic regression: model, loss, gradient, accuracy</H>
          <FnRef
            rows={[
              ["sigmoid(z)", <><code>1 / (1 + np.exp(-z))</code>: squashes any number into (0, 1), entry by entry.</>, "same as z", "1 / (1 + np.exp(-np.array([0.0, 2.0])))", "[0.5        0.88079708]"],
              ["logistic_model(X_tilde, theta)", <><code>sigmoid(X_tilde @ theta)</code>: the probability of class 1 for every example.</>, "(N, D+1), (D+1,) → (N,)"],
              ["bce(y, y_hat)", <><code>-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()</code>. For each example only one of the two terms is non-zero, because y is 0 or 1.</>, "(N,), (N,) → scalar"],
              ["accuracy(y, y_hat)", <><code>((y_hat &gt; 0.5).astype(float) == y).mean()</code>: predict class 1 above 0.5, then the fraction of correct predictions.</>, "(N,), (N,) → scalar"],
              ["gd_step(X_tilde, y, theta, lr)", <><code>grad = X_tilde.T @ (y_hat - y) / y.size</code>: error times input, averaged. No factor 2 (unlike MSE).</>, "→ (D+1,)"],
            ]}
          />
          <CodeWalk
            title="One logistic-regression gradient step from θ = 0"
            code={py`
X_tilde = np.array([[0.0, 1], [1.0, 1], [3.0, 1], [4.0, 1]])
y = np.array([0.0, 0.0, 1.0, 1.0])
theta = np.zeros(2)
y_hat = 1 / (1 + np.exp(-(X_tilde @ theta)))
grad = X_tilde.T @ (y_hat - y) / y.size
theta = theta - 0.5 * grad
`}
            steps={[
              { line: [1, 2], say: "Four 1-D examples (already augmented with the 1). Small x are class 0, large x are class 1.", vars: [["X_tilde.shape", "(4, 2)"], ["y.shape", "(4,)"]] },
              { line: 3, say: <><code>np.zeros(2)</code>: w = 0, b = 0. The model can't tell anything apart yet.</>, vars: [["theta", "[0., 0.]"]] },
              { line: 4, say: <>All logits are 0, and <M>{t`\sigma(0) = 0.5`}</M>: every example gets probability 0.5. The BCE is <M>{t`-\ln 0.5 \approx 0.693`}</M>.</>, vars: [["X_tilde @ theta", "[0., 0., 0., 0.]"], ["y_hat", "[0.5, 0.5, 0.5, 0.5]"]] },
              { line: 5, say: <>Errors <code>y_hat - y</code> = [0.5, 0.5, −0.5, −0.5]. <code>X_tilde.T</code> has shape (2, 4): row 0 holds the x values, row 1 the ones. So entry 0 is <M>{t`(0\cdot0.5 + 1\cdot0.5 - 3\cdot0.5 - 4\cdot0.5)/4 = -0.75`}</M> and entry 1 is <M>{t`(0.5+0.5-0.5-0.5)/4 = 0`}</M>.</>, vars: [["y_hat - y", "[0.5, 0.5, -0.5, -0.5]"], ["grad", "[-0.75, 0.]"]] },
              { line: 6, say: <>The weight increases (the gradient is negative), so larger x now gets a higher probability of class 1. New predictions: [0.5, 0.593, 0.755, 0.818], the accuracy is 0.75 (x = 0 sits exactly at 0.5 and counts as class 0), and the BCE dropped from 0.693 to 0.518.</>, vars: [["theta", "[0.375, 0.]"]] },
            ]}
          />
          <H>Multiclass: softmax, CCE, accuracy</H>
          <CodeWalk
            title="The multiclass tutorial's softmax and cross-entropy"
            code={py`
def softmax(Z):
    e_Z = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e_Z / e_Z.sum(axis=1, keepdims=True)

Z = np.array([[2.0, 1.0, 0.0], [1000.0, 1001.0, 999.0]])
P = softmax(Z)
Y = np.array([[1, 0, 0], [0, 0, 1]])
cce = -(Y * np.log(P)).sum(axis=1).mean()
acc = (P.argmax(axis=1) == Y.argmax(axis=1)).mean()
`}
            steps={[
              { line: 5, say: "Logits for N = 2 examples and K = 3 classes, one row per example. The second row has huge numbers.", vars: [["Z.shape", "(2, 3)"]] },
              { line: 2, say: <><code>Z.max(axis=1, keepdims=True)</code> is each row's largest logit, kept as a (2, 1) column so it broadcasts back over the row. Subtracting it makes the biggest entry 0, so <code>np.exp</code> can't overflow. Without it, <M>{t`e^{1000}`}</M> is infinity and the second row becomes NaN.</>, vars: [["Z.max(axis=1, keepdims=True)", "[[2.], [1001.]]"], ["Z - max", "[[0, -1, -2], [-1, 0, -2]]"], ["e_Z", "[[1, 0.368, 0.135], [0.368, 1, 0.135]]"]] },
              { line: 3, say: <>Divide every row by its own sum (1.503 for both rows), so each row adds up to 1. Subtracting the max changed nothing: softmax only depends on <em>differences</em> of logits.</>, vars: [["P", "[[0.665, 0.245, 0.090], [0.245, 0.665, 0.090]]"]] },
              { line: 7, say: "One-hot labels: example 0 is class 0, example 1 is class 2.", vars: [["Y.shape", "(2, 3)"]] },
              { line: 8, say: <><code>Y * np.log(P)</code> keeps only the log-probability of the true class in each row; <code>.sum(axis=1)</code> collects it, <code>.mean()</code> averages over examples. CCE = mean of <M>{t`-\ln 0.665`}</M> and <M>{t`-\ln 0.090`}</M>.</>, vars: [["per example", "[0.408, 2.408]"], ["cce", "1.408"]] },
              { line: 9, say: <><code>argmax(axis=1)</code> turns each row into a class index. Predicted [0, 1], true [0, 2]: one of two correct.</>, vars: [["P.argmax(axis=1)", "[0, 1]"], ["acc", "0.5"]] },
            ]}
          />
          <p>
            The tutorial's multiclass gradient step is <code>grad = (Y_hat - Y).T @ X_tilde / len(Y)</code>. Check the shape:
            (K, N) @ (N, D+1) → (K, D+1), the shape of <code>W</code>, with one row per class.
          </p>
          <PyCell
            title="All the tutorial functions together"
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
          <p className="muted">Try it: change 3000 to 10. Predict first: is (w, b) already close to (0.375, −0.125)? (No: after 10 steps it's about (0.13, 0.79). The bias starts far too high, so GD first pulls w down to compensate, and needs many more steps to sort out the two.)</p>
          <Private id="labs/from-scratch/a3" quiet />
          <H>scikit-learn, call by call</H>
          <FnRef
            rows={[
              ["LinearRegression()", <>Creates an (untrained) least-squares model. By default it adds its own intercept, so pass <code>X</code> <em>without</em> the column of ones (or use <code>fit_intercept=False</code>).</>, "—"],
              [".fit(X, Y)", <>Trains the model (here: solves the closed form) and returns the model itself, so you can write <code>model = LinearRegression().fit(X, Y)</code>.</>, "X (N, D), Y (N,) or (N, K)"],
              [".coef_, .intercept_", <>The learned weights and bias. With Y of shape (N, K): <code>coef_</code> is (K, D), so the tutorial reads <code>coef_[0, 0]</code>.</>, "(K, D), (K,)"],
              [".predict(X)", "Predicted values (regression) or predicted classes (classification).", "(N, D) → (N,)"],
              ["LogisticRegression(C=np.inf)", <>Logistic regression. The default <code>C=1</code> adds an L2 penalty; <code>C=np.inf</code> switches it off so it matches plain BCE. Smaller C = stronger penalty (<M>{t`\lambda = 1/(CN)`}</M>).</>, "—"],
              [".fit(X, y) (classifier)", <>Needs integer class labels. The multiclass tutorial converts one-hot labels with <code>Y_train.argmax(axis=1)</code>.</>, "X (N, D), y (N,)"],
              [".predict_proba(X)", <>One probability per class. For binary problems, column 1 (<code>[:, 1]</code>) is the probability of class 1.</>, "(N, D) → (N, K)"],
              [".score(X, y)", <>Accuracy for a classifier, but <strong>R²</strong> (not MSE) for a regressor. The error rate in percent is (1 − accuracy) · 100.</>, "→ scalar"],
              ["MinMaxScaler().fit(X_train)", <>Learns each feature's min and max <em>on the training set</em>; <code>.transform(X)</code> rescales to [0, 1]; <code>.fit_transform</code> does both.</>, "(N, D) → (N, D)"],
            ]}
          />
          <PyCell
            title="scikit-learn agrees with the closed form (first run loads scikit-learn, give it a few seconds)"
            code={py`
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.0], [3.0], [5.0]])          # no column of ones: sklearn adds the intercept
Y = np.array([[0.2], [1.1], [1.7]])
model = LinearRegression().fit(X, Y)
print("coef_", model.coef_.round(4), "shape", model.coef_.shape)
print("intercept_", model.intercept_.round(4))
print("predict", model.predict(np.array([[2.0]])).round(4))
`}
          />
          <Table
            head={["Setting", "Meaning"]}
            rows={[
              [<code>LinearRegression(fit_intercept=False)</code>, "use it when X_tilde already has the column of ones; otherwise sklearn adds its own intercept"],
              [<code>LogisticRegression()</code>, <>default <code>C=1</code> <strong>regularizes</strong> (L2 on the weights, not the bias). It does not minimize plain BCE.</>],
              [<><code>C</code></>, <>the <strong>inverse</strong> regularization strength: smaller C = stronger penalty. Equivalent to <M>{t`\lambda = 1/(CN)`}</M>. <code>C=np.inf</code> = no penalty.</>],
              [<code>model.score(X, y)</code>, "accuracy for classifiers; the error rate in percent is (1 − score) · 100"],
              [<code>coef_</code>, "LogisticRegression: shape (K, D), one row per class, like our W"],
            ]}
          />
          <Private id="labs/from-scratch/a3-sklearn" />
          <Callout kind="note">
            Lab 3's last remark: on <strong>linearly separable</strong> data, unregularized logistic regression has no minimum —
            the loss keeps decreasing as the weights grow, so solvers stop only at their tolerance (sklearn's got weights ≈ 10 and 17).
            That's one reason regularization exists.
          </Callout>
          <H>Predict the output</H>
          <Think q="In gd_step for linear regression, what would happen if you wrote (Y_hat - Y).mean() * X.mean() instead of ((Y_hat - Y) * X).mean()?">
            It's a different number: the mean of a product isn't the product of the means. Here it would be 2 · 2.5 · 3 = 15 instead
            of 18.33. The gradient needs each example's error multiplied by <em>that example's</em> input before averaging.
          </Think>
          <Think q={<>What's the shape of <code>grad</code> in the multiclass step, with N = 150 examples, D = 4 features and K = 3 classes?</>}>
            <code>(Y_hat - Y).T</code> is (3, 150), <code>X_tilde</code> is (150, 5), so the gradient is <strong>(3, 5)</strong>: one row
            of D + 1 = 5 parameters per class, the shape of W.
          </Think>
          <Think q={<>What does <code>softmax(np.array([[0.0, 0.0]]))</code> return?</>}>
            <code>[[0.5 0.5]]</code>: equal logits give equal probabilities. The same holds for <code>[[7, 7]]</code> or <code>[[-3, -3]]</code>,
            because only differences matter.
          </Think>
          <Think q="Which loss function do you get when you apply the CCE code to K = 2 classes with one-hot labels?">
            Exactly the BCE: for the true class the term is <M>{t`-\ln\hat y`}</M>, and the other class's probability is 1 − ŷ. Binary
            logistic regression is the K = 2 case of multiclass logistic regression.
          </Think>
          <LabRef>assignment 3, tutorials on linear, logistic and multiclass logistic regression, and polynomial fitting</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ PyTorch basics
    {
      id: "pytorch-basics",
      title: "PyTorch tensors & autograd",
      summary: "What a tensor is, dtypes, the tensor functions the labs use, how backward() fills .grad, why gradients add up, and nn.Module/nn.Linear.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            PyTorch is NumPy plus two things: it can run on a GPU, and it computes gradients for you (<strong>autograd</strong>).
            The PyTorch tutorial of Lab 4 introduces both. PyTorch can't run in this browser, so these cells are read-only, but
            every printed value below was checked by running the code.
          </p>
          <H>Tensors</H>
          <p>
            A <strong>tensor</strong> is PyTorch's array: a vector is a 1-D tensor, a matrix a 2-D tensor, a batch of images a
            4-D tensor. Almost every NumPy function has a tensor twin, with one naming difference: PyTorch says{" "}
            <code>dim=</code> where NumPy says <code>axis=</code>.
          </p>
          <FnRef
            rows={[
              ["torch.tensor(list)", "Makes a tensor from (nested) lists. Integers give int64, decimals give float32.", "list → tensor", "torch.tensor([[1, 2], [3, 4]]).shape", "torch.Size([2, 2])"],
              ["t.dtype", <>The number type. Models compute in <strong>float32</strong>; class labels for <code>CrossEntropyLoss</code> must be <strong>int64</strong> ("long").</>, "—", "torch.tensor([1, 2]).dtype", "torch.int64"],
              ["torch.zeros/ones/rand/randn(shape)", <>Filled with 0, 1, uniform random numbers in [0, 1), or Gaussian random numbers. (<code>torch.empty</code> leaves the memory as it was, so its values are garbage.)</>, "→ shape", "torch.zeros(2, 3).shape", "torch.Size([2, 3])"],
              ["torch.arange(n)", <>Like <code>np.arange</code>.</>, "→ (n,)", "torch.arange(4)", "tensor([0, 1, 2, 3])"],
              ["torch.from_numpy(a)", <>NumPy array → tensor (sharing memory). NumPy's float64 stays float64, so the labs add <code>.float()</code>.</>, "array → tensor", "torch.from_numpy(np.array([1., 2.]))", "tensor([1., 2.], dtype=torch.float64)"],
              ["t.float(), t.long()", "Convert to float32 or int64. long() cuts off the decimals.", "same shape", "torch.tensor([0.7, 1.2]).long()", "tensor([0, 1])"],
              ["t.numpy()", <>Tensor → NumPy. Needs <code>.detach()</code> first if the tensor tracks gradients, and <code>.cpu()</code> first if it's on a GPU.</>, "tensor → array"],
              ["t.view(r, c)", <>Like <code>reshape</code>.</>, "(6,) → (2, 3)", "torch.arange(6).view(2, 3)", "tensor([[0, 1, 2], [3, 4, 5]])"],
              ["t.permute(...)", "Reorders the axes: permute(1, 0) swaps two axes (a transpose); permute(2, 0, 1) moves the last axis first.", "(H, W, C) → (C, H, W)", "torch.zeros(10, 12, 1).permute(2, 0, 1).shape", "torch.Size([1, 10, 12])"],
              ["t.unsqueeze(d), t.squeeze(dim=d)", "Insert an axis of length 1 at position d, or remove one.", "(N, 1) → (N,)", "torch.zeros(5, 1).squeeze(dim=1).shape", "torch.Size([5])"],
              ["torch.flatten(t, start_dim=1)", "Merge all axes from start_dim on into one, keeping the batch axis.", "(N, 1, 28, 28) → (N, 784)", "torch.flatten(torch.zeros(8, 1, 28, 28), start_dim=1).shape", "torch.Size([8, 784])"],
              ["torch.matmul(a, b), a @ b", "Matrix product, as in NumPy.", "(2, 3) @ (3, 3) → (2, 3)"],
              ["torch.argmax(t, dim=1)", "Position of the largest entry in each row: the predicted class.", "(N, K) → (N,)", "torch.argmax(torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 0.5]]), dim=1)", "tensor([1, 0])"],
              ["torch.sigmoid(t)", "The sigmoid, entry by entry.", "same as t", "torch.sigmoid(torch.tensor(0.0))", "tensor(0.5000)"],
              ["t.item()", <>A one-number tensor → a plain Python float. Use it to store losses: a stored <em>tensor</em> keeps its whole computation graph alive.</>, "→ float", "torch.tensor(3.5).item()", "3.5"],
              ["x.add_(y)", <>A trailing underscore means <strong>in place</strong>: it overwrites x instead of making a new tensor.</>, "—"],
              ["torch.manual_seed(0)", "Fixes PyTorch's random numbers (initial weights, shuffles), so runs are reproducible.", "—"],
              ["t.to(device)", <>Moves a tensor (or a model) to the CPU or GPU. Inputs and model must be on the <em>same</em> device.</>, "—"],
            ]}
          />
          <H>Autograd: how <code>backward()</code> finds gradients</H>
          <p>
            When a tensor has <code>requires_grad=True</code>, PyTorch records every operation done with it: the
            <strong> computation graph</strong>. Calling <code>.backward()</code> on the final number walks that graph backwards
            with the chain rule (backpropagation, Lecture 4) and stores <M>{t`\partial y/\partial x`}</M> in <code>x.grad</code>.
          </p>
          <CodeWalk
            title="The PyTorch tutorial's computation graph, line by line"
            code={py`
x = torch.arange(3, dtype=torch.float32, requires_grad=True)
a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
y.backward()
print(x.grad)
`}
            steps={[
              { line: 1, say: <>x = [0, 1, 2]. <code>requires_grad=True</code> says "record what happens to x". Only float tensors can have gradients, hence <code>dtype=torch.float32</code>.</>, vars: [["x", "tensor([0., 1., 2.], requires_grad=True)"]] },
              { line: 2, say: <>Each operation creates a new tensor <em>and</em> a node in the graph. The node's <code>grad_fn</code> knows how to send gradients back to its input.</>, vars: [["a", "tensor([2., 3., 4.], grad_fn=<AddBackward0>)"]] },
              { line: 3, say: "Squares every entry.", vars: [["b", "tensor([4., 9., 16.], grad_fn=<PowBackward0>)"]] },
              { line: 4, say: "Adds 3 to every entry.", vars: [["c", "tensor([7., 12., 19.], grad_fn=<AddBackward0>)"]] },
              { line: 5, say: <>The mean of the three entries: (7 + 12 + 19)/3 = 38/3. This is the single number we differentiate.</>, vars: [["y", "tensor(12.6667, grad_fn=<MeanBackward0>)"]] },
              { line: 6, say: <>Backpropagation through the graph, from y back to x. Chain rule per entry: <M>{t`\frac{\partial y}{\partial c_i} = \frac13`}</M>, <M>{t`\frac{\partial c_i}{\partial b_i} = 1`}</M>, <M>{t`\frac{\partial b_i}{\partial a_i} = 2a_i`}</M>, <M>{t`\frac{\partial a_i}{\partial x_i} = 1`}</M>, so <M>{t`\frac{\partial y}{\partial x_i} = \frac23(x_i + 2)`}</M>.</> },
              { line: 7, say: <><M>{t`\frac23\cdot(2, 3, 4) = (\frac43, 2, \frac83)`}</M>.</>, vars: [["x.grad", "tensor([1.3333, 2.0000, 2.6667])"]] },
            ]}
          />
          <H>Gradients add up: why <code>zero_grad()</code> exists</H>
          <PyCell
            readOnly
            title="Calling backward twice (read-only; the outputs are from a real run)"
            code={py`
x = torch.arange(3, dtype=torch.float32, requires_grad=True)
((x + 2) ** 2 + 3).mean().backward()
print(x.grad)          # tensor([1.3333, 2.0000, 2.6667])
((x + 2) ** 2 + 3).mean().backward()
print(x.grad)          # tensor([2.6667, 4.0000, 5.3333])  <- ADDED to the old gradient
`}
          />
          <p>
            <code>backward()</code> <strong>adds</strong> the new gradient to whatever is already in <code>.grad</code> (useful when a
            parameter is used several times in one graph). In a training loop, the gradient of the previous minibatch would be added
            to the current one, so every loop calls <code>optimizer.zero_grad()</code> before <code>loss.backward()</code>.
          </p>
          <FnRef
            rows={[
              ["requires_grad=True / x.requires_grad_(True)", "Track operations on x so gradients with respect to x can be computed. Model parameters have it on automatically.", "—", "torch.ones(3).requires_grad", "False"],
              ["y.backward()", "Backpropagation from the scalar y: fills .grad of every tensor that requires gradients.", "—"],
              ["x.grad", "The gradient ∂y/∂x, same shape as x. None before the first backward().", "same as x"],
              ["t.detach()", "The same values, cut off from the graph (no more gradient tracking).", "same as t"],
              ["with torch.no_grad():", "Inside the block, no graph is recorded: less memory, faster. Use it for evaluation and for changing parameters by hand.", "—"],
            ]}
          />
          <H>Models: <code>nn.Module</code>, <code>nn.Linear</code>, <code>nn.Sequential</code></H>
          <CodeWalk
            title="The PyTorch tutorial's SimpleClassifier"
            code={py`
class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
logits = model(torch.rand(8, 2))
`}
            steps={[
              { line: 1, say: <>Our model is a subclass of <code>nn.Module</code>, which provides <code>__call__</code>, <code>parameters()</code>, <code>train()</code>/<code>eval()</code>, <code>to(device)</code> and more.</> },
              { line: 3, say: <><code>super().__init__()</code> runs nn.Module's own constructor first. Without it, PyTorch can't register the layers you assign next.</> },
              { line: 4, say: <><code>nn.Linear(2, 4)</code> computes <code>x @ W.T + b</code> with a weight of shape <strong>(out, in) = (4, 2)</strong>, one row per hidden neuron, and a bias of shape (4,). Assigning it to <code>self.linear1</code> registers its parameters.</>, vars: [["linear1.weight", "(4, 2)"], ["linear1.bias", "(4,)"]] },
              { line: 5, say: "The tanh activation. It has no parameters.", },
              { line: 6, say: "The output layer: 4 hidden values → 1 logit.", vars: [["linear2.weight", "(1, 4)"], ["linear2.bias", "(1,)"]] },
              { line: [8, 12], say: <><code>forward</code> chains the layers. Note it returns a <strong>logit</strong>, not a probability: the tutorial uses <code>BCEWithLogitsLoss</code>, which applies the sigmoid itself.</> },
              { line: 14, say: <>Total parameters: (2·4 + 4) + (4·1 + 1) = 12 + 5 = 17.</>, vars: [["sum(p.numel() for p in model.parameters())", "17"]] },
              { line: 15, say: <>A batch of 8 random inputs. <code>model(...)</code> calls <code>forward</code>: (8, 2) → linear1 → (8, 4) → tanh → (8, 4) → linear2 → (8, 1).</>, vars: [["logits.shape", "torch.Size([8, 1])"]] },
            ]}
          />
          <PyCell
            readOnly
            title="The same network with nn.Sequential (read-only; outputs from a real run)"
            code={py`
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
)
for name, param in model.named_parameters():
    print(name, tuple(param.shape))
# 0.weight (4, 2)
# 0.bias (4,)
# 2.weight (1, 4)       <- layer 1 is the Tanh, which has no parameters
# 2.bias (1,)
`}
          />
          <FnRef
            rows={[
              ["nn.Linear(in, out)", <>A dense layer <code>x @ W.T + b</code>. Weight <strong>(out, in)</strong>, bias (out,). Parameters: (in + 1) · out.</>, "(N, in) → (N, out)"],
              ["nn.ReLU(), nn.Tanh(), nn.Sigmoid()", "Activation functions as layers, applied entry by entry. No parameters.", "same shape"],
              ["nn.Sequential(l1, l2, ...)", <>Runs the layers in order. <code>model[0]</code> is the first layer, so <code>model[0].weight</code> is its weight.</>, "—"],
              ["nn.Flatten()", "Flattens everything except the batch axis (start_dim=1 by default).", "(N, C, H, W) → (N, C·H·W)"],
              ["model.parameters()", "All the trainable tensors, which you hand to the optimizer.", "—"],
              ["p.numel()", "The number of entries of a tensor: sum them to count parameters.", "→ int", "torch.zeros(4, 2).numel()", "8"],
              ["model.state_dict(), torch.save / torch.load, load_state_dict", "Save a trained model's parameters to a file and load them into a new model with the same architecture.", "—"],
            ]}
          />
          <H>Predict the output</H>
          <Think q={<>What does this print?<Pre>{py`
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad)
`}</Pre></>}>
            <code>tensor(8.)</code>: <M>{t`\frac{dy}{dx} = 2x + 2 = 8`}</M> at x = 3.
          </Think>
          <Think q={<>What's the shape of <code>nn.Linear(784, 32).weight</code>, and how many parameters does the layer have?</>}>
            <strong>(32, 784)</strong>, one row per output neuron. Parameters: 32·784 + 32 = 785·32 = <strong>25,120</strong>.
          </Think>
          <Think q={<>Why does <code>torch.from_numpy(X).float()</code> need the <code>.float()</code>?</>}>
            NumPy arrays are float64 by default and <code>from_numpy</code> keeps the type, but the model's parameters are float32.
            Mixing them raises a dtype error.
          </Think>
          <LabRef>the PyTorch tutorial (lab 4)</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ Training
    {
      id: "pytorch",
      title: "Training an MLP: by hand in NumPy, then in PyTorch",
      summary: "The MLP tutorial's forward and backward pass with real numbers, the PyTorch training loop line by line, losses, and the sklearn ↔ PyTorch correspondence.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            The MLP tutorial first trains a one-hidden-layer network with backpropagation written by hand in NumPy, then trains
            the same network in PyTorch. Seeing both side by side shows what <code>loss.backward()</code> and{" "}
            <code>optimizer.step()</code> actually do.
          </p>
          <H>Forward and backward pass by hand</H>
          <p>
            These are the lines of the tutorial's <code>MLP.forward</code> and <code>MLP.backward</code>, with tiny weights chosen
            so you can follow every number: N = 2 examples, D = 2 inputs, H = 2 hidden ReLU neurons, one sigmoid output, BCE loss.
          </p>
          <CodeWalk
            title="One forward and backward pass of the MLP tutorial's network"
            code={py`
X = np.array([[1.0, 2.0], [0.0, -1.0]])     # (N, D)
y = np.array([1.0, 0.0])
W1 = np.array([[0.5, -0.5], [1.0, 0.0]])    # (H, D)
b1 = np.zeros(2)
w2 = np.array([1.0, -1.0])                  # (H,)
b2 = np.zeros(1)

Z1 = X @ W1.T + b1        # forward
H = relu(Z1)
z2 = H @ w2 + b2
y_hat = sigmoid(z2)

N = y.size                # backward
z2_grad = (y_hat - y) / N
w2_grad = H.T @ z2_grad
b2_grad = z2_grad.sum(keepdims=True)
Z1_grad = np.outer(z2_grad, w2) * relu_derivative(Z1)
W1_grad = Z1_grad.T @ X
b1_grad = Z1_grad.sum(axis=0)
`}
            steps={[
              { line: [1, 6], say: <>The data and the parameters. <code>W1</code> has one row per hidden neuron, like <code>nn.Linear(2, 2).weight</code>.</> },
              { line: 8, say: <>Hidden pre-activations: (2, 2) @ (2, 2) → (2, 2), one row per example. Row 0: <M>{t`(0.5\cdot1 - 0.5\cdot2,\ 1\cdot1 + 0\cdot2) = (-0.5, 1)`}</M>.</>, vars: [["Z1", "[[-0.5, 1.0], [0.5, 0.0]]"]] },
              { line: 9, say: "ReLU sets the negative entries to 0.", vars: [["H", "[[0.0, 1.0], [0.5, 0.0]]"]] },
              { line: 10, say: <>Output logits: <M>{t`0\cdot1 + 1\cdot(-1) = -1`}</M> and <M>{t`0.5\cdot1 + 0\cdot(-1) = 0.5`}</M>.</>, vars: [["z2", "[-1.0, 0.5]"]] },
              { line: 11, say: <>Probabilities of class 1. Both are bad: example 0 is class 1 but gets 0.27; example 1 is class 0 but gets 0.62.</>, vars: [["y_hat", "[0.2689, 0.6225]"]] },
              { line: [13, 14], say: <>For sigmoid + BCE the gradient with respect to the logit simplifies to <M>{t`(\hat y - y)/N`}</M> (Lecture 3). Negative for example 0 ("raise its logit"), positive for example 1.</>, vars: [["z2_grad", "[-0.3655, 0.3112]"]] },
              { line: 15, say: <>"Error × input": the input of the output layer is H. <code>H.T @ z2_grad</code> sums <M>{t`\text{error}_n \cdot \mathbf h_n`}</M> over examples: <M>{t`(0.5\cdot0.3112,\ 1\cdot(-0.3655))`}</M>.</>, vars: [["w2_grad", "[0.1556, -0.3655]"]] },
              { line: 16, say: <>The bias's input is always 1, so its gradient is just the sum of the errors. <code>keepdims=True</code> keeps the shape (1,), matching <code>b2</code>.</>, vars: [["b2_grad", "[-0.0543]"]] },
              { line: 17, say: <>Backpropagate into the hidden layer. <code>np.outer(z2_grad, w2)</code> sends each example's error back through each outgoing weight; <code>* relu_derivative(Z1)</code> zeroes it wherever the ReLU was off (Z1 &lt; 0 or = 0).</>, vars: [["np.outer(z2_grad, w2)", "[[-0.3655, 0.3655], [0.3112, -0.3112]]"], ["relu_derivative(Z1)", "[[0, 1], [1, 0]]"], ["Z1_grad", "[[0, 0.3655], [0.3112, 0]]"]] },
              { line: 18, say: <>Error × input again, now for the first layer: (H, N) @ (N, D) → (H, D), the shape of W1.</>, vars: [["W1_grad", "[[0, -0.3112], [0.3655, 0.7311]]"]] },
              { line: 19, say: "Sum of the hidden errors over the examples.", vars: [["b1_grad", "[0.3112, 0.3655]"]] },
            ]}
          />
          <PyCell
            title="Run the pass yourself (PyTorch's autograd gives exactly these gradients)"
            code={py`
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu = lambda z: np.maximum(z, 0)
relu_derivative = lambda z: np.heaviside(z, 0)

X = np.array([[1.0, 2.0], [0.0, -1.0]])
y = np.array([1.0, 0.0])
W1 = np.array([[0.5, -0.5], [1.0, 0.0]])
b1 = np.zeros(2)
w2 = np.array([1.0, -1.0])
b2 = np.zeros(1)

Z1 = X @ W1.T + b1
H = relu(Z1)
z2 = H @ w2 + b2
y_hat = sigmoid(z2)

N = y.size
z2_grad = (y_hat - y) / N
w2_grad = H.T @ z2_grad
b2_grad = z2_grad.sum(keepdims=True)
Z1_grad = np.outer(z2_grad, w2) * relu_derivative(Z1)
W1_grad = Z1_grad.T @ X
b1_grad = Z1_grad.sum(axis=0)
print("y_hat  ", y_hat.round(4))
print("w2_grad", w2_grad.round(4), " b2_grad", b2_grad.round(4))
print("W1_grad", (W1_grad + 0.0).round(4).tolist(), " b1_grad", b1_grad.round(4))
`}
          />
          <p className="muted">Try it: change <code>W1[1, 0]</code> from 1.0 to −1.0 and predict which entries of <code>W1_grad</code> become 0. (The whole second row: neuron 2 is then off for both examples, so no error flows back through it.)</p>
          <H>The same network in PyTorch</H>
          <PyCell
            readOnly
            title="The model, loss and optimizer from the MLP tutorial (read-only)"
            code={py`
model = torch.nn.Sequential(
    torch.nn.Linear(D, H),           # W1 of shape (H, D), b1 of shape (H,)
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),           # w2 of shape (1, H), b2 of shape (1,)
    torch.nn.Sigmoid(),
    torch.nn.Flatten(start_dim=0),   # (N, 1) -> (N,), like our y
)
loss_function = torch.nn.BCELoss()                    # takes probabilities (the model ends with Sigmoid)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
`}
          />
          <CodeWalk
            title="The MLP tutorial's PyTorch training loop (D = 2, H = 32, 200 training examples, batch_size = 100)"
            code={py`
X_train_pt = torch.from_numpy(X_train).float()
y_train_pt = torch.from_numpy(y_train).float()

for epoch in range(1, N_epochs + 1):
    perm = torch.randperm(X_train_pt.shape[0])
    for start in range(0, X_train_pt.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        X_batch = X_train_pt[idx]
        y_batch = y_train_pt[idx]

        y_hat = model(X_batch)
        loss = loss_function(y_hat, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_loss_pt.append(loss_function(model(X_train_pt), y_train_pt).item())
`}
            steps={[
              { line: [1, 2], say: <>NumPy arrays → float32 tensors. Without <code>.float()</code> they'd be float64 and clash with the float32 parameters.</>, vars: [["X_train_pt.shape", "torch.Size([200, 2])"], ["y_train_pt.shape", "torch.Size([200])"]] },
              { line: 4, say: <>One <strong>epoch</strong> = one pass over all training examples. <code>range(1, N_epochs + 1)</code> counts from 1 so that "epoch 100" means the 100th.</> },
              { line: 5, say: <>A fresh random order of 0…199 every epoch, so the minibatches differ from epoch to epoch (SGD, Lecture 2).</>, vars: [["perm.shape", "torch.Size([200])"]] },
              { line: [6, 9], say: <><code>start</code> takes 0 and 100. <code>idx</code> picks 100 shuffled positions, and indexing both tensors with the <em>same</em> <code>idx</code> keeps inputs and labels paired.</>, vars: [["X_batch.shape", "torch.Size([100, 2])"], ["y_batch.shape", "torch.Size([100])"]] },
              { line: 11, say: <><strong>Forward pass</strong>: the model's prediction for the minibatch. PyTorch records the computation graph as it goes.</>, vars: [["y_hat.shape", "torch.Size([100])"]] },
              { line: 12, say: <>The mean BCE over the 100 examples: one number (a 0-D tensor) at the end of the graph.</>, vars: [["loss.shape", "torch.Size([])"]] },
              { line: 14, say: <><strong>Reset</strong> every parameter's <code>.grad</code> to zero. Without it, this minibatch's gradient would be added to the previous ones (debugging tutorial, bug 3).</> },
              { line: 15, say: <><strong>Backward pass</strong>: fills <code>.grad</code> of all parameters, the same numbers as our NumPy <code>backward</code>.</> },
              { line: 16, say: <><strong>Update</strong>: for every parameter, <code>p ← p − lr · p.grad</code>. That's our NumPy loop <code>params[j] = params[j] - lr * grads[j]</code>.</> },
              { line: [18, 19], say: <>After each epoch, measure the loss on the whole training set. <code>torch.no_grad()</code> skips the graph (we won't call backward on it), and <code>.item()</code> stores a plain float, not a tensor.</>, vars: [["train_loss_pt", "[0.69..., ...]: one float per epoch"]] },
            ]}
          />
          <Table
            head={["If you delete…", "What happens"]}
            rows={[
              [<code>optimizer.zero_grad()</code>, "Gradients accumulate across minibatches; the steps get larger and point in stale directions. The loss stays high and erratic. No error."],
              [<code>loss.backward()</code>, <>No gradients are computed (<code>.grad</code> stays None or zero), so <code>step()</code> changes nothing. The loss never moves.</>],
              [<code>optimizer.step()</code>, "Gradients are computed but never used: the parameters never change."],
              [<><code>perm = torch.randperm(...)</code> (using the examples in their original order)</>, "Same minibatches in the same order every epoch: still trains, but it's no longer stochastic in the usual sense."],
              [<code>with torch.no_grad():</code>, "Still correct, but every monitoring pass builds a graph, wasting memory."],
              [<code>.item()</code>, "The list holds tensors, which keep their graphs (or GPU memory) alive. Memory grows."],
            ]}
          />
          <PyCell
            readOnly
            title="Evaluating: from logits to accuracy (PyTorch tutorial's eval_model, read-only)"
            code={py`
model.eval()                                     # dropout off, batch norm uses running statistics
with torch.no_grad():                            # no graph during evaluation
    for data_inputs, data_labels in data_loader: # DataLoader yields (inputs, labels) minibatches
        preds = model(data_inputs).squeeze(dim=1)    # (N, 1) logits -> (N,)
        preds = torch.sigmoid(preds)                 # logits -> probabilities
        pred_labels = (preds >= 0.5).long()          # probabilities -> 0/1 classes
        true_preds += (pred_labels == data_labels).sum()
        num_preds += data_labels.shape[0]
acc = true_preds / num_preds
`}
          />
          <Private id="labs/pytorch/a4" />
          <H>Losses, optimizers and data loaders</H>
          <Table
            head={["Thing", "What to remember"]}
            rows={[
              [<code>nn.Linear(in, out)</code>, <>weight shape <strong>(out, in)</strong>, one row per output unit; bias (out,)</>],
              [<code>CrossEntropyLoss</code>, "input: logits (N, K); target: class indices, int64 (N,). Don't add a softmax to the model."],
              [<code>BCEWithLogitsLoss</code>, "binary: takes logits, applies the sigmoid itself; targets are floats"],
              [<code>BCELoss</code>, "binary: takes probabilities (model must end with Sigmoid)"],
              [<code>torch.optim.SGD(params, lr, momentum=…)</code>, "gradient descent on minibatches; momentum as in Lecture 2"],
              [<code>torch.optim.Adam(params, lr)</code>, "Adam (Lecture 2); used by the debugging tutorial"],
              [<code>DataLoader(dataset, batch_size, shuffle, drop_last)</code>, <>serves minibatches; <code>shuffle=True</code> reshuffles every epoch (training only); <code>drop_last=True</code> skips an incomplete last batch</>],
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
          <H>Predict the output</H>
          <Think q="In the NumPy backward pass, why is b2_grad the sum of z2_grad but w2_grad is H.T @ z2_grad?">
            Every weight's gradient is "error × the input it multiplies", summed over examples. The bias multiplies a constant 1,
            so its gradient is the sum of the errors times 1. <code>w2</code> multiplies the hidden values H, so each error gets
            weighted by them.
          </Think>
          <Think q="With 200 training examples and batch_size = 100, how many times is optimizer.step() called in 50 epochs?">
            2 minibatches per epoch × 50 epochs = <strong>100</strong> steps.
          </Think>
          <Think q="The model ends with nn.Sigmoid() and is trained with BCEWithLogitsLoss. What's wrong?">
            The sigmoid is applied twice: BCEWithLogitsLoss expects logits and applies its own sigmoid. Either remove the Sigmoid
            layer, or keep it and use BCELoss.
          </Think>
          <LabRef>assignment 4, tutorials on PyTorch, MLPs and polynomial fitting</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ Convolutions
    {
      id: "conv-code",
      title: "Convolutions in code",
      summary: "The tutorial's four-loop convolution step by step, (N, C, H, W) tensors, Conv2d/MaxPool2d/Flatten shapes, and finding in_features of the head.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            The convolution tutorial of Lab 5 writes a 2-D convolution as plain Python loops, checks it against PyTorch's
            <code> Conv2d</code>, and then tracks tensor shapes through a small CNN. The exam's classic CNN question is a shape
            question: "what is <code>in_features</code> of the final linear layer?"
          </p>
          <MB>{t`a_{i,j} = \sum_{u=0}^{F-1}\sum_{v=0}^{F-1} w_{u,v}\,x_{i+u,\,j+v},\qquad \text{output } (H - F + 1)\times(W - F + 1)\ \text{without padding}`}</MB>
          <p>
            In words: put the F × F kernel on top of the image with its corner at (i, j), multiply the overlapping numbers pair by
            pair, and add them all up. That sum is output entry (i, j). Then slide the kernel one step and repeat.
          </p>
          <H>The convolution as loops</H>
          <CodeWalk
            title="conv2d_python from the convolution tutorial, on a 3 × 3 image"
            code={py`
def conv2d_python(image, w):
    H, W = image.shape
    F, _ = w.shape
    activation = np.zeros((H - F + 1, W - F + 1), dtype=np.float32)
    for i in range(H - F + 1):          # output row
        for j in range(W - F + 1):      # output column
            for u in range(F):          # kernel row
                for v in range(F):      # kernel column
                    activation[i, j] += w[u, v] * image[i + u, j + v]
    return activation

image = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
w_vertical = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
print(conv2d_python(image, w_vertical))
`}
            steps={[
              { line: [12, 13], say: <>A 3 × 3 image that is dark (0) on the left and bright (1) in the last column, and the tutorial's 2 × 2 <strong>vertical edge detector</strong>: right column minus left column.</> },
              { line: [2, 3], say: <>Tuple unpacking of the shapes. <code>F, _ = w.shape</code> keeps the kernel size and throws away the second (equal) number.</>, vars: [["H, W", "3, 3"], ["F", "2"]] },
              { line: 4, say: <>The output has <M>{t`(3 - 2 + 1)\times(3 - 2 + 1) = 2\times2`}</M> entries, all starting at 0.</>, vars: [["activation.shape", "(2, 2)"]] },
              { line: [5, 6], say: <>Two loops over the output positions: (0, 0), (0, 1), (1, 0), (1, 1).</> },
              { line: [7, 9], say: <>Two loops over the kernel. For (i, j) = (0, 0) the window is [[0, 0], [0, 0]]: every product is 0, so <code>activation[0, 0]</code> = 0.</>, vars: [["activation[0, 0]", "0"]] },
              { line: 9, say: <>For (i, j) = (0, 1) the window is image[0:2, 1:3] = [[0, 1], [0, 1]]. The four products: <M>{t`(-1)\cdot0 + 1\cdot1 + (-1)\cdot0 + 1\cdot1 = 2`}</M>. The kernel "fires" where brightness jumps from left to right.</>, vars: [["activation[0, 1]", "2"]] },
              { line: [10, 14], say: "Row 1 sees the same pattern, so the output marks the edge in its right column.", vars: [["printed", "[[0. 2.] [0. 2.]]"]] },
            ]}
          />
          <PyCell
            code={py`
import numpy as np

def conv2d_python(image, w):
    H, W = image.shape
    F, _ = w.shape
    activation = np.zeros((H - F + 1, W - F + 1), dtype=np.float32)
    for i in range(H - F + 1):
        for j in range(W - F + 1):
            for u in range(F):
                for v in range(F):
                    activation[i, j] += w[u, v] * image[i + u, j + v]
    return activation

image = np.zeros((5, 6)); image[:, 3:] = 1                   # dark left half, bright right half
w_vertical = np.array([[-1, 1], [-1, 1]])
print(conv2d_python(image, w_vertical))                       # lights up at the vertical edge
`}
          />
          <p className="muted">Try it: use the tutorial's horizontal edge detector <code>[[-1, -1], [1, 1]]</code> on the same image. Predict first: the image only changes from left to right, so the output is all zeros.</p>
          <H>From (H, W) to (N, C, H, W)</H>
          <p>
            PyTorch layers always process a <em>minibatch</em> of images with channels, in the order <strong>(N, C, H, W)</strong>:
            images, channels, height, width. The tutorial's <code>to_nchw</code> converts one image:
          </p>
          <CodeWalk
            code={py`
x = torch.as_tensor(image)              # image of shape (10, 12)
x = x.unsqueeze(2)                      # (H, W) -> (H, W, 1)
x = x.permute(2, 0, 1)                  # (H, W, C) -> (C, H, W)
x = x.unsqueeze(0)                      # (C, H, W) -> (1, C, H, W)
`}
            steps={[
              { line: 1, say: "The NumPy image becomes a tensor (sharing memory).", vars: [["x.shape", "torch.Size([10, 12])"]] },
              { line: 2, say: <>Adds a channel axis of length 1 at position 2 (the end). A colour photo would already have 3 channels here.</>, vars: [["x.shape", "torch.Size([10, 12, 1])"]] },
              { line: 3, say: <>Moves the channel axis to the front: new axis 0 is old axis 2, then old axes 0 and 1.</>, vars: [["x.shape", "torch.Size([1, 10, 12])"]] },
              { line: 4, say: "Adds the batch axis in front: a batch of one image.", vars: [["x.shape", "torch.Size([1, 1, 10, 12])"]] },
            ]}
          />
          <H>Every layer, with its shapes</H>
          <FnRef
            rows={[
              ["nn.Conv2d(C_in, C_out, F)", <>C_out kernels of size F × F, each looking at all C_in input channels. Weight <strong>(C_out, C_in, F, F)</strong>, bias (C_out,). Parameters: <M>{t`F\cdot F\cdot C_{\text{in}}\cdot C_{\text{out}} + C_{\text{out}}`}</M>, independent of the image size.</>, "(N, C_in, H, W) → (N, C_out, H−F+1, W−F+1)", "torch.nn.Conv2d(1, 2, 2).weight.shape", "torch.Size([2, 1, 2, 2])"],
              ["padding=P", <>Adds P rings of zeros around the image first. With F = 3, <code>padding=1</code> keeps the size ("same" padding).</>, "H → H + 2P − F + 1", "torch.nn.Conv2d(3, 8, 3, padding=1)(torch.zeros(1, 3, 32, 32)).shape", "torch.Size([1, 8, 32, 32])"],
              ["stride=S", "Moves the kernel S steps at a time: the output shrinks by about a factor S.", "H → ⌊(H + 2P − F)/S⌋ + 1", "torch.nn.Conv2d(3, 8, 3, stride=2)(torch.zeros(1, 3, 32, 32)).shape", "torch.Size([1, 8, 15, 15])"],
              ["nn.Conv1d(C_in, C_out, F)", "The same for 1-D signals. Weight (C_out, C_in, F).", "(N, C_in, W) → (N, C_out, W−F+1)"],
              ["nn.MaxPool2d(2)", "Keeps the largest value of each 2 × 2 block. Halves H and W (rounding down). No parameters.", "(N, C, H, W) → (N, C, ⌊H/2⌋, ⌊W/2⌋)", "torch.nn.MaxPool2d(2)(torch.zeros(1, 1, 7, 7)).shape", "torch.Size([1, 1, 3, 3])"],
              ["nn.ReLU()", "Entry by entry, so the shape doesn't change. Turns activation maps into feature maps.", "same shape"],
              ["nn.Flatten()", "Merges channels, height and width into one axis for the linear head; the batch axis stays.", "(N, C, H, W) → (N, C·H·W)", "torch.nn.Flatten()(torch.zeros(5, 8, 14, 14)).shape", "torch.Size([5, 1568])"],
              ["torch.allclose(a, b)", "Tensor version of np.allclose: the tutorial checks that its loops and Conv2d agree.", "→ bool"],
              ["with torch.no_grad(): conv.weight[0, 0] = ...", "Writes hand-made kernels into a layer. The no_grad block stops autograd from recording the assignment.", "—"],
            ]}
          />
          <Table
            head={["Max pooling on one 4 × 4 channel", "Result"]}
            rows={[
              [<code>[[1, 3, 2, 0], [4, 0, 1, 5], [0, 2, 0, 0], [1, 1, 3, 1]]</code>, <><code>[[4, 5], [2, 3]]</code>: the max of each 2 × 2 block (top-left block 1, 3, 4, 0 → 4)</>],
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
          <H>Predict the output</H>
          <Think q="Conv2d(1, 4, 3) on (N, 1, 28, 28), no padding, then MaxPool2d(2), then Flatten. What is in_features of the next Linear layer?">
            28 − 3 + 1 = 26, pooled to 13: (N, 4, 13, 13), flattened to 4 · 13 · 13 = <strong>676</strong>.
          </Think>
          <Think q="How many parameters does Conv2d(3, 16, 3) have? Does it change for 64 × 64 images instead of 32 × 32?">
            3 · 3 · 3 · 16 + 16 = <strong>448</strong>, for any image size: the same kernels slide over the whole image (weight sharing).
          </Think>
          <Think q={<>An image tensor has shape (32, 32, 3) (height, width, RGB). Which call puts it in the order Conv2d expects for a batch of one?</>}>
            <code>x.permute(2, 0, 1).unsqueeze(0)</code>: first (3, 32, 32), then (1, 3, 32, 32).
          </Think>
          <LabRef>assignment 5, the convolution layer tutorial</LabRef>
        </>
      ),
    },
    // ------------------------------------------------------------------ Debugging
    {
      id: "debugging",
      title: "Debugging neural networks",
      summary: "The step-by-step strategy, and the debugging tutorial's buggy training code: which line causes each of the five bugs and how it's fixed.",
      body: () => (
        <>
          <H>What this is for</H>
          <p>
            In ordinary programs a bug usually crashes the code. In machine learning a bug often lets the code run and the model
            just learns badly, so you need a strategy and checks. The debugging tutorial of Lab 5 gives you a buggy MNIST training
            script with five bugs; the midterm can show you code like it and ask what's wrong.
          </p>
          <H>The strategy</H>
          <ol>
            <li><strong>Check the data</strong>: inputs and labels match (also after shuffling), shapes and value ranges as expected.</li>
            <li><strong>Start simple</strong> (e.g. a linear model): the loss must decrease at least a little. If not: check minibatches, tensor shapes, whether parameters change at all (forgotten <code>backward()</code>/<code>step()</code>), softmax axis, a few learning rates.</li>
            <li><strong>Implement the final model</strong>; fix shape mismatches; <code>train()</code> for training, <code>eval()</code> for evaluation.</li>
            <li><strong>Overfitting test</strong>: 5–10 examples, many epochs, large learning rate. A correct model drives the training loss to ≈ 0. If it can't even memorize, there's a bug.</li>
            <li><strong>Run the real experiment</strong>; NaNs → smaller learning rate or gradient clipping; tune on validation.</li>
          </ol>
          <H>The buggy code, line by line</H>
          <p>
            This is the tutorial's first version of the two functions (the model is a 784-500-10 MLP with batch norm, trained with
            Adam on MNIST, with <code>batch_size = 131</code>). Step through it and look for the problems before you read the
            explanations.
          </p>
          <CodeWalk
            title="Find the five bugs"
            code={py`
def train_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    model.train()
    for images, labels in train_loader:
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def evaluate(test_loader, model, loss_fn):
    losses = []
    num_errors = 0
    model.eval()
    for images, labels in test_loader:
        logits = model(images)
        losses.append(loss_fn(logits, labels))
        predictions = torch.argmax(logits, dim=1)
        num_errors += (predictions != labels).sum()
    return torch.tensor(losses).mean(), num_errors / len(test_loader.dataset)
`}
            steps={[
              { line: 3, say: <><code>model.train()</code>: batch norm uses minibatch statistics. Correct.</> },
              { line: 4, say: <>The <code>DataLoader</code> yields minibatches of (images, labels): images (131, 1, 28, 28) on the CPU, labels (131,) int64. 60,000 = 458 · 131 + 2, so the last minibatch has only <strong>2</strong> images (<strong>bug 5</strong>, see below).</>, vars: [["images.shape", "torch.Size([131, 1, 28, 28])"]] },
              { line: 5, say: <><strong>Bug 1</strong>: on a GPU the model lives on <code>cuda</code> but the images are still on the CPU → "Expected all tensors to be on the same device". <strong>Bug 2</strong>: the first <code>Linear(784, 500)</code> wants (N, 784), not (N, 1, 28, 28) → "shapes cannot be multiplied (28x28 and 784x500)".</> },
              { line: 6, say: "CrossEntropyLoss on logits and integer labels. Correct." },
              { line: [7, 8], say: <><strong>Bug 3</strong>: there's no <code>optimizer.zero_grad()</code> before <code>backward()</code>, so gradients pile up over all previous minibatches. No crash; the loss stays large and erratic.</> },
              { line: 9, say: <>Stores <code>loss.item()</code>, a plain float. Correct.</> },
              { line: [15, 17], say: <><code>model.eval()</code> is right, but there's no <code>with torch.no_grad():</code>. <strong>Bug 4</strong> starts here: every forward pass builds a graph.</> },
              { line: 18, say: <>…and <strong>bug 4</strong> finishes here: the list stores loss <em>tensors</em>, each holding its graph, so memory grows with every test minibatch. Fix: <code>no_grad</code>, and <code>.item()</code>.</> },
              { line: [19, 21], say: <><code>argmax(dim=1)</code> gives the predicted digit per image; comparing and summing counts the errors. Correct.</> },
            ]}
          />
          <H>The five bugs, fixed</H>
          <Table
            head={["Symptom", "Broken", "Fixed"]}
            rows={[
              [<>1. RuntimeError: tensors on different devices (cpu vs cuda)</>, <code>logits = model(images)</code>, <code>images, labels = images.to(device), labels.to(device)</code>],
              [<>2. RuntimeError: shapes cannot be multiplied (28x28 and 784x500)</>, "images are (N, 1, 28, 28)", <code>images = torch.flatten(images, start_dim=1)</code>],
              ["3. Runs, but the loss stays large and erratic", <><code>loss.backward()</code> with no reset</>, <><code>optimizer.zero_grad()</code> right before <code>loss.backward()</code></>],
              ["4. Memory grows during evaluation", <><code>losses.append(loss_fn(logits, labels))</code> outside no_grad</>, <><code>with torch.no_grad():</code> around the loop, and <code>.item()</code></>],
              ["5. Loss spike at the end of every epoch", <>the last training minibatch has 2 examples; batch norm with B = 2 maps every feature to ±1</>, <><code>DataLoader(..., drop_last=True)</code> for the <em>training</em> loader only</>],
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
          <Callout kind="mistake">
            Never use <code>drop_last=True</code> on the <em>test</em> loader: it would silently skip test examples and change the
            reported error.
          </Callout>
          <H>Predict the symptom</H>
          <Think q="Someone writes optimizer.zero_grad() after loss.backward() but before optimizer.step(). What happens?">
            The gradient is computed and then immediately erased, so <code>step()</code> sees zero gradients. Plain SGD then doesn't
            change the parameters at all: the loss stays flat from the first step.
          </Think>
          <Think q="The model contains Dropout. Evaluation is run without calling model.eval(). What do you observe?">
            Dropout stays on during evaluation: random units are switched off, so the test error is worse than it should be and
            changes from run to run on the same data.
          </Think>
          <Think q="The training loss is NaN after a few hundred steps. What do you try first?">
            A smaller learning rate (or gradient clipping): the updates are probably so large that the parameters blow up.
          </Think>
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
