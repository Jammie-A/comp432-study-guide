import { Callout, FromLab, Formula, H, M, MB, MathCode, Slides, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { LineFitWidget, NegLogWidget, ProbeStepWidget } from "../widgets/l1";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

export const lecture1: Lecture = {
  id: "l1",
  number: 1,
  title: "Introduction",
  sections: [
    {
      id: "notation",
      title: "Notation & what ML is",
      summary: "The symbols used all course, and the idea of learning a function from data.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Most confusion in this course comes from symbols. Learn this table once and every formula becomes readable.
            The course uses the same notation as the textbooks.
          </p>
          <Table
            head={["Symbol", "Meaning", "Example"]}
            rows={[
              [<M>x</M>, "scalar (plain, lowercase)", "a temperature, 30"],
              [<M>{t`\mathbf x`}</M>, "vector (bold, lowercase)", "a flower's 4 measurements"],
              [<M>{t`\mathbf X`}</M>, "matrix (bold, uppercase)", "all flowers' measurements, one row each"],
              [<M>{t`x_i`}</M>, <>i-th entry of <M>{t`\mathbf x`}</M></>, <M>{t`x_2`}</M>],
              [<M>{t`\mathbb R`}</M>, "the real numbers", ""],
              [<M>{t`\mathbf x\in\mathbb R^D`}</M>, "x has D real entries", <M>{t`\mathbf x\in\mathbb R^4`}</M>],
              [<><M>y</M>, <M>{t`\hat y`}</M></>, "label (truth), prediction (\"y-hat\")", ""],
              [<M>\theta</M>, "model parameters (\"theta\")", <M>{t`\theta=(b,w)`}</M>],
            ]}
          />
          <Callout kind="exam">
            Important sizes: <strong>N</strong> examples, <strong>D</strong> features, <strong>K</strong> outputs (classes),
            <strong> P</strong> parameters. Whenever you see a matrix, ask "what's its shape in terms of N, D, K, P?"
          </Callout>
          <H>What is machine learning?</H>
          <ul>
            <li><strong>AI</strong>: any system doing a task we associate with intelligence, however it's built.</li>
            <li><strong>ML</strong>: a system whose behaviour is <em>fitted to data</em> instead of programmed by hand.</li>
            <li><strong>Deep learning</strong>: ML where the model is a neural network with many layers.</li>
          </ul>
          <p>
            Traditional programming: a human writes the rules (good for sorting, searching). ML: we collect examples and
            <em> learn</em> a function <M>{t`f(\mathbf x)`}</M> (good when rules are hard to write but data is easy to get, e.g. face recognition).
          </p>
          <H>The problem formulation</H>
          <MB>{t`\mathbf x\in\mathbb R^D\ \xrightarrow{\ f\ }\ \hat y`}</MB>
          <p>
            Inputs are vectors (pixels of an image, audio samples, features of a house). The <strong>output decides the
            problem type</strong>:
          </p>
          <Table
            head={["Output", "Problem", "Example"]}
            rows={[
              [<>real number(s), <M>{t`\hat y\in\mathbb R^K`}</M></>, "regression", "house features → price 665k$"],
              [<>a category, <M>{t`\hat y\in\{1,\dots,K\}`}</M></>, "classification", "image → \"cat\""],
            ]}
          />
          <Slides lecture={1} pages="3–7, 13" />
        </>
      ),
    },
    {
      id: "pipeline",
      title: "Datasets, learning types & the pipeline",
      summary: "Feature matrices, train/test splits, supervised vs unsupervised, and training → testing → inference.",
      body: () => (
        <>
          <H>The four components of every ML system</H>
          <ol>
            <li><strong>Dataset</strong>: examples of the input–output mapping we want.</li>
            <li><strong>Model</strong>: a function <M>{t`f_\theta(\mathbf x)`}</M> with adjustable parameters θ.</li>
            <li><strong>Objective function</strong> <M>J</M>: one number saying how well <M>{t`f_\theta`}</M> fits the data.</li>
            <li><strong>Optimization algorithm</strong>: searches for the θ that makes J best.</li>
          </ol>
          <H>Datasets and the feature matrix</H>
          <Formula
            plain="The data set is N pairs (input, label). Stack the N inputs as the rows of a table X: row n is example n, column d is feature d."
            tex={t`\mathcal D = \{(\mathbf x_1,y_1),\dots,(\mathbf x_N,y_N)\},\qquad \mathbf X = \begin{pmatrix}x_{1,1}&\cdots&x_{1,D}\\\vdots&&\vdots\\x_{N,1}&\cdots&x_{N,D}\end{pmatrix}\in\mathbb R^{N\times D}`}
            symbols={[
              [t`\mathcal D`, "the dataset: a set of (input, label) pairs"],
              [t`\mathbf X`, "feature matrix: ONE ROW PER EXAMPLE, one column per feature"],
              [t`x_{n,d}`, "feature d of example n (row n, column d)"],
            ]}
          />
          <p>Iris: 150 flowers, 4 features (sepal length/width, petal length/width) → <M>{t`\mathbf X\in\mathbb R^{150\times4}`}</M>.</p>
          <PyCell
            code={py`
import numpy as np
# first three iris flowers: sepal length, sepal width, petal length, petal width
X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9, 3.0, 1.4, 0.2],
              [7.0, 3.2, 4.7, 1.4]])
N, D = X.shape
print("N =", N, " D =", D)
print("example 3 (row):", X[2])          # Python counts from 0!
print("petal length of all (column):", X[:, 2])
`}
          />
          <Callout kind="mistake">
            Math counts from 1 (<M>{t`\mathbf x_1`}</M>), Python from 0 (<code>X[0]</code>). <code>X[n]</code> is a row (an example),
            <code> X[:, d]</code> is a column (a feature).
          </Callout>
          <H>Types of learning</H>
          <Table
            head={["Type", "Learns from", "Example"]}
            rows={[
              ["Supervised", <>inputs <M>{t`\mathbf X\in\mathbb R^{N\times D}`}</M> <em>and</em> labels <M>{t`\mathbf Y\in\mathbb R^{N\times K}`}</M></>, "linear models, neural nets, SVMs, trees (most of this course)"],
              ["Unsupervised", "inputs only, no labels", "clustering: grouping nearby points"],
              ["Reinforcement", "interacting with an environment, rewards", "game playing, robotics"],
            ]}
          />
          <H>Training → testing → inference</H>
          <Table
            head={["Stage", "Data", "What happens"]}
            rows={[
              ["1. Training", "training set (with labels)", "adjust θ until predictions ŷₙ match labels yₙ"],
              ["2. Testing", "test set (labels, never seen in training)", "θ frozen; compare predictions with labels → measures generalization"],
              ["3. Inference", "live data, no labels", "the shipped model just predicts; no learning anymore"],
            ]}
          />
          <Callout kind="exam">
            The training and test sets must contain <strong>different</strong> examples. Evaluated on examples it has already seen,
            a model's performance is overestimated. What actually matters is performance on unseen data: <strong>generalization</strong>.
          </Callout>
          <FromLab to="labs/numpy" label="NumPy: shapes, views, vectorization">
            Lab 1 drilled the code side of this page. A row <code>X[n]</code> is one example and a column <code>X[:, d]</code> is
            one feature. When you shuffle a dataset, use <em>one</em> permutation for both arrays:{" "}
            <code>perm = rng.permutation(N); X, Y = X[perm], Y[perm]</code>. Two separate shuffles silently pair inputs with the
            wrong labels.
          </FromLab>
          <Slides lecture={1} pages="9–12, 15–22" />
        </>
      ),
    },
    {
      id: "model-mse",
      title: "Models, parameter space & MSE",
      summary: "A model is a family of functions; parameters pick one. MSE scores how well it fits.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The line <M>{t`\hat y = wx + b`}</M> isn't one function; it's a whole family. Each choice of
            <M>{t`\ \theta = (b, w)`}</M> picks one line. All possible θ together form the <strong>parameter space</strong> (one
            dimension per parameter: 2 here, billions for modern models). Training = searching that space for the θ whose line
            fits best. To search, we need a score: the <strong>objective function</strong>.
          </p>
          <H>The formula, decoded</H>
          <Formula
            plain="For each example, take the prediction error and square it (so it's positive and big errors count more), then average over all N examples."
            title="Mean squared error"
            tex={t`\text{MSE} = \frac1N\sum_{n=1}^N(y_n - \hat y_n)^2`}
            symbols={[
              [t`y_n - \hat y_n`, "the error on example n (truth minus prediction)"],
              [t`(\cdot)^2`, "squared: always positive, and big errors count much more than small ones"],
              [t`\sum_{n=1}^N`, "add up over all N examples"],
              [t`\frac1N`, "divide by N: the average"],
            ]}
          />
          <p>Convention: <strong>smaller = better</strong>; training <em>minimizes</em> J. "Objective", "loss", "cost" and "criterion" all mean the same thing.</p>
          <H>Worked example: ice cream sales (slide 26)</H>
          <p>Data: temperatures <M>{t`x = (30, 20, 10)`}</M>, sales <M>{t`y = (40, 20, 0)`}</M>. Compare two candidate lines.</p>
          <Steps
            steps={[
              { title: "Candidate 1: ŷ = x + 10 (θ = (10, 1)). Predictions", body: <MB>{t`\hat y = (30+10,\ 20+10,\ 10+10) = (40,\ 30,\ 20)`}</MB> },
              { title: "Squared errors", body: <MB>{t`(40-40)^2 = 0,\quad (20-30)^2 = 100,\quad (0-20)^2 = 400`}</MB> },
              { title: "Average", body: <MB>{t`\text{MSE} = \tfrac13(0 + 100 + 400) \approx 166.7`}</MB> },
              { title: "Candidate 2: ŷ = 2x − 20 (θ = (−20, 2))", body: <MB>{t`\hat y = (40,\ 20,\ 0)\ \Rightarrow\ \text{errors all } 0\ \Rightarrow\ \text{MSE} = 0`}</MB> },
              { title: "Conclusion", body: <p>Candidate 2 fits perfectly. The MSE turned "which line is better?" into comparing two numbers.</p> },
            ]}
          />
          <H>Visual</H>
          <LineFitWidget />
          <H>Math → code</H>
          <MathCode
            rows={[
              [t`\hat y_n = wx_n + b`, "y_hat = w * x + b", "x is an array → all n at once"],
              [t`y_n - \hat y_n`, "y - y_hat", "element-wise"],
              [t`(\cdot)^2`, "(y - y_hat) ** 2", "** is power"],
              [t`\frac1N\sum_n`, ".mean()", "average over the array"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
x = np.array([30.0, 20.0, 10.0])
y = np.array([40.0, 20.0, 0.0])

def mse(w, b):
    y_hat = w * x + b
    return ((y - y_hat) ** 2).mean()

print("y = x + 10   :", mse(1, 10))
print("y = 2x - 20  :", mse(2, -20))
`}
          />
          <Slides lecture={1} pages="8, 23–29" />
        </>
      ),
    },
    {
      id: "classification-metrics",
      title: "Accuracy & cross-entropy",
      summary: "Accuracy counts correct answers; cross-entropy also rewards confidence, which is what training needs.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            For classification, the model outputs a <strong>probability per class</strong>, e.g. (cat 0.3, dog 0.5, bird 0.2). The
            prediction is the most probable class (<strong>argmax</strong>). How do we score this?
          </p>
          <H>Accuracy</H>
          <Formula
            plain="Count the examples where the predicted class equals the true class and divide by N: the fraction you got right."
            tex={t`\text{Acc} = \frac1N\sum_{n=1}^N \mathbb 1[\hat y_n = y_n]`}
            symbols={[
              [t`\mathbb 1[\cdot]`, "indicator: 1 if the statement is true, 0 if false"],
              [t`\hat y_n`, "predicted class = argmax of the probabilities"],
            ]}
          />
          <Table
            head={["", "cat", "dog", "bird", "prediction", "label", "correct?"]}
            rows={[
              ["x₁", "0.3", "0.5", "0.2", "dog", "dog", "1"],
              ["x₂", "0.6", "0.2", "0.2", "cat", "cat", "1"],
              ["x₃", "0.1", "0.5", "0.4", "dog", "bird", "0"],
            ]}
          />
          <p>Acc = 2/3 ≈ 0.67.</p>
          <H>Accuracy is a "hard" metric</H>
          <p>
            Model 2 below gives the same class predictions as Model 1 — so the same accuracy, 2/3 — but it is far more
            confident when it's right. Accuracy can't tell them apart. And for training we want a <em>soft</em> score that
            changes gradually as probabilities improve (so we can follow its slope).
          </p>
          <Table
            head={["Model 2", "cat", "dog", "bird"]}
            rows={[["x₁", "0.0", "1.0", "0.0"], ["x₂", "0.9", "0.1", "0.0"], ["x₃", "0.1", "0.5", "0.4"]]}
          />
          <H>Categorical cross-entropy (CCE)</H>
          <Formula
            plain="For each example, take minus the log of the probability the model gave to the correct class (the one-hot y picks it out), then average over the examples. Confident and right gives about 0; confident and wrong gives a large value."
            tex={t`\text{CCE} = -\frac1N\sum_{n=1}^N\sum_{k=1}^K y_{n,k}\ln(p_{n,k}) = -\frac1N\sum_{n=1}^N\mathbf y_n^\top\ln(\mathbf p_n)`}
            symbols={[
              [t`\mathbf y_n`, <>the label as a <strong>one-hot</strong> vector: dog → (0, 1, 0)</>],
              [t`p_{n,k}`, "the model's probability for class k on example n"],
              [t`\ln`, "natural log; ln(1) = 0, ln(small) = very negative"],
              [t`\sum_k y_{n,k}\ln p_{n,k}`, "because y is one-hot, only ONE term survives: ln of the probability on the correct class"],
              [t`-`, "minus sign so the result is positive (ln p ≤ 0)"],
            ]}
          />
          <Steps
            title="Term by term for x₁ (Lecture 2, slide 3)"
            steps={[
              { title: "Label and prediction", body: <MB>{t`\mathbf y_1 = (0, 1, 0),\qquad \mathbf p_1 = (0.3, 0.5, 0.2)`}</MB> },
              { title: "Take −ln of each probability", body: <MB>{t`-\ln\mathbf p_1 = (1.20,\ 0.69,\ 1.61)`}</MB> },
              { title: "Dot product with the one-hot label picks the correct class", body: <MB>{t`\mathbf y_1^\top(-\ln\mathbf p_1) = 0\cdot1.20 + 1\cdot0.69 + 0\cdot1.61 = 0.69`}</MB> },
            ]}
          />
          <Steps
            title="Comparing the two models"
            steps={[
              { title: "Model 1: correct-class probabilities 0.5, 0.6, 0.4", body: <MB>{t`\text{CCE} = -\tfrac13(\ln0.5 + \ln0.6 + \ln0.4) = \tfrac13(0.69 + 0.51 + 0.92) \approx 0.71`}</MB> },
              { title: "Model 2: correct-class probabilities 1.0, 0.9, 0.4", body: <MB>{t`\text{CCE} = -\tfrac13(\ln1.0 + \ln0.9 + \ln0.4) = \tfrac13(0 + 0.11 + 0.92) \approx 0.34`}</MB> },
              { title: "Verdict", body: <p>Same accuracy, but CCE says Model 2 is better (lower). Perfect predictions give CCE = 0; confident wrong predictions make it blow up.</p> },
            ]}
          />
          <NegLogWidget />
          <H>Math → code</H>
          <PyCell
            code={py`
import numpy as np
P1 = np.array([[0.3, 0.5, 0.2],
               [0.6, 0.2, 0.2],
               [0.1, 0.5, 0.4]])
Y = np.array([[0, 1, 0],      # dog
              [1, 0, 0],      # cat
              [0, 0, 1]])     # bird   (one-hot labels)

acc = (P1.argmax(axis=1) == Y.argmax(axis=1)).mean()
cce = -(Y * np.log(P1)).sum(axis=1).mean()
print(f"accuracy = {acc:.2f}, CCE = {cce:.2f}")
`}
          />
          <Callout kind="mistake">
            In <code>(Y * np.log(P1)).sum(axis=1)</code>, <code>axis=1</code> sums over <em>classes</em> (columns) for each example;
            then <code>.mean()</code> averages over examples. Swapping the axes gives a wrong answer.
          </Callout>
          <Callout kind="note">CCE is also called the <strong>negative log-likelihood (NLL)</strong>. Why this exact formula? That's the probabilistic view (extra lecture).</Callout>
          <FromLab to="labs/numpy" label="NumPy: shapes, views, vectorization">
            In the lab tutorials the CCE is <code>-(Y * np.log(Y_hat)).sum(axis=1).mean()</code>: <code>sum(axis=1)</code> adds
            over the classes, <code>.mean()</code> averages over the examples. Forget the mean and you get the midterm's sample
            question 3: a loss 100 times too big (230.26 instead of ln 10 ≈ 2.30) on a batch of 100.
          </FromLab>
          <Slides lecture={1} pages="30–34; Lecture 2 page 4" />
        </>
      ),
    },
    {
      id: "naive-training",
      title: "Optimization & the naive training loop",
      summary: "Training = minimizing J. Our first algorithm: probe nearby parameters, keep the best, repeat.",
      body: () => (
        <>
          <H>Why it matters</H>
          <MB>{t`\theta^* = \arg\min_\theta\ J\big(\mathbf Y_{\text{train}},\ f_\theta(\mathbf X_{\text{train}})\big)`}</MB>
          <p>
            "Find the θ that makes the loss on the training set as small as possible." The algorithm that does this is the
            <strong> optimizer</strong>. How?
          </p>
          <ul>
            <li><strong>Brute force</strong> (try every θ): impossible, there are infinitely many.</li>
            <li><strong>Random guessing</strong>: hopeless in a huge space.</li>
            <li><strong>Iterative</strong>: start somewhere, improve step by step. ✓</li>
          </ul>
          <H>Probe and step</H>
          <Steps
            title="One epoch of the naive loop (slide 37: 60 noisy points, start w = 1, b = 0)"
            steps={[
              { title: "Evaluate the current loss", body: <p>MSE(w = 1, b = 0) = 77.33 (poor fit).</p> },
              { title: "Probe all 8 neighbours (±0.05 or 0 in each of w, b)", body: <p>e.g. (+0.05, +0.05) → 71.46, (+0.05, 0) → 72.25, …, (−0.05, −0.05) → 83.44.</p> },
              { title: "Move to the best neighbour", body: <p>Best is (+0.05, +0.05): new w = 1.05, b = 0.05, MSE = 71.46.</p> },
              { title: "Repeat", body: <p>After 47 epochs: ŷ = 2.10x + 2.35, training MSE 1.52, test MSE 2.19 — slightly worse on test, as is typical, but close: it generalizes.</p> },
            ]}
          />
          <ProbeStepWidget />
          <Think q="With P parameters and 3 choices per parameter (+δ, 0, −δ), how many neighbours must we evaluate per step?">
            3^P − 1 (all combinations except "no change"). P = 2 → 8. P = 20 → about 3.5 billion. That's why probing doesn't
            scale, and Lecture 2 uses the <strong>gradient</strong>, which gives the best direction directly.
          </Think>
          <PyCell
            code={py`
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 60)
y = 2 * x + 2 + rng.normal(0, 1.2, 60)       # noisy points around a line

mse = lambda w, b: ((y - (w * x + b)) ** 2).mean()
w, b, d = 1.0, 0.0, 0.05
for epoch in range(60):
    probes = [(w + dw, b + db) for dw in (d, 0, -d) for db in (d, 0, -d) if (dw, db) != (0, 0)]
    best = min(probes, key=lambda p: mse(*p))
    if mse(*best) >= mse(w, b):
        break                                 # no neighbour is better: stop
    w, b = best
    if epoch % 10 == 0:
        print(f"epoch {epoch}: w={w:.2f} b={b:.2f} MSE={mse(w, b):.2f}")
print(f"final: y_hat = {w:.2f} x + {b:.2f}, MSE = {mse(w, b):.2f}")
`}
          />
          <Slides lecture={1} pages="36–41" />
        </>
      ),
    },
  ],
};
