import { Callout, H, M, MB, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { glossary } from "../glossary";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

export const basics: Lecture = {
  id: "start",
  number: 0,
  title: "Read this first",
  label: "Start here",
  short: "Start",
  sections: [
    {
      id: "notation",
      title: "How to read the notation",
      summary: "Every symbol the slides use, what to say when you read it, what it means, and the NumPy line that computes it.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The slides pack a lot into a few symbols. Once you can read a formula out loud in plain words, most of them turn out
            to say something simple, like "average the squared errors". This page is the decoder: every symbol, what to say
            when you see it, and the NumPy code that does the same thing.
          </p>
          <H>The letters that are always the same</H>
          <Table
            head={["Letter", "Always means", "Example"]}
            rows={[
              [<M>N</M>, "number of examples (rows of the data)", "150 flowers in the iris data"],
              [<M>D</M>, "number of features per example (columns)", "4 measurements per flower"],
              [<M>K</M>, "number of classes (or outputs)", "3 iris species, 10 digits"],
              [<M>P</M>, "number of parameters", "D + 1 for linear regression"],
              [<M>B</M>, "minibatch size", "100 examples per gradient step"],
              [<M>H</M>, "number of hidden units in a layer", "32 neurons"],
              [<><M>n</M>, <M>d</M>, <M>k</M></>, "the index of one example, one feature, one class", <><M>{t`x_{n,d}`}</M> = feature d of example n</>],
              [<M>{t`\theta`}</M>, "all the parameters together (theta)", "(w, b)"],
              [<><M>w</M>, <M>b</M></>, "weights and bias", <M>{t`\hat y = wx + b`}</M>],
              [<M>{t`\eta`}</M>, "the learning rate (eta)", "0.1"],
              [<M>{t`\lambda`}</M>, "the regularization strength (lambda)", "0.01"],
            ]}
          />
          <H>The symbols, decoded</H>
          <div className="table-wrap">
            <Table
              head={["Symbol", "Say", "Means", "NumPy"]}
              rows={[
                [<M>{t`x_n`}</M>, "x sub n", "the n-th example (subscripts are indices)", <code>X[n]</code>],
                [<M>{t`\mathbf x`}</M>, "bold x", "a vector (bold lowercase)", <code>x</code>, ],
                [<M>{t`\mathbf X`}</M>, "bold capital X", "a matrix (bold capital)", <code>X</code>],
                [<M>{t`\mathbf X\in\mathbb R^{N\times D}`}</M>, "X is a real N by D matrix", <>X has N rows and D columns of real numbers; <M>\in</M> means "is in" or "has the type"</>, <code>X.shape == (N, D)</code>],
                [<M>{t`\hat y`}</M>, "y hat", "a prediction (the hat means 'estimated')", <code>y_hat</code>],
                [<M>{t`\tilde{\mathbf x}`}</M>, "x tilde", "x with a 1 appended (for the bias)", <code>np.append(x, 1)</code>],
                [<M>{t`\mathbf x^\top`}</M>, "x transpose", "column ↔ row; for a matrix, rows ↔ columns", <code>X.T</code>],
                [<M>{t`\mathbf a^\top\mathbf b`}</M>, "a transpose b", "dot product: multiply entry by entry, add up", <code>a @ b</code>],
                [<M>{t`\sum_{n=1}^N a_n`}</M>, "sum over n from 1 to N of a n", "add up a₁ + a₂ + … + a_N", <code>a.sum()</code>],
                [<M>{t`\frac1N\sum_{n=1}^N a_n`}</M>, "one over N times the sum", "the average (mean)", <code>a.mean()</code>],
                [<M>{t`\odot`}</M>, "element-wise product", "multiply entry by entry (same shapes)", <code>a * b</code>],
                [<M>{t`\|\theta\|_2^2`}</M>, "squared L2 norm of theta", "sum of the squared entries", <code>(theta ** 2).sum()</code>],
                [<M>{t`\frac{\partial J}{\partial\theta_m}`}</M>, "partial J by partial theta m", "slope of J when only θₘ moves", "—"],
                [<M>{t`\nabla_\theta J`}</M>, "gradient of J with respect to theta", "all the partial derivatives as a vector, same shape as θ", <code>grad</code>],
                [<M>{t`\arg\min_\theta J(\theta)`}</M>, "arg min over theta of J", "the θ that makes J smallest (not the smallest value itself)", "—"],
                [<M>{t`\theta_t,\ \theta_{t+1}`}</M>, "theta at step t, t plus 1", "the parameters before and after one update", <code>theta = theta - lr * grad</code>],
                [<M>{t`\ln x,\ e^x`}</M>, "natural log of x, e to the x", "the natural logarithm and the exponential (inverse of each other)", <code>np.log(x), np.exp(x)</code>],
                [<M>{t`\sigma(z)`}</M>, "sigma of z", "the sigmoid function", <code>1 / (1 + np.exp(-z))</code>],
                [<M>{t`\mathbb 1[\hat y_n = y_n]`}</M>, "indicator", "1 if the condition is true, else 0", <code>(y_hat == y)</code>],
                [<M>{t`\lfloor x\rfloor`}</M>, "floor of x", "round down to a whole number", <code>np.floor(x)</code>],
                [<M>{t`f:\mathbb R^D\to\mathbb R^K`}</M>, "f maps R D to R K", "f takes D numbers in and gives K numbers out", "—"],
                [<M>{t`\mathbf W^{(1)}`}</M>, "W superscript one", "the weights of layer 1 (a label, not a power)", <code>model[0].weight</code>],
              ]}
            />
          </div>
          <H>How to read any formula</H>
          <Steps
            title="Example: the mean squared error"
            steps={[
              { title: "Find what's being defined (left of the =)", body: <><MB>{t`J(\theta) = \frac1N\sum_{n=1}^N(\hat y_n - y_n)^2`}</MB><p>J of θ: the loss, as a function of the parameters.</p></> },
              { title: "Find the loop: which index runs, and from where to where?", body: <p><M>{t`\sum_{n=1}^N`}</M>: n runs over all N examples. Everything with a subscript n belongs to one example.</p> },
              { title: "Read the inside for one example", body: <p><M>{t`(\hat y_n - y_n)^2`}</M>: the prediction for example n minus its label, squared.</p> },
              { title: "Read the outside", body: <p><M>{t`\frac1N\sum`}</M> = add up and divide by N = average. So: "the average, over all examples, of the squared prediction error".</p> },
              { title: "Translate to code", body: <p><code>J = ((y_hat - y) ** 2).mean()</code>. The loop over n disappears because NumPy works on all entries at once.</p> },
              { title: "Check the shapes", body: <p><code>y_hat</code> and <code>y</code> are both (N,); the result is one number. If they had different shapes, broadcasting could silently give an (N, N) array.</p> },
            ]}
          />
          <PyCell
            code={py`
import numpy as np
y_hat = np.array([0.4, 1.0, 1.6])     # predictions for N = 3 examples
y = np.array([0.2, 1.1, 1.7])         # labels
N = len(y)

J_loop = 0.0
for n in range(N):                    # the sum, written as a loop
    J_loop += (y_hat[n] - y[n]) ** 2
J_loop = J_loop / N                   # ... divided by N

J_vec = ((y_hat - y) ** 2).mean()     # the same thing, vectorized
print(round(J_loop, 4), round(J_vec, 4))
`}
          />
          <Callout kind="intuition">
            Most formulas in this course are one of three shapes: an <strong>average over examples</strong> (losses), a
            <strong> weighted sum over features</strong> (a linear model: <M>{t`\theta^\top\tilde{\mathbf x}`}</M>), or an
            <strong> update rule</strong> (new θ = old θ minus something). Spot which one you're looking at first.
          </Callout>
          <H>Practise</H>
          <Think q={<>Read this out loud in plain words: <M>{t`\text{Acc} = \frac1N\sum_{n=1}^N\mathbb 1[\hat y_n = y_n]`}</M></>}>
            "The average over all examples of 1 if the prediction is right, 0 if it's wrong": the fraction of correct predictions.
            In code: <code>(y_hat == y).mean()</code>.
          </Think>
          <Think q={<>What does <M>{t`\theta_{t+1} = \theta_t - \eta\nabla_\theta J(\theta_t)`}</M> say?</>}>
            "The new parameters are the old ones minus the learning rate times the gradient of the loss at the old parameters": one
            gradient-descent step. In code: <code>theta = theta - lr * grad</code>.
          </Think>
          <Think q={<>If <M>{t`\mathbf X\in\mathbb R^{N\times D}`}</M> and <M>{t`\mathbf w\in\mathbb R^D`}</M>, what is the shape of <M>{t`\mathbf X\mathbf w`}</M>, and what does entry n hold?</>}>
            Shape (N,): one number per example. Entry n is the dot product of example n's features with the weights,
            <M>{t`\ \mathbf x_n^\top\mathbf w`}</M>.
          </Think>
          <p>
            If the <em>Python</em> in the code cells is what's unfamiliar, read{" "}
            <a href="#/labs/python">Python you need to read the lab code</a> next; then <a href="#/labs/numpy">NumPy from zero</a>.
          </p>
        </>
      ),
    },
    {
      id: "vocabulary",
      title: "The ML vocabulary in plain English",
      summary: "The everyday words of machine learning, each with a concrete example, and the whole pipeline in one paragraph.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            The lectures use these words from the first slide on, without stopping to define them. Each one is explained below
            with one everyday example: predicting house prices. Harder terms are listed in the{" "}
            <a href="#/glossary">glossary</a>, and every section shows a "Terms on this page" strip at the top. Click a term in
            it to see its definition.
          </p>
          <H>The whole pipeline in one paragraph</H>
          <p>
            You collect <strong>examples</strong> (houses), each described by <strong>features</strong> (size, age) and a
            <strong> label</strong> (the price). You pick a <strong>model</strong>, a formula with adjustable
            <strong> parameters</strong> (price ≈ w · size + b). A <strong>loss</strong> measures how wrong its
            <strong> predictions</strong> are on the <strong>training set</strong>. <strong>Training</strong> adjusts the
            parameters to make the loss small, usually by gradient descent on <strong>minibatches</strong>, one
            <strong> epoch</strong> after another. Finally you measure the error on a <strong>test set</strong> the model never
            saw, to estimate how well it <strong>generalizes</strong>.
          </p>
          <H>The words</H>
          <div className="table-wrap">
            <Table
              head={["Word", "Meaning", "House-price example"]}
              rows={glossary
                .filter((g) => g.basic)
                .map((g) => [<strong>{g.term}</strong>, g.plain, g.example ?? ""])}
            />
          </div>
          <Callout kind="mistake">
            "Parameter" and "hyperparameter" are different things. <strong>Parameters</strong> (weights, biases) are learned by
            training. <strong>Hyperparameters</strong> (learning rate, number of epochs, number of hidden units) are chosen by
            you, using the validation set.
          </Callout>
          <H>Practise</H>
          <Think q="A data set has 500 emails, each described by 40 word counts, and each marked spam or not spam. What are N, D and K, and is this regression or classification?">
            N = 500 examples, D = 40 features, K = 2 classes. The label is a category, so it's classification (binary).
          </Think>
          <Think q="Training loss keeps going down, test loss goes down and then up again. What's happening?">
            Overfitting: after a point, the model fits noise in the training data that doesn't carry over to new data. Early
            stopping would keep the parameters from where the validation loss was lowest.
          </Think>
        </>
      ),
    },
  ],
};
