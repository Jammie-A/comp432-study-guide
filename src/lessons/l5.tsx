import { Callout, FromLab, Formula, H, M, MB, MathCode, Slides, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { ClipWidget, ConvWidget, DropoutWidget, NormWidget, VanishingWidget } from "../widgets/l5";
import { Lecture } from "./types";
import { Private } from "../private";

const t = String.raw;
const py = String.raw;

export const lecture5: Lecture = {
  id: "l5",
  number: 5,
  title: "Deep networks in practice, and CNNs",
  sections: [
    {
      id: "vanishing",
      title: "Vanishing & exploding gradients",
      summary: "Backprop multiplies one factor per layer. Many factors below 1 vanish; above 1 they explode.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            In a deep network, the first layers are far from the loss. Their gradient is a long product of local derivatives,
            one per layer (Lecture 4's chain rule). If those factors are typically below 1, the product shrinks exponentially
            and early layers barely learn (<strong>vanishing gradient</strong>, common). Above 1, it blows up
            (<strong>exploding gradient</strong>, rarer).
          </p>
          <H>The scalar chain</H>
          <Formula
            plain="A deep chain with one neuron per layer: each layer multiplies by a weight, adds a bias and applies the sigmoid; the loss compares the last output with y."
            sheet="no"
            tex={t`h_0 = x,\quad z_k = w_kh_{k-1} + b_k,\quad h_k = \sigma(z_k),\quad J = (h_L - y)^2`}
            symbols={[
              [t`\frac{dJ}{dz_k} = \frac{dJ}{dh_k}\,\sigma'(z_k)`, "backward through the sigmoid"],
              [t`\frac{dJ}{dh_{k-1}} = \frac{dJ}{dz_k}\,w_k`, "backward through the affine map"],
              [t`\frac{\partial J}{\partial w_k} = \frac{dJ}{dz_k}\,h_{k-1}`, "the weight's gradient"],
            ]}
          />
          <MB>{t`\frac{\partial J}{\partial w_1} = \frac{dJ}{dh_L}\cdot \underbrace{w_L\sigma'(z_L)}_{\text{layer }L}\cdots\underbrace{w_2\sigma'(z_2)}_{\text{layer }2}\cdot\sigma'(z_1)\,x`}</MB>
          <Steps
            title="The slide's numbers: 4 sigmoid layers, x = ½, y = 1, biases −w/2"
            steps={[
              { title: "Every pre-activation is 0", body: <MB>{t`z_1 = w\cdot\tfrac12 - \tfrac w2 = 0 \Rightarrow h_1 = \sigma(0) = \tfrac12,\ \text{and the same in every layer}`}</MB> },
              { title: "So every sigmoid derivative is ¼", body: <MB>{t`\sigma'(0) = \sigma(0)(1-\sigma(0)) = \tfrac12\cdot\tfrac12 = \tfrac14`}</MB> },
              { title: "Start at the loss", body: <MB>{t`\frac{dJ}{dh_4} = 2(h_4 - y) = 2(\tfrac12 - 1) = -1`}</MB> },
              { title: "Vanishing (w = 1): each layer multiplies by w·¼ = ¼", body: <MB>{t`\frac{dJ}{dh_k}:\quad -1 \to -\tfrac14 \to -\tfrac1{16} \to -\tfrac1{64}`}</MB> },
              { title: "Exploding (w = 8): each layer multiplies by 8·¼ = 2", body: <MB>{t`-1 \to -2 \to -4 \to -8`}</MB> },
              { title: "Over depth L", body: <p>∂J/∂w₁ ÷ ∂J/∂w_L = factor^(L−1). With ¼ and L = 20: about 4·10⁻¹². The first layer effectively stops learning.</p> },
            ]}
          />
          <VanishingWidget />
          <Callout kind="intuition">
            In real networks, each factor is a Jacobian matrix and the factors differ per layer, but the long product behaves
            the same way. Measured on a 20-layer net: with sigmoid the gradient norm shrinks by a constant factor per layer;
            with ReLU it stays level. The loss landscape gets very steep <strong>cliffs</strong> (exploding) and very flat
            <strong> plateaus</strong> (vanishing).
          </Callout>
          <H>Fix for exploding gradients: gradient clipping</H>
          <Formula
            plain="If the gradient is longer than C, shrink it to length C, keeping its direction; otherwise leave it as it is."
            sheet="no"
            tex={t`\mathbf g \leftarrow \mathbf g\cdot\min\left(1,\ \frac{C}{\|\mathbf g\|}\right)`}
            symbols={[
              [t`\mathbf g`, "the gradient of ALL parameters, as one vector"],
              [t`C`, "threshold, set by trial and error"],
              [t`\min(1, \cdot)`, "if ‖g‖ ≤ C nothing changes; otherwise g is rescaled to length exactly C. Direction is kept."],
            ]}
          />
          <Steps
            title="Slide example: C = 2.5, g = (3, 4)"
            steps={[
              { title: "Norm", body: <MB>{t`\|\mathbf g\| = \sqrt{3^2 + 4^2} = 5`}</MB> },
              { title: "Factor", body: <MB>{t`\min(1,\ 2.5/5) = 0.5`}</MB> },
              { title: "Clipped", body: <MB>{t`\mathbf g = 0.5\cdot(3,4) = (1.5,\ 2)`}</MB> },
            ]}
          />
          <ClipWidget />
          <PyCell
            code={py`
import numpy as np
def clip(g, C):
    return g * min(1.0, C / np.linalg.norm(g))

print(clip(np.array([3.0, 4.0]), 2.5))   # too long → rescaled
print(clip(np.array([0.3, 0.4]), 2.5))   # short enough → unchanged
# PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), C), between backward() and step()
`}
          />
          <FromLab to="labs/debugging" label="Debugging neural networks">
            The debugging tutorial's last step: if the loss turns into <code>NaN</code> during training, first try a smaller
            learning rate, then gradient clipping (<code>torch.nn.utils.clip_grad_norm_</code>, called between{" "}
            <code>backward()</code> and <code>step()</code>).
          </FromLab>
          <Slides lecture={5} pages="7–14" />
        </>
      ),
    },
    {
      id: "remedies",
      title: "Remedies: ReLU and shortcuts (ResNets)",
      summary: "Use non-saturating activations, and give the gradient a path that skips layers.",
      body: () => (
        <>
          <H>Remedy 1: non-saturating activations</H>
          <p>
            Sigmoid and tanh have two <strong>saturation zones</strong> where f′(z) ≈ 0; any neuron there contributes a near-zero
            factor to the gradient chain. So avoid them in hidden layers. Use <strong>ReLU</strong> (derivative exactly 1 for
            z &gt; 0, so the gradient flows unchanged) or <strong>leaky ReLU</strong>:
          </p>
          <MB>{t`\mathrm{LeakyReLU}(z) = \begin{cases} z & z > 0\\ \alpha z & z\le 0\end{cases},\quad \alpha\approx 0.01\text{–}0.1,\qquad f'(z) = 1 \text{ or } \alpha`}</MB>
          <p>ReLU can vanish only on one side (z &lt; 0); leaky ReLU never gives exactly 0.</p>
          <H>Remedy 2: gradient shortcuts</H>
          <Formula
            plain="A skip connection adds a block's input to its output. In the backward pass the gradient then has a direct path through the addition, besides the path through the block, so it can't vanish completely."
            sheet="no"
            tex={t`\text{forward: } \mathbf s = \mathbf h^{(1)} + \mathbf h^{(2)},\qquad \text{backward: } \nabla_{\mathbf h^{(1)}}J = \underbrace{\nabla_{\mathbf s}J}_{\text{skipped }f^{(2)}} + (\mathbf J_{\mathbf h^{(1)}}\mathbf h^{(2)})^\top\nabla_{\mathbf s}J`}
            symbols={[
              ["+", <>a <strong>sum node</strong> copies its incoming gradient to <em>both</em> of its inputs</>],
              [t`\nabla_{\mathbf s}J`, "arrives at h⁽¹⁾ directly, without being multiplied by f⁽²⁾'s Jacobian"],
            ]}
          />
          <Callout kind="intuition">
            In the one-unit chain, the factor <M>{t`w_k\sigma'(z_k)`}</M> becomes <M>{t`1 + w_k\sigma'(z_k)`}</M>. Even if the layer's own
            factor is tiny, the 1 passes the gradient on. Try the "shortcut" checkbox in the vanishing-gradient visual on the
            previous page.
          </Callout>
          <Table
            head={["Flavor", "Wiring", "Notes"]}
            rows={[
              ["Residual network (ResNet)", <M>{t`\mathbf h \mapsto \mathbf h + f(\mathbf h)`}</M>, "shortcut across each block; allows 100 or even 1000 layers"],
              ["Skips to the output", "every hidden layer also feeds the last one", "the output sums the features of all layers"],
              ["DenseNet", "each layer gets all previous layers' features", "concatenated, not summed"],
            ]}
          />
          <H>The residual block</H>
          <MB>{t`\mathbf h' = \mathrm{ReLU}\big(\mathbf h + f(\mathbf h)\big),\qquad f = \text{Conv} \to \text{Norm} \to \text{ReLU} \to \text{Conv} \to \text{Norm}`}</MB>
          <p>
            The block learns the <strong>residual</strong> f(h) = output − input. With f = 0 it is the identity, so adding blocks
            never costs representational power. Depth over the years: LeNet (1998) 5 layers, AlexNet (2012) 8, VGG (2014) 19,
            ResNet (2015) 152. ResNet-18: 8 blocks → 1 + 2·8 + 1 = 18 weight layers.
          </p>
          <PyCell
            code={py`
import numpy as np
sig_prime = 0.25           # σ'(0)
w, L = 1.0, 10
plain = (w * sig_prime) ** (L - 1)
residual = (1 + w * sig_prime) ** (L - 1)
print(f"plain chain   : first/last gradient ratio = {plain:.2e}")
print(f"with shortcuts: first/last gradient ratio = {residual:.2f}")
`}
          />
          <Callout kind="exam">
            Know <em>why</em> each remedy works in terms of the chain-rule product: ReLU makes factors 1 for active units, shortcuts
            add a 1 to each factor, and normalization (next) keeps the inputs of every layer out of the saturation zones.
          </Callout>
          <Slides lecture={5} pages="16–19, 48" />
        </>
      ),
    },
    {
      id: "normalization",
      title: "Normalization: standardization, batch norm, layer norm",
      summary: "Put every feature, and every hidden unit, on a common scale so one learning rate suits all weights.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            In a linear model, <M>{t`\partial J/\partial w_d = \frac2N\sum_n r_nx_{n,d}`}</M>: a weight's gradient scales with the
            feature it multiplies. If <M>{t`x_2 \gg x_1`}</M>, then <M>{t`|\partial J/\partial w_2| \gg |\partial J/\partial w_1|`}</M>, no single
            learning rate suits both, and GD zig-zags (Lecture 2's narrow valley). Every layer of a network is such a linear
            model fed by the previous layer, so the problem repeats at every layer.
          </p>
          <H>Input standardization</H>
          <Formula
            plain="For each feature, subtract its mean over the training examples and divide by its standard deviation, so every feature has mean 0 and spread 1."
            sheet="no"
            tex={t`\mu_d = \frac1N\sum_{n=1}^N x_{n,d},\qquad \sigma_d^2 = \frac1N\sum_{n=1}^N(x_{n,d}-\mu_d)^2,\qquad \hat x_{n,d} = \frac{x_{n,d}-\mu_d}{\sigma_d}`}
            symbols={[
              [t`\mu_d, \sigma_d`, "mean and standard deviation of feature d, computed on the TRAINING set"],
              [t`\hat x_{n,d}`, "shifted to mean 0, then scaled to variance 1"],
            ]}
          />
          <Steps
            title="Standardize x = (2, 4, 6)"
            steps={[
              { title: "Mean", body: <MB>{t`\mu = (2+4+6)/3 = 4`}</MB> },
              { title: "Variance (divide by N, not N − 1)", body: <MB>{t`\sigma^2 = \tfrac13\big((-2)^2 + 0^2 + 2^2\big) = \tfrac83,\quad \sigma \approx 1.63`}</MB> },
              { title: "Standardize", body: <MB>{t`\hat x = (-2, 0, 2)/1.63 \approx (-1.22,\ 0,\ 1.22)`}</MB> },
            ]}
          />
          <Callout kind="mistake">
            Compute μ and σ on the training set only, then reuse those same numbers for validation and test data. Computing
            them on the test set leaks test information.
          </Callout>
          <H>Batch normalization (BN)</H>
          <Formula
            plain="The same standardization, but inside the network, for each unit, using the current minibatch's mean and variance; then a learned scale γ and shift β let the network undo it where that helps."
            sheet="no"
            tex={t`\mu_{\mathcal B,d} = \frac1B\sum_{n=1}^B z_{n,d},\quad \sigma^2_{\mathcal B,d} = \frac1B\sum_{n=1}^B(z_{n,d}-\mu_{\mathcal B,d})^2,\quad \hat z_{n,d} = \frac{z_{n,d}-\mu_{\mathcal B,d}}{\sqrt{\sigma^2_{\mathcal B,d}+\epsilon}},\quad \tilde z_{n,d} = \gamma_d\hat z_{n,d} + \beta_d`}
            symbols={[
              [t`z_{n,d}`, "pre-activation of hidden unit d for sample n of the minibatch"],
              [t`\mu_{\mathcal B}, \sigma_{\mathcal B}`, "one entry PER UNIT, averaged over the samples of the minibatch (the batch axis)"],
              [t`\epsilon`, "small constant to avoid dividing by zero"],
              [t`\gamma_d, \beta_d`, "learnable scale and shift per unit, start at γ = 1, β = 0 (identity)"],
            ]}
          />
          <p>
            Complications compared to input standardization: hidden features change during training, and SGD only sees
            minibatches. At <strong>inference</strong> there may be no batch at all (a single image), so BN keeps
            <strong> running averages</strong> during training and uses them at inference:
          </p>
          <MB>{t`\mu_{\text{run}} \leftarrow \rho\,\mu_{\text{run}} + (1-\rho)\,\mu_{\mathcal B},\qquad \sigma^2_{\text{run}} \leftarrow \rho\,\sigma^2_{\text{run}} + (1-\rho)\,\sigma^2_{\mathcal B}`}</MB>
          <p className="muted">(PyTorch's <code>momentum</code> argument is 1 − ρ, default 0.1.)</p>
          <H>Layer normalization (LN)</H>
          <p>
            Same formulas, axis swapped: one mean and variance <strong>per sample</strong>, over its units. No minibatch
            statistics → no running averages → training and inference are identical. LN is the default in transformers and
            RNNs; BN in CNNs.
          </p>
          <NormWidget />
          <MathCode
            rows={[
              ["\\text{BN: }\\mu_{\\mathcal B}", "Z.mean(axis=0)", "(units,) — over samples"],
              ["\\text{LN: }\\mu", "Z.mean(axis=1)", "(samples,) — over units"],
              ["\\sigma \\text{ (divide by N)}", "Z.std(axis=0)", "NumPy's default ddof=0 = PyTorch correction=0"],
            ]}
          />
          <PyCell
            code={py`
import numpy as np
Z = np.array([[ 0.2, 12.0, 3.0],
              [-0.4,  9.0, 1.0],
              [ 0.6, 14.0, 2.0],
              [-0.4,  5.0, 6.0]])      # (B=4 samples, 3 units)

mu, sigma = Z.mean(axis=0), Z.std(axis=0)            # batch norm: per column
print("BN mu", mu.round(2) + 0.0, " sigma", sigma.round(2))   # + 0.0 turns -0. into 0.
print(((Z - mu) / sigma).round(2))

mu, sigma = Z.mean(axis=1, keepdims=True), Z.std(axis=1, keepdims=True)   # layer norm: per row
print("LN mu", mu.ravel().round(2))
print(((Z - mu) / sigma).round(2))
`}
          />
          <Think q="A model with batch norm gives a different prediction for the same image depending on which other images are in the batch. What did the user forget?">
            To call <code>model.eval()</code>. In training mode BN normalizes with the current minibatch's statistics, so the other
            images change the result. In eval mode it uses the stored running averages, so a prediction depends only on its own input.
          </Think>
          <FromLab to="labs/debugging" label="Debugging neural networks">
            Lab 4 rescaled the MNIST pixels with <code>MinMaxScaler</code>: <code>fit_transform</code> on the training set,
            only <code>transform</code> on the test set, the same "training-set statistics only" rule as here. The debugging
            tutorial shows batch norm going wrong: a final minibatch of 2 examples normalizes every feature to ±1.
          </FromLab>
          <Slides lecture={5} pages="22–30" />
        </>
      ),
    },
    {
      id: "regularization",
      title: "Regularizing deep networks",
      summary: "Augment inputs, smooth targets, average models, drop units — and know what switches off at inference.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            Deep networks can memorize the training set. <strong>Regularization</strong> is any change to training that aims at
            better generalization, often at the expense of training fit. There are four places to intervene:
            the <strong>inputs</strong> (augmentation), the <strong>targets</strong> (label smoothing), the <strong>model</strong>
            (ensembles, dropout) and the <strong>parameters</strong> (L2, Lecture 4).
          </p>
          <H>Inputs: data augmentation</H>
          <p>
            More data is the most reliable regularizer. If you can't collect it, create it: transform inputs in ways that
            <strong> keep the label</strong>. Images: crop, flip, rotate, scale, noise, hue. Audio: noise, time shift, stretch.
          </p>
          <Callout kind="mistake">
            The transformation must not change the correct label. Rotating a "6" by 180° makes it a "9"; flipping an arrow
            pointing left makes it point right. Exam sample question 2 tests exactly this.
          </Callout>
          <H>Targets: label smoothing</H>
          <Formula
            plain="Replace the one-hot target by a softened one: the true class gets 1 − ε + ε/K and every other class ε/K. The model is no longer pushed towards infinite confidence."
            sheet="no"
            tex={t`\tilde{\mathbf y} = (1-\epsilon)\,\mathbf y + \frac{\epsilon}{K}`}
            symbols={[
              [t`\mathbf y`, "one-hot label"],
              [t`\epsilon`, "smoothing amount (0.1 is a common default)"],
              [t`\epsilon/K`, "spread uniformly over all K classes"],
            ]}
          />
          <Steps
            title="K = 3, ε = 0.15, y = (0, 1, 0)"
            steps={[
              { title: "Scale the one-hot part", body: <MB>{t`(1 - 0.15)(0,1,0) = (0,\ 0.85,\ 0)`}</MB> },
              { title: "Add ε/K = 0.05 everywhere", body: <MB>{t`\tilde{\mathbf y} = (0.05,\ 0.90,\ 0.05)`}</MB> },
              { title: "Effect", body: <p>The soft label goes into the cross-entropy, so the model is never pushed to output probability exactly 1: it becomes less overconfident.</p> },
            ]}
          />
          <H>Model: ensembles and dropout</H>
          <p>
            <strong>Ensemble:</strong> train E models independently, average their predictions <M>{t`\hat y = \frac1E\sum_e\hat y_e`}</M>.
            Different initializations and data orders make their errors partly cancel. Cost: E trainings and E forward passes.
          </p>
          <Formula
            plain="During training, set each hidden unit to 0 with probability p and divide the survivors by 1 − p, so the expected value is unchanged. At test time, do nothing."
            sheet="no"
            title="Dropout (training only)"
            tex={t`\mathbf h \mapsto \mathbf m\odot\mathbf h/(1-p),\qquad m_i = 0 \text{ with probability } p`}
            symbols={[
              [t`p`, "drop probability"],
              [t`\mathbf m`, "random 0/1 mask, a NEW one per sample and training step"],
              [t`1/(1-p)`, "rescales the survivors so each unit's expected value stays h"],
            ]}
          />
          <p>Slide example, p = 0.5:</p>
          <MB>{t`(0,1,0,1)\odot(0.5,0.2,0.0,1.4)/0.5 = (0,\ 0.4,\ 0,\ 2.8)`}</MB>
          <DropoutWidget />
          <Table
            head={["", "Ensemble", "Dropout"]}
            rows={[
              ["Members", "E separate networks", "sub-networks of one network"],
              ["Weights", "independent", "shared"],
              ["Training", "E trainings", "one training"],
              ["Prediction", "average of E passes", "one pass with all units (approximate)"],
            ]}
          />
          <H>Training vs inference: what switches off?</H>
          <Table
            head={["", "Training", "Inference / validation"]}
            rows={[
              ["Dropout", "masks, scaled by 1/(1 − p)", "off, all units"],
              ["Batch norm", "minibatch μ_B, σ_B", "running averages"],
              ["Augmentation", "random transforms", "off, plain input"],
              ["Standardization", "training-set μ, σ", "the same μ, σ"],
              ["Layer norm", "per-sample statistics", "the same"],
              ["Penalties (L2)", "change the loss", "no effect"],
            ]}
          />
          <Callout kind="exam">
            <code>model.eval()</code> switches dropout and batch norm to inference mode; <code>model.train()</code> switches back.
            Augmentation lives in the data pipeline, so you must turn it off yourself.
          </Callout>
          <PyCell
            code={py`
import numpy as np
rng = np.random.default_rng(0)
h = np.array([0.5, 0.2, 0.0, 1.4])
p = 0.5
means = []
for step in range(10000):
    m = (rng.random(h.shape) >= p).astype(float)   # keep with probability 1 - p
    means.append(m * h / (1 - p))
print("one mask       :", means[0])
print("average output :", np.mean(means, axis=0).round(2), " ≈ h =", h)

y = np.array([0, 1, 0]); eps, K = 0.15, 3
print("label smoothing:", (1 - eps) * y + eps / K)
`}
          />
          <FromLab to="labs/pytorch" label="Training an MLP in NumPy and PyTorch">
            The lab training loops call <code>model.train()</code> before training and <code>model.eval()</code> before
            evaluating, which switches dropout and batch norm. Evaluation also runs inside <code>with torch.no_grad():</code>.
            That's a different switch: it stops PyTorch from building a computation graph, which saves memory, and does
            nothing to dropout or batch norm.
          </FromLab>
          <Slides lecture={5} pages="32–38" />
        </>
      ),
    },
    {
      id: "convolution",
      title: "Convolutions: kernels, padding, stride, pooling",
      summary: "Look locally and share weights: slide one small kernel across the input.",
      body: () => (
        <>
          <H>Why not an MLP on images?</H>
          <ul>
            <li><strong>Too many weights</strong>: 224 × 224 × 3 pixels into 500 units is 224·224·3·500 ≈ 75 million weights (300 MB) in one layer.</li>
            <li><strong>Position is baked in</strong>: every pixel has its own weights, so a pattern learned in one place doesn't carry over to another.</li>
          </ul>
          <p>A convolution fixes both with <strong>local connections</strong> (each output looks at a small window) and <strong>weight sharing</strong> (the same kernel at every position). Shifting the input shifts the output.</p>
          <H>The 1-D convolution</H>
          <Formula
            plain="Output i is the dot product of the F kernel weights with a window of F consecutive inputs, starting at position (i − 1)s + 1, plus a bias. The same weights are reused at every position."
            sheet="no"
            tex={t`z_i = \sum_{j=1}^{F} w_j\,x_{(i-1)s + j} + b \qquad(\text{input already zero-padded})`}
            symbols={[
              [t`\mathbf w = (w_1,\dots,w_F)`, <>the <strong>kernel</strong> (filter) of size F, learned like MLP weights; F + 1 parameters with the bias</>],
              [t`s`, <><strong>stride</strong>: how far the window jumps per output</>],
              [t`P`, <><strong>padding</strong>: zeros added per side. <M>{t`P = (F-1)/2`}</M> (odd F, stride 1) keeps the length ("same")</>],
            ]}
          />
          <Formula
            plain="Pad the input with P zeros on each side (length D + 2P), then count how many positions a window of size F can start at when it jumps s steps at a time."
            sheet="no"
            title="Output length"
            tex={t`D_{\text{out}} = \left\lfloor\frac{D + 2P - F}{s}\right\rfloor + 1`}
            symbols={[[t`\lfloor\cdot\rfloor`, "round down"]]}
          />
          <Think q="Where does ⌊(D + 2P − F)/s⌋ + 1 come from?">
            After padding the input has D + 2P entries. A window of F entries fits at start positions 0, s, 2s, … as long as
            its last entry stays inside, i.e. while the start is at most D + 2P − F. The number of multiples of s from 0 to
            D + 2P − F is <M>{t`\lfloor(D+2P-F)/s\rfloor`}</M>, plus 1 for the start at 0. With s = 1 and P = 0 this is the familiar
            D − F + 1.
          </Think>
          <Steps
            title="Slide example: x = (3, 5, 2, −1, 0, 4, −2, −3), kernel (−1, 0, 1), P = 1, s = 1"
            steps={[
              { title: "Pad with one zero per side", body: <MB>{t`(0,\ 3,\ 5,\ 2,\ -1,\ 0,\ 4,\ -2,\ -3,\ 0)`}</MB> },
              { title: "Output length", body: <MB>{t`\lfloor(8 + 2 - 3)/1\rfloor + 1 = 8`}</MB> },
              { title: "z₁: window (0, 3, 5)", body: <MB>{t`-1\cdot0 + 0\cdot3 + 1\cdot5 = 5`}</MB> },
              { title: "z₂: window (3, 5, 2)", body: <MB>{t`-1\cdot3 + 0\cdot5 + 1\cdot2 = -1`}</MB> },
              { title: "All outputs", body: <MB>{t`\mathbf z = (5,\ -1,\ -6,\ -2,\ 5,\ -2,\ -7,\ 2)`}</MB> },
              { title: "With stride 2", body: <p>Every second output remains: (5, −6, 5, −7), and <M>{t`\lfloor 7/2\rfloor + 1 = 4`}</M> ✓.</p> },
            ]}
          />
          <ConvWidget />
          <Callout kind="note">
            Strictly, this is a <em>cross-correlation</em> (a true convolution flips the kernel), but CNNs call it convolution anyway.
            The kernel's weights decide the effect: all-ones blurs, a difference kernel finds edges. CNNs <em>learn</em> them.
          </Callout>
          <H>Pooling</H>
          <p>
            Downsamples without a kernel: summarize each window by its <strong>max</strong> (max pooling) or <strong>average</strong>.
            No parameters (window size and stride are hyperparameters). Keeps the strongest response, so small shifts of the input
            change the output little. Slide: max pooling (size 2, stride 2) on the padded input above gives (3, 5, 0, 4, 0).
          </p>
          <H>Channels</H>
          <p>
            One kernel detects one pattern → one <strong>output channel</strong>. For C_out patterns, use C_out kernels. With
            C_in input channels (stereo audio, RGB), each kernel has F × C_in weights and still yields one channel. PyTorch:
            <code> Conv1d(C_in, C_out, F, padding="same")</code> maps (B, C_in, D) → (B, C_out, D).
          </p>
          <PyCell
            code={py`
import numpy as np
x = np.array([3, 5, 2, -1, 0, 4, -2, -3], dtype=float)
w = np.array([-1, 0, 1], dtype=float)
P, s, F = 1, 1, len(w)

xp = np.pad(x, P)                                   # zero padding
D_out = (len(x) + 2 * P - F) // s + 1
z = np.array([w @ xp[i * s : i * s + F] for i in range(D_out)])
print("padded:", xp)
print("output:", z, " length", D_out)
print("max pool (2, stride 2):", xp.reshape(-1, 2).max(axis=1))
`}
          />
          <FromLab to="labs/conv-code" label="Convolutions in code">
            The convolution tutorial wrote the 2-D version as four nested loops,{" "}
            <M>{t`a_{i,j} = \sum_{u,v} w_{u,v}\,x_{i+u,j+v}`}</M>, and checked it against <code>torch.nn.Conv2d</code>. The
            hand-made kernel <code>[[-1, 1], [-1, 1]]</code> lights up vertical edges, and its transpose lights up horizontal
            ones. Without padding, the output shrinks to (H − F + 1) × (W − F + 1).
          </FromLab>
          <Slides lecture={5} pages="40–46" />
        </>
      ),
    },
    {
      id: "cnn",
      title: "CNN architectures & counting parameters",
      summary: "Stack conv blocks, add a classification head, and count parameters independent of input size.",
      body: () => (
        <>
          <H>A convolutional network</H>
          <MB>{t`\mathbf x \to \underbrace{\text{Conv}\to\text{BatchNorm}\to\text{ReLU}\to\text{MaxPool}}_{\text{block 1}} \to \underbrace{\cdots}_{\text{block 2, \dots}} \to \text{Flatten}\to\text{Linear}\to\text{Softmax}\to\hat{\mathbf y}`}</MB>
          <ul>
            <li>After each convolution: (optional) normalization and a non-linearity. Batch norm in a CNN: one mean and variance <strong>per channel</strong>, over samples and positions.</li>
            <li>Pooling or stride &gt; 1 shortens the sequence. After the non-linearity, channels are called <strong>feature maps</strong>.</li>
            <li><strong>Head</strong>: Flatten stacks all channels into one vector (or global pooling: one number per channel), then a linear layer gives logits.</li>
            <li>Training is nothing new: cross-entropy (PyTorch's takes the logits), SGD and variants, autodiff.</li>
          </ul>
          <H>Counting parameters</H>
          <Formula
            plain="Each of the C_out kernels has one weight per kernel position and per input channel, plus one bias. The input length doesn't appear, because the weights are shared across positions."
            sheet="no"
            tex={t`\text{1-D: } P_{\text{conv}} = F\cdot C_{\text{in}}\cdot C_{\text{out}} + C_{\text{out}}\qquad \text{2-D: } P_{\text{conv}} = F_1\cdot F_2\cdot C_{\text{in}}\cdot C_{\text{out}} + C_{\text{out}}`}
            symbols={[
              [t`F\cdot C_{\text{in}}`, "weights of ONE kernel"],
              [t`\times C_{\text{out}}`, "one kernel per output channel"],
              [t`+ C_{\text{out}}`, "one bias per output channel"],
              ["D", "the input length does NOT appear: thanks to weight sharing"],
            ]}
          />
          <Steps
            title="Practice counts"
            steps={[
              { title: "F = 3, C_in = 1, C_out = 1", body: <MB>{t`3 + 1 = 4`}</MB> },
              { title: "F = 3, C_in = 1, C_out = 16", body: <MB>{t`3\cdot16 + 16 = 64`}</MB> },
              { title: "F = 5, C_in = 16, C_out = 32", body: <MB>{t`5\cdot16\cdot32 + 32 = 2560 + 32 = 2592`}</MB> },
              { title: "2-D: 3×3 kernel, RGB input (C_in = 3), 64 output channels", body: <MB>{t`3\cdot3\cdot3\cdot64 + 64 = 1728 + 64 = 1792`}</MB> },
              { title: "Compare: a dense layer from 224×224×3 to 500 units", body: <MB>{t`224\cdot224\cdot3\cdot500 \approx 75\ \text{million weights}`}</MB> },
            ]}
          />
          <Think q="A Conv1d(4, 8, 5) layer is applied to a signal of length 100, then to one of length 1000. How many parameters does it have in each case?">
            The same: 5·4·8 + 8 = 168 in both cases. The kernel slides over whatever length it gets; only the <em>output</em> length changes.
          </Think>
          <H>Shapes through a small 1-D CNN</H>
          <PyCell
            code={py`
# Track shapes and parameter counts, no learning needed.
D, C = 64, 1                       # input length 64, one channel
total = 0
for (F, C_out, P, s, pool) in [(5, 8, 2, 1, 2), (3, 16, 1, 1, 2)]:
    params = F * C * C_out + C_out
    D = (D + 2 * P - F) // s + 1   # conv
    D = D // pool                  # max pool, size = stride = pool
    C = C_out
    total += params
    print(f"conv F={F}: {params:4d} params -> shape ({C}, {D})")
K = 10
head = C * D * K + K               # Flatten -> Linear(C*D, K)
total += head
print(f"head Linear({C * D}, {K}): {head} params;  total = {total}")
`}
          />
          <Callout kind="exam">
            Expect "how many parameters / what output length" questions. No calculator, so the numbers will be small: write
            the formula first, then plug in.
          </Callout>
          <FromLab to="labs/conv-code" label="Convolutions in code">
            The convolution tutorial checks this page's formula in code: a <code>Conv2d</code> weight has shape (C_out, C_in, F, F),
            and <code>sum(p.numel() for p in conv.parameters())</code> equals F·F·C_in·C_out + C_out, whatever the image
            size. Lab 5's CNNs use exactly these counts, so practise tracking shapes layer by layer.
            <Private id="l5/cnn/a5-lab" quiet />
          </FromLab>
          <Slides lecture={5} pages="47–51" />
        </>
      ),
    },
    {
      id: "transfer",
      title: "Transfer learning, multitask learning & summary",
      summary: "Reuse a CNN trained on a big dataset: keep its layers, swap the head.",
      body: () => (
        <>
          <H>Why it matters</H>
          <p>
            CNNs learn hierarchical features: early layers respond to edges, colors and textures; late layers to object parts
            and whole objects (pixels → edges → parts → objects, Lecture 4's compositionality). Early features are generic, so
            they transfer between tasks. Most tasks only have a few thousand labeled images — not enough to train a deep CNN
            from scratch.
          </p>
          <H>Transfer learning</H>
          <ol>
            <li><strong>Pretrain</strong> on a big dataset, e.g. ImageNet (1.2M images, 1000 classes).</li>
            <li><strong>Transfer</strong>: keep the layers, replace the head (e.g. 1000 classes → 37 pet breeds), train on the target set.</li>
          </ol>
          <Table
            head={["Variant", "Pretrained layers", "Optimizer gets"]}
            rows={[
              ["Feature extraction", "frozen (requires_grad = False)", "only the new head's parameters"],
              ["Fine-tuning", "updated too", "all parameters, with a smaller learning rate"],
            ]}
          />
          <PyCell
            readOnly
            title="Transfer learning in PyTorch (read-only)"
            code={py`
import torch, torchvision
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1    # 11.7M parameters
model = torchvision.models.resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False                 # freeze everything
model.fc = torch.nn.Linear(512, K)          # new trainable head (created after freezing → trainable)
opt = torch.optim.SGD(model.fc.parameters(), lr=0.01)
`}
          />
          <Callout kind="mistake">
            Frozen weights still don't freeze batch norm's running statistics: they keep moving unless the model is in
            <code> eval()</code> mode. Also preprocess inputs exactly as in pretraining (<code>weights.transforms()</code>).
          </Callout>
          <H>Multitask learning</H>
          <p>
            Transfer reuses a representation <em>sequentially</em>; multitask learning trains it <em>jointly</em>: one input,
            shared early layers, one small head per task (e.g. a speech recording → speaker, words, mood). The shared layers see
            all tasks' data and often generalize better, but unrelated tasks can hurt each other (<strong>negative transfer</strong>).
          </p>
          <H>Everything at a glance</H>
          <div className="table-wrap">
            <Table
              head={["", "Linear / basis function model", "MLP", "CNN"]}
              rows={[
                ["Features", "raw or fixed by hand", "learned", "learned, local"],
                ["Objective", "convex", "non-convex", "non-convex"],
                ["Gradient", "error × features", "backpropagation", "backpropagation"],
                ["Training aids", "—", "ReLU, shortcuts, normalization, clipping", "same"],
                ["Regularizers", "L2", "L2, augmentation, label smoothing, dropout", "same"],
                ["Reuse", "—", "transfer learning: pretrained layers, new head", "same"],
              ]}
            />
          </div>
          <Slides lecture={5} pages="52–57" />
        </>
      ),
    },
  ],
};
