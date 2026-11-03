import { Callout, CodeWalk, H, M, MB, Steps, Table, Think } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { CapacityWidget, GD1DWidget, ValleyWidget } from "../widgets/l2";
import { Lecture } from "./types";

const t = String.raw;
const py = String.raw;

function Up() {
  return <span aria-label="goes up">↑</span>;
}
function Down() {
  return <span aria-label="goes down">↓</span>;
}

export const practice: Lecture = {
  id: "practice",
  number: 7,
  title: "Think it through: deep-understanding practice",
  label: "Practice",
  short: "Practice",
  sections: [
    // ------------------------------------------------------------ How to reason
    {
      id: "how-to-reason",
      title: "How to answer deep-understanding questions",
      summary: "The four kinds of question the midterm asked, a step-by-step method for each, and a table of what every common change does.",
      body: () => (
        <>
          <H>Why these questions feel hard</H>
          <p>
            Knowing a formula isn't enough for a question like "what happens if you double the learning rate?" or "why does
            this code print 230.26?". You have to <em>run the mechanism in your head</em>: follow what the change does to the
            gradient, the step, the fit to the training data, and finally the result. The good news is that the same few
            moves solve almost all of them.
          </p>
          <Table
            head={["Question type", "Example", "What to do"]}
            rows={[
              ["What happens if…", "…the learning rate is doubled? …λ is increased? …the batch size is 1?", "Find what the change acts on (step size, gradient noise, capacity, data), then follow it to training loss and validation loss."],
              ["Predict the code's result", "What does this PyTorch snippet print?", "Track shapes line by line, know each module's default behaviour, compute small numbers by hand."],
              ["Why is the result what it is / how to fix it", "The loss is stuck at 2.30. The validation loss is below the training loss.", "Compare with a baseline value, compare training with validation, then name the mechanism and the fix that targets it."],
              ["Short math", "One GD step, a cross-entropy, a parameter count", "Use the few values worth memorizing (ln 2, ln 10, σ(0)) and look for the structure that cancels."],
            ]}
          />
          <H>The method</H>
          <Steps
            title="Six moves for any 'what happens if' question"
            steps={[
              { title: "1. Locate the change", body: <p>Does it change the <strong>data</strong> (more examples, augmentation, label noise), the <strong>model</strong> (capacity: units, layers, degree), the <strong>loss</strong> (penalty, sum vs mean), the <strong>optimizer</strong> (η, momentum, batch size), or only the <strong>evaluation</strong> (train/eval mode, which set)?</p> },
              { title: "2. Follow it through one update", body: <><MB>{t`\theta \leftarrow \theta - \eta\,\mathbf g`}</MB><p>Does the step get longer or shorter? Noisier? Does it point somewhere else? That decides speed and stability.</p></> },
              { title: "3. Follow it to the two losses", body: <p>Does the change make it easier to fit the training data (training loss <Down />) and does that fit carry over (validation loss)? More capacity or less regularization: training <Down />, gap <Up />. More data or more regularization: training <Up />, gap <Down />.</p> },
              { title: "4. Push it to the extremes", body: <p>η → 0: nothing moves. η huge: divergence. λ → ∞: all weights 0, a constant model. p = 1 dropout: nothing gets through. B = 1: very noisy steps. The truth for moderate values usually lies between the two extremes.</p> },
              { title: "5. Try a tiny example", body: <p>One parameter, two data points, or <M>{t`J(\theta) = \theta^2`}</M>. Two lines of arithmetic often settle the question.</p> },
              { title: "6. Eliminate impossible options", body: <p>Probabilities sum to 1, losses are ≥ 0, a gradient has the shape of its parameter, a softmax's argmax is the largest logit's position, nothing changes in eval mode for dropout.</p> },
            ]}
          />
          <H>What each common change does</H>
          <div className="table-wrap">
            <Table
              head={["Increase…", "Training loss", "Gap (validation − training)", "Speed / stability"]}
              rows={[
                ["learning rate η", <>falls faster, up to a point; then oscillates or diverges</>, "—", "faster, then unstable"],
                ["batch size B (same η)", "—", "—", "fewer, less noisy updates per epoch"],
                ["number of epochs", <><Down /> keeps falling</>, <><Up /> eventually (overfitting)</>, "—"],
                ["capacity (units, layers, degree)", <Down />, <Up />, "more computation per step"],
                ["L2 penalty λ", <Up />, <><Down /> (until it underfits)</>, "—"],
                ["dropout rate p", <Up />, <><Down /> (until it underfits)</>, "noisier, slower"],
                ["training data N", <><Up /> slightly</>, <Down />, "slower epochs"],
                ["momentum ρ (below 1)", <>falls faster in narrow valleys</>, "—", "close to 1: overshoots"],
                ["feature standardization", <>falls faster</>, "—", "rounder valley, larger η possible"],
                ["data augmentation", <><Up /> slightly</>, <Down />, "—"],
              ]}
            />
          </div>
          <H>Worked examples</H>
          <Steps
            title="'GD with η = 0.4 converges on J(θ) = 2θ². What if η = 0.8?'"
            steps={[
              { title: "Write one step", body: <MB>{t`\theta \leftarrow \theta - \eta J'(\theta) = \theta - \eta\cdot4\theta = (1 - 4\eta)\,\theta`}</MB> },
              { title: "η = 0.4", body: <p>The factor is 1 − 1.6 = −0.6: θ flips sign but shrinks by 40% per step. It converges, oscillating.</p> },
              { title: "η = 0.8", body: <p>The factor is 1 − 3.2 = −2.2: θ flips sign and grows. It <strong>diverges</strong>.</p> },
              { title: "The general rule", body: <p>GD converges exactly when <M>{t`|1 - \eta h| < 1`}</M>, i.e. <M>{t`0 < \eta < 2/h`}</M>, where h is the curvature <M>{t`J''`}</M> (here 4, so η &lt; 0.5).</p> },
            ]}
          />
          <Steps
            title="'Training accuracy 99%, validation 70%. Which change helps most?'"
            steps={[
              { title: "Read the numbers", body: <p>The model fits the training data almost perfectly, and the gap is 29 points: <strong>overfitting</strong>.</p> },
              { title: "What reduces overfitting?", body: <p>Less effective capacity (smaller model, more λ, more dropout, early stopping) or more information (more data, augmentation).</p> },
              { title: "Eliminate", body: <p>"More hidden units", "smaller λ", "more epochs", "no dropout" all add capacity: they make the gap worse. "More data / augmentation" is the one that helps.</p> },
            ]}
          />
          <Steps
            title="'A 10-class model with near-zero weights prints a loss of 230.26 instead of about 2.30. Why?'"
            steps={[
              { title: "Baseline", body: <p>Near-zero weights → logits ≈ 0 → uniform probabilities 1/10 → cross-entropy <M>{t`\ln 10 \approx 2.30`}</M> per example.</p> },
              { title: "Compare", body: <p>230.26 = 100 × 2.3026, and the batch has 100 examples.</p> },
              { title: "Mechanism", body: <p>The code sums the loss over the examples instead of averaging it: the 1/N is missing.</p> },
              { title: "Consequence", body: <p>Gradients 100× larger, as if the learning rate were 100× larger.</p> },
            ]}
          />
          <H>Predict first</H>
          <Think q="You multiply all the weights and the bias of a trained logistic regression by 10. What happens to the decision boundary and to the probabilities?">
            The boundary <M>{t`\theta^\top\tilde{\mathbf x} = 0`}</M> is unchanged (10 · 0 = 0), so every prediction stays the same.
            But every logit is 10× further from 0, so the probabilities move towards 0 and 1: the model becomes far more confident.
            Its loss drops on correctly classified points and rises a lot on misclassified ones.
          </Think>
          <Think q="Why can't the training loss of a correctly implemented model go up when you add capacity (say, add a hidden unit) and train to the optimum?">
            The bigger model contains the smaller one (set the new unit's outgoing weight to 0), so its best training loss is at
            most the smaller model's. In practice, training might not find that optimum. Validation loss, however, can go
            either way.
          </Think>
        </>
      ),
    },
    // ------------------------------------------------------------ Optimization
    {
      id: "optimization",
      title: "What happens if… (learning rate, batch size, momentum, scaling)",
      summary: "How each optimizer knob changes the step, and therefore speed and stability, with the 1 − ηh rule and momentum's 1/(1 − ρ).",
      body: () => (
        <>
          <H>Everything acts on one line</H>
          <MB>{t`\theta_{t+1} = \theta_t - \eta\,\mathbf g_t`}</MB>
          <p>
            Every optimization knob changes either the <strong>length</strong> of the step (η, the scale of the loss, clipping),
            its <strong>noise</strong> (batch size, shuffling), or its <strong>direction</strong> (momentum, Adam, feature scaling).
            Ask which one, and the effect follows.
          </p>
          <H>The 1 − ηh rule</H>
          <p>
            Near a minimum, any smooth loss looks like a bowl <M>{t`J(\theta) \approx \frac h2(\theta - \theta^\star)^2`}</M> with curvature h.
            One GD step multiplies the distance to the minimum by <M>{t`(1 - \eta h)`}</M>:
          </p>
          <Table
            head={["ηh", "Factor 1 − ηh", "Behaviour"]}
            rows={[
              ["0 < ηh < 1", "between 0 and 1", "smooth, monotone convergence (slow if ηh is tiny)"],
              ["ηh = 1", "0", "jumps straight to the minimum in one step"],
              ["1 < ηh < 2", "between −1 and 0", "converges, but overshoots: θ flips side every step"],
              ["ηh > 2", "below −1", "diverges: each step lands further away on the other side"],
            ]}
          />
          <GD1DWidget />
          <Callout kind="intuition">
            With several parameters, every direction has its own curvature. η must be below 2/h for the <em>steepest</em>
            direction, which makes progress in the flattest direction slow. That's the narrow-valley problem, and it's why
            feature scaling, momentum and Adam help.
          </Callout>
          <ValleyWidget />
          <H>Momentum's 1/(1 − ρ)</H>
          <Steps
            title="Why momentum ρ = 0.9 can be up to 10× faster along a valley"
            steps={[
              { title: "Velocity update", body: <MB>{t`\mathbf v_{t+1} = \rho\,\mathbf v_t - \eta\,\mathbf g_t`}</MB> },
              { title: "If the gradient stays the same, g", body: <MB>{t`\mathbf v \to -\eta\mathbf g\,(1 + \rho + \rho^2 + \dots) = -\frac{\eta\mathbf g}{1-\rho}`}</MB> },
              { title: "With ρ = 0.9", body: <p>The step becomes <M>{t`10\,\eta\mathbf g`}</M>: consistent directions (along the valley) speed up 10×.</p> },
              { title: "If the gradient flips sign every step (across the valley)", body: <p>Consecutive contributions have opposite signs and largely cancel: the zig-zag is damped.</p> },
            ]}
          />
          <H>Batch size, shuffling, and the scale of the loss</H>
          <Table
            head={["Change", "Mechanism", "Visible effect"]}
            rows={[
              ["B = 1 (pure SGD)", "each step uses one example's gradient: very noisy", "cheap steps, a jumpy loss curve, never settles exactly"],
              ["B = N (full batch)", "the exact gradient, one step per epoch", "smooth curve, but few updates per epoch"],
              ["no shuffling, data sorted by class", "consecutive minibatches pull in opposite directions", "oscillating loss, poor model"],
              ["loss summed instead of averaged", "gradient × B", "same as η × B: likely divergence"],
              ["loss × 2 with Adam", "Adam divides by the gradient's own scale", "almost no change (with SGD: like η × 2)"],
              ["gradient clipping at C", <>if <M>{t`\|\mathbf g\| > C`}</M>: <M>{t`\mathbf g \leftarrow C\,\mathbf g/\|\mathbf g\|`}</M></>, "same direction, length capped: no exploding steps"],
              ["standardized features", "every direction gets a similar curvature", "rounder bowl: faster, a larger η is safe"],
            ]}
          />
          <H>Predict first</H>
          <Think q="GD on J(θ) = 5θ² with η = 0.1. What happens, and in how many steps?">
            h = 10, so ηh = 1 and the factor 1 − ηh = 0: the first step lands exactly on θ = 0, from any start.
          </Think>
          <Think q="Adam and plain SGD both train well with the loss averaged over the batch. You switch to the summed loss and leave η unchanged. Which optimizer still trains fine?">
            Adam. Its step depends on the ratio of the averaged gradient to the square root of the averaged squared gradient,
            so scaling the gradient by B cancels out. SGD's steps become B times longer.
          </Think>
          <Think q="Why does a learning rate that works for 2 input features suddenly diverge after you add a third feature measured in thousands?">
            The new feature adds a very steep direction (curvature grows with the square of the feature's values). The old
            η is now above 2/h for that direction. Standardizing the new feature fixes it.
          </Think>
        </>
      ),
    },
    // ------------------------------------------------------------ Generalization
    {
      id: "generalization",
      title: "What happens if… (data, capacity, regularization, evaluation)",
      summary: "Reading training vs validation loss, and how capacity, λ, dropout, data, augmentation and evaluation choices move them.",
      body: () => (
        <>
          <H>Two numbers tell you almost everything</H>
          <Table
            head={["Training loss", "Validation loss", "Diagnosis", "What helps"]}
            rows={[
              ["high", "high (similar)", "underfitting (high bias)", "more capacity, more features, less regularization, train longer"],
              ["low", "much higher", "overfitting (high variance)", "more data, augmentation, more regularization (λ, dropout), early stopping, smaller model"],
              ["low", "low", "good fit", "stop here; check the test set once"],
              ["validation far better than training", "", "usually dropout/augmentation active only during training, or a validation set that's easier", "check the evaluation setup"],
            ]}
          />
          <CapacityWidget withLambda />
          <H>Pushing the knobs to their extremes</H>
          <Table
            head={["Knob", "At 0 (or tiny)", "Very large"]}
            rows={[
              ["L2 penalty λ", "no regularization: may overfit", "all weights ≈ 0: predicts the mean label everywhere (underfits)"],
              ["dropout p", "no dropout", "almost nothing gets through: can't fit even the training data"],
              ["polynomial degree / hidden units", "a constant or a line: underfits", "fits the noise: overfits (unless N is large)"],
              ["training set size N", "can't learn anything general", "the gap closes; only the model's capacity limits the error"],
              ["epochs (no early stopping)", "the starting model", "training loss → its minimum; validation loss rises after a point"],
            ]}
          />
          <Steps
            title="Worked example: D = 500 features, N = 100 examples, linear regression"
            steps={[
              { title: "Count", body: <p>θ has 501 entries, but there are only 100 equations (one per example).</p> },
              { title: "Consequence for the closed form", body: <p><M>{t`\tilde{\mathbf X}^\top\tilde{\mathbf X}`}</M> is 501 × 501 but has rank at most 100: not invertible. Infinitely many θ fit the training data perfectly (training loss 0).</p> },
              { title: "Consequence for generalization", body: <p>A perfect fit to 100 noisy points in 500 dimensions: severe overfitting.</p> },
              { title: "Fix", body: <><p>L2 regularization adds <M>{t`\frac{N\lambda}{2}\mathbf I`}</M>, which makes the matrix invertible and picks the smallest-weight solution:</p><MB>{t`\theta^\star = \Big(\tilde{\mathbf X}^\top\tilde{\mathbf X} + \tfrac{N\lambda}{2}\mathbf I\Big)^{-1}\tilde{\mathbf X}^\top\mathbf y`}</MB></> },
            ]}
          />
          <H>Evaluation mistakes that change the result</H>
          <ul>
            <li><strong>Tuning on the test set</strong>: every choice made by looking at test results makes the reported test score too optimistic. Tune on validation.</li>
            <li><strong>Forgetting <code>model.eval()</code></strong>: dropout keeps dropping and batch norm uses minibatch statistics, so test results are worse and random.</li>
            <li><strong>Preprocessing fitted on all data</strong>: the scaler's mean and standard deviation then contain test information (leakage).</li>
            <li><strong>Accuracy on imbalanced data</strong>: 95% accuracy can mean "always predicts the majority class".</li>
            <li><strong>Label-changing augmentation</strong>: rotating a 6 by 180° makes a 9.</li>
          </ul>
          <H>Predict first</H>
          <Think q="You double the training set. Will the training loss go up or down?">
            Usually <strong>up</strong> slightly: with more examples, the same model can't fit each one as closely. The validation
            loss goes down, so the gap shrinks.
          </Think>
          <Think q="Early stopping picks the epoch with the lowest validation loss. Why is that a form of regularization?">
            Training starts from small weights and moves towards the training-set optimum. Stopping early keeps the model closer to
            its simple starting point, which limits its effective capacity, like a penalty on large weights.
          </Think>
          <Think q="A tiny data set (500 images): train a big CNN from scratch, or fine-tune a pretrained one?">
            Fine-tune (or train only a new head on top of) the pretrained network. Its early layers already detect generic
            features, so far fewer examples are needed. Training from scratch would overfit.
          </Think>
        </>
      ),
    },
    // ------------------------------------------------------------ PyTorch results
    {
      id: "pytorch",
      title: "Predict the PyTorch result",
      summary: "The defaults that decide what PyTorch code prints: loss reductions, train/eval behaviour, gradient accumulation, detach, dims and shapes.",
      body: () => (
        <>
          <H>How to predict a PyTorch snippet's output</H>
          <ol>
            <li><strong>Write the shape next to every line.</strong> Most questions are decided by a shape (broadcasting, flatten, dim).</li>
            <li><strong>Know the default of each module</strong>: the table below.</li>
            <li><strong>Gradients add up</strong> until something resets them.</li>
            <li><strong>Compute with small numbers by hand</strong>; use ln 2, ln 10, σ(0) = 0.5, σ′(0) = 0.25.</li>
          </ol>
          <H>The defaults that decide the answer</H>
          <div className="table-wrap">
            <Table
              head={["Thing", "Default behaviour", "Classic trap"]}
              rows={[
                [<code>CrossEntropyLoss()</code>, "takes logits (N, K) and int64 labels (N,); applies softmax itself; averages over the batch", "a Softmax layer before it (applied twice); reduction='sum' gives N× the loss"],
                [<code>BCEWithLogitsLoss()</code>, "takes logits, applies the sigmoid itself; float labels; averages", "a Sigmoid layer before it"],
                [<code>MSELoss()</code>, "averages the squared differences, broadcasting the two shapes", "(N, 1) vs (N,) broadcasts to (N, N): wrong value, only a warning"],
                [<code>Dropout(p)</code>, "training mode: zero with probability p, divide the rest by 1 − p; eval mode: identity", "forgetting model.eval() at test time"],
                [<code>BatchNorm1d</code>, "training mode: minibatch mean/variance (and updates running averages); eval mode: running averages", "tiny batches (B = 2 → ±1)"],
                [<code>loss.backward()</code>, <>adds into every <code>.grad</code></>, <>no <code>zero_grad()</code>: stale gradients accumulate</>],
                [<code>torch.no_grad()</code>, "no graph: results have requires_grad=False", "calling backward on something computed inside it"],
                [<code>t.detach()</code>, "same values, treated as a constant by autograd", "gradients that skip a path"],
                [<code>argmax(t, dim)</code>, "dim=1: per row (per example); dim=0: per column", "dim=0 on (N, K) scores"],
                [<code>nn.Linear(in, out)</code>, <><code>x @ W.T + b</code>, W of shape (out, in)</>, "counting parameters as in·out (forgetting + out)"],
                [<code>SGD(..., momentum=ρ)</code>, "buf ← ρ·buf + g; p ← p − lr·buf", "the second step is (1 + ρ) times the first for a constant gradient"],
                [<><code>/</code> on int tensors</>, "true division: floats", <>integer division is <code>{"//"}</code></>],
              ]}
            />
          </div>
          <H>A snippet, line by line</H>
          <CodeWalk
            title="What happens when zero_grad() is missing"
            code={py`
w = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
for _ in range(2):
    loss = (w - 1) ** 2
    loss.backward()           # no zero_grad: gradients add up
    opt.step()
print(round(w.item(), 4))
`}
            steps={[
              { line: [1, 2], say: "One parameter starting at 0, plain SGD with learning rate 0.1. The loss (w − 1)² is smallest at w = 1.", vars: [["w", "0.0"], ["w.grad", "None"]] },
              { line: [4, 5], say: <>First pass: loss = 1, gradient <M>{t`2(w-1) = -2`}</M> is stored in <code>w.grad</code>.</>, vars: [["w.grad", "-2.0"]] },
              { line: 6, say: <>Step: <M>{t`w \leftarrow 0 - 0.1\cdot(-2) = 0.2`}</M>.</>, vars: [["w", "0.2"]] },
              { line: [4, 5], say: <>Second pass: the new gradient is <M>{t`2(0.2-1) = -1.6`}</M>, but backward <strong>adds</strong> it to the old −2.</>, vars: [["w.grad", "-3.6 (should be -1.6)"]] },
              { line: 6, say: <>Step: <M>{t`w \leftarrow 0.2 - 0.1\cdot(-3.6) = 0.56`}</M>. With <code>opt.zero_grad()</code> at the top of the loop it would be 0.36.</>, vars: [["printed", "0.56"]] },
            ]}
          />
          <H>Predict first</H>
          <Think q="A model ends with nn.Softmax(dim=1) and is trained with nn.CrossEntropyLoss(). It trains, but slowly and to a worse accuracy. Why?">
            CrossEntropyLoss applies its own softmax, so the model's probabilities (all between 0 and 1) are treated as logits.
            Logits squeezed into [0, 1] can only produce nearly uniform probabilities, and the gradients through the extra softmax
            are small. Remove the Softmax layer.
          </Think>
          <Think q="x = torch.tensor(2.0, requires_grad=True); y = x**2; z = y.detach() * x. What is dz/dx after z.backward()?">
            4: autograd treats y as the constant 4, so z = 4x. Without detach it would be 3x² = 12.
          </Think>
          <Think q="What does nn.Conv2d(3, 16, 5) do to a (1, 3, 32, 32) input, and how many parameters does it have?">
            Output (1, 16, 28, 28), since 32 − 5 + 1 = 28. Parameters: 5·5·3·16 + 16 = 1,216.
          </Think>
        </>
      ),
    },
    // ------------------------------------------------------------ Diagnose
    {
      id: "diagnose",
      title: "Why is the result what it is, and how do I fix it?",
      summary: "Compare with baseline values, compare training with validation, then match the symptom to its mechanism and fix.",
      body: () => (
        <>
          <H>The detective method</H>
          <Steps
            steps={[
              { title: "Compare with a baseline value", body: <p>What would a model that knows nothing score? (Table below.) A loss exactly at the baseline means "nothing learned"; far above it means something is scaled wrong.</p> },
              { title: "Compare training with validation", body: <p>Both bad: underfitting or a training bug. Training good, validation bad: overfitting, or (if validation is at chance) a validation-pipeline bug.</p> },
              { title: "Check that the parameters change", body: <p>A frozen loss means no updates: missing <code>step()</code>, η = 0, the wrong parameters handed to the optimizer, or dead units.</p> },
              { title: "Look at the scale", body: <p>nan or huge values: steps too large, log(0), exploding gradients. Values 10× or 100× too large: a sum instead of a mean, or a missing normalization.</p> },
              { title: "Name the mechanism, then the fix that targets it", body: <p>"Validation is worse because of overfitting, so add regularization" beats "try a different optimizer".</p> },
            ]}
          />
          <H>Baseline values to know</H>
          <Table
            head={["Quantity", "Model that knows nothing", "Value"]}
            rows={[
              ["cross-entropy, K classes", "uniform probabilities 1/K", <><M>{t`\ln K`}</M>: 0.69 (K = 2), 1.10 (K = 3), 2.30 (K = 10)</>],
              ["accuracy, balanced K classes", "random guess", "1/K"],
              ["accuracy, imbalanced classes", "always the majority class", "the majority's share (e.g. 0.95)"],
              ["MSE", "always predicts the mean label", "the variance of the labels"],
            ]}
          />
          <H>Symptom → mechanism → fix</H>
          <div className="table-wrap">
            <Table
              head={["Symptom", "Likely mechanism", "Fix"]}
              rows={[
                ["loss stuck at exactly ln K", "parameters not updating", "check step(), η, optimizer parameters, gradients"],
                ["loss far above ln K at the start", "huge initial logits, or a summed loss", "smaller init, mean reduction"],
                ["loss becomes nan", "too large steps, log(0)", "smaller η, clipping, losses on logits"],
                ["loss decreases very slowly with zig-zags", "badly conditioned problem", "standardize, momentum, Adam (not a larger η)"],
                ["training ≪ validation loss, gap growing", "overfitting", "more data/augmentation, λ, dropout, early stopping"],
                ["both losses high and close", "underfitting", "more capacity, train longer, less regularization"],
                ["validation accuracy at chance from the start", "validation pipeline bug (labels, preprocessing)", "check the validation data path"],
                ["validation loss below training loss", "dropout/augmentation only during training", "usually fine; compare in eval mode"],
                ["spike at the same point every epoch", "tiny last minibatch with batch norm", "drop_last=True for training"],
                ["always predicts the majority class", "class imbalance", "class weights, resampling, per-class metrics"],
                ["early layers don't learn (deep sigmoid net)", "vanishing gradients", "ReLU, skip connections, normalization"],
                ["weights grow forever (logistic regression)", "separable data: no minimum", "L2 or early stopping"],
              ]}
            />
          </div>
          <Steps
            title="Worked example: training accuracy 95%, validation accuracy 10% (10 classes) from the first epoch"
            steps={[
              { title: "Baseline", body: <p>10% is exactly chance for 10 classes: on validation data the model is no better than guessing.</p> },
              { title: "Is it overfitting?", body: <p>No. Overfitting shows validation accuracy rising first and then falling, not stuck at chance while the training accuracy climbs.</p> },
              { title: "So the validation outputs have nothing to do with the validation labels", body: <p>Candidates: inputs and labels shuffled separately, a different preprocessing (e.g. pixels not divided by 255), or the wrong label file.</p> },
              { title: "Fix and check", body: <p>Evaluate the trained model on a few <em>training</em> examples through the validation code path. If that also gives chance accuracy, the bug is in that path.</p> },
            ]}
          />
          <H>Predict first</H>
          <Think q="A binary classifier's loss starts at 0.69 and stays there. Its accuracy is 50%. What do you check first?">
            0.69 = ln 2 = the uniform-guess baseline, so nothing is being learned. Check that gradients are non-zero and the
            parameters change after a step (step() called, η &gt; 0, the right parameters given to the optimizer).
          </Think>
          <Think q="Your loss is 10× larger than your classmate's on the same data and model, but your accuracy is the same. What's the likely difference?">
            A different reduction: you sum over 10 things (e.g. a batch of 10) where they average. The scale of the loss differs,
            not the model. With plain SGD this also means 10× larger steps.
          </Think>
        </>
      ),
    },
    // ------------------------------------------------------------ Short math
    {
      id: "short-math",
      title: "Short math problems without a calculator",
      summary: "The handful of values worth memorizing, the tricks that make exam arithmetic short, and worked problems.",
      body: () => (
        <>
          <H>Values worth memorizing</H>
          <Table
            head={["Quantity", "Value", "Where it shows up"]}
            rows={[
              [<M>{t`\ln 2`}</M>, "0.693", "BCE of a 50/50 guess, CE with p = 0.5"],
              [<M>{t`\ln 3`}</M>, "1.099", "CE of a uniform 3-class guess"],
              [<M>{t`\ln 10`}</M>, "2.303", "CE of a uniform 10-class guess; −ln 0.1"],
              [<M>e</M>, "2.718", "softmax with logits differing by 1"],
              [<M>{t`e^{-1},\ e^{-2}`}</M>, "0.368, 0.135", "softmax, sigmoid"],
              [<M>{t`\sigma(0),\ \sigma'(0)`}</M>, "0.5, 0.25", "zero logits; the largest sigmoid slope"],
              [<M>{t`\sigma(2),\ \sigma(-2)`}</M>, "0.881, 0.119", "a fairly confident logistic prediction"],
              [<M>{t`1/(1-\rho)`}</M>, "10 for ρ = 0.9", "momentum's eventual speed-up"],
            ]}
          />
          <H>Tricks that make the arithmetic short</H>
          <ul>
            <li><strong>Log rules</strong>: <M>{t`\ln(ab) = \ln a + \ln b`}</M>, <M>{t`\ln(1/a) = -\ln a`}</M>. So <M>{t`-\ln 0.25 = \ln 4 = 2\ln 2`}</M>, <M>{t`\ln 1.5 = \ln 3 - \ln 2`}</M>.</li>
            <li><strong>Softmax ratios</strong>: only differences of logits matter. Logits differing by <M>{t`\ln k`}</M> give probabilities in the ratio k : 1.</li>
            <li><strong>GD on a quadratic</strong>: one step multiplies the distance to the minimum by <M>{t`1 - \eta h`}</M>.</li>
            <li><strong>Error × input</strong>: for linear/logistic/softmax regression the gradient of an example is <M>{t`(\hat y - y)\tilde{\mathbf x}`}</M> (×2 for MSE). Often only the sign is asked.</li>
            <li><strong>Parameter counts</strong>: dense <M>{t`(\text{in}+1)\cdot\text{out}`}</M>; convolution <M>{t`F\cdot F\cdot C_{\text{in}}\cdot C_{\text{out}} + C_{\text{out}}`}</M>.</li>
            <li><strong>Output size</strong>: <M>{t`\lfloor(D + 2P - F)/s\rfloor + 1`}</M>; pooling by 2 halves (rounding down).</li>
          </ul>
          <H>Worked problems</H>
          <Steps
            title="Softmax and cross-entropy of the logits (ln 4, 0, 0) with true class 1"
            steps={[
              { title: "Exponentiate (no calculator needed)", body: <MB>{t`(e^{\ln 4},\ e^0,\ e^0) = (4,\ 1,\ 1),\quad\text{sum} = 6`}</MB> },
              { title: "Probabilities", body: <MB>{t`\hat{\mathbf y} = (\tfrac46,\ \tfrac16,\ \tfrac16) = (0.667,\ 0.167,\ 0.167)`}</MB> },
              { title: "Cross-entropy with a log rule", body: <MB>{t`-\ln\tfrac23 = \ln\tfrac32 = \ln 3 - \ln 2 \approx 1.099 - 0.693 = 0.405`}</MB> },
            ]}
          />
          <Steps
            title="Two GD steps on J(θ) = (θ − 3)², θ₀ = 0, η = 0.25"
            steps={[
              { title: "Curvature", body: <p><M>{t`J'' = h = 2`}</M>, so each step multiplies the distance to 3 by <M>{t`1 - 0.25\cdot2 = 0.5`}</M>.</p> },
              { title: "Distances", body: <p>3 → 1.5 → 0.75.</p> },
              { title: "Positions", body: <p><M>{t`\theta_1 = 3 - 1.5 = 1.5`}</M>, <M>{t`\theta_2 = 3 - 0.75 = 2.25`}</M>. (Check step 1 directly: <M>{t`0 - 0.25\cdot2(0-3) = 1.5`}</M> ✓.)</p> },
            ]}
          />
          <Steps
            title="Parameters of Conv2d(3, 8, 3) → ReLU → MaxPool2d(2) → Flatten → Linear(?, 10) on 32 × 32 RGB images"
            steps={[
              { title: "Convolution", body: <p><M>{t`3\cdot3\cdot3\cdot8 + 8 = 224`}</M> parameters; output 32 − 3 + 1 = 30, so (N, 8, 30, 30).</p> },
              { title: "Pooling and flatten", body: <p>(N, 8, 15, 15) → <M>{t`8\cdot15\cdot15 = 1800`}</M> features.</p> },
              { title: "Linear head", body: <p><M>{t`1800\cdot10 + 10 = 18\,010`}</M> parameters.</p> },
              { title: "Total", body: <p><M>{t`224 + 18\,010 = 18\,234`}</M>.</p> },
            ]}
          />
          <PyCell
            title="Check the three problems"
            code={py`
import numpy as np
p = np.exp([np.log(4), 0, 0]); p /= p.sum()
print("softmax", p.round(3), " CE", round(-np.log(p[0]), 3))

theta = 0.0
for _ in range(2):
    theta -= 0.25 * 2 * (theta - 3)
print("theta_2", theta)

conv = 3 * 3 * 3 * 8 + 8
side = (32 - 3 + 1) // 2
head = 8 * side * side * 10 + 10
print("conv", conv, " head", head, " total", conv + head)
`}
          />
        </>
      ),
    },
    // ------------------------------------------------------------ Beyond the slides
    {
      id: "beyond",
      title: "Beyond the slides: common ML exam topics",
      summary: "Bias–variance, precision/recall/F1, cross-validation, L1 vs L2, initialization, learning-rate schedules, leakage, receptive fields, temperature, class weights.",
      body: () => (
        <>
          <Callout kind="note">
            These topics aren't in COMP 432's slides, but they're standard in machine learning courses and build directly on what
            the course does teach. Questions about them are tagged <span className="qextra">Beyond the slides</span>. Study the
            course material first.
          </Callout>
          <H>Bias and variance</H>
          <p>
            The error of a model on new data comes from two sources besides noise. <strong>Bias</strong>: the model is too
            simple to represent the true pattern (underfitting); it's wrong the same way whatever the training sample.
            <strong> Variance</strong>: the model is so flexible that it changes a lot with the particular training sample
            (overfitting). More capacity lowers bias and raises variance; more data lowers variance.
          </p>
          <H>Precision, recall, F1</H>
          <Table
            head={["", "Predicted positive", "Predicted negative"]}
            rows={[
              ["Actually positive", "TP (true positive)", "FN (false negative)"],
              ["Actually negative", "FP (false positive)", "TN (true negative)"],
            ]}
          />
          <MB>{t`\text{precision} = \frac{TP}{TP + FP},\qquad \text{recall} = \frac{TP}{TP + FN},\qquad F_1 = \frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}}`}</MB>
          <p>
            Precision: of what you flagged, how much was right. Recall: of what was there, how much you found. With rare positives
            they reveal what accuracy hides.
          </p>
          <H>Cross-validation</H>
          <p>
            k-fold cross-validation splits the training data into k parts, trains k models (each on k − 1 parts) and validates
            each on the part it didn't see. The average validation score is a more reliable basis for choosing hyperparameters
            than a single split, at k times the cost.
          </p>
          <H>L1 versus L2 regularization</H>
          <p>
            L2 adds <M>{t`\frac\lambda2\|\theta\|_2^2`}</M>; its gradient <M>{t`\lambda\theta`}</M> fades as weights shrink, so weights get
            small but rarely exactly 0. L1 adds <M>{t`\lambda\|\theta\|_1 = \lambda\sum|\theta_p|`}</M>; its gradient
            <M>{t`\ \lambda\,\mathrm{sign}(\theta)`}</M> has the same size however small θ is, so it drives many weights to exactly 0
            (a sparse model that selects features).
          </p>
          <H>Initialization</H>
          <p>
            Weights all equal (e.g. all 0) make every unit of a layer compute the same thing and receive the same gradient
            forever: random initialization breaks this symmetry. The scale matters too: a unit adds up fan_in inputs, so weights
            with variance about 1/fan_in (Xavier/Glorot, for tanh) or 2/fan_in (He, for ReLU) keep activations and gradients
            at a similar size from layer to layer.
          </p>
          <H>Learning-rate schedules</H>
          <p>
            Large steps early make fast progress; with constant η, SGD's noise keeps it bouncing around the minimum. Lowering η
            over time (step decay, cosine decay) lets it settle. "Warm-up" does the opposite at the very start, ramping η up from a
            small value so the first steps, taken with unreliable statistics, don't blow up.
          </p>
          <H>Data leakage</H>
          <p>
            Anything computed from the test (or validation) data that influences training, such as the scaler's statistics,
            duplicates across splits or features derived from the label, makes test scores optimistic. Fit every preprocessing
            step on the training set only.
          </p>
          <H>Receptive field and 1×1 convolutions</H>
          <p>
            Each 3×3 convolution (stride 1) extends what a unit sees by 2 pixels: two layers see 5×5, three see 7×7, with fewer
            parameters than one large kernel. A 1×1 convolution mixes channels at each pixel (C_in·C_out + C_out parameters) and
            is used to reduce the number of channels cheaply.
          </p>
          <H>Softmax temperature and class weights</H>
          <p>
            Dividing logits by a temperature T &gt; 1 flattens the softmax (T &lt; 1 sharpens it) without changing the argmax.
            For imbalanced classes, a weighted cross-entropy multiplies each example's loss by a weight that's larger for rare
            classes, so mistakes on them cost more.
          </p>
          <H>Predict first</H>
          <Think q="Precision 1.0, recall 0.1. What kind of classifier is this?">
            Very cautious: it flags only a few cases, all of them correct, but misses 90% of the actual positives. F₁ is low
            (2·1·0.1/1.1 ≈ 0.18), because it's dominated by the smaller number.
          </Think>
          <Think q="Why does L1 give exactly-zero weights but L2 doesn't?">
            Near θ = 0, L2's pull λθ becomes tiny, so the data gradient easily balances it at some small non-zero value. L1's pull
            stays λ in size, so unless the data pushes back by at least λ, the weight is driven all the way to 0.
          </Think>
        </>
      ),
    },
  ],
};
