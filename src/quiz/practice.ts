// Deep-understanding practice: "what happens if", predict the PyTorch result, diagnose a result,
// short math, and questions from beyond the slides. Every multiple-choice question has five options,
// so the mock exam's "Deep understanding" mode can draw from them.
import { MCQuestion, NumQuestion, Question, QuizKind } from "./types";

const r = String.raw;

type Five = [string, string, string, string, string];

function mc(kind: QuizKind, prompt: string, options: Five, answer: number, explain: string, code?: string): MCQuestion {
  return { type: "mc", kind, prompt, options, answer, explain, code };
}
function num(kind: QuizKind, prompt: string, answer: number, explain: string, code?: string): NumQuestion {
  return { type: "num", kind, prompt, answer, explain, code };
}
/** Marks a question as going beyond what the slides cover. */
const extra = <Q extends Question>(q: Q): Q => ({ ...q, extra: true });

export const practiceQuiz: Record<string, Question[]> = {
  // ------------------------------------------------------------ How to reason
  "practice/how-to-reason": [
    mc("reason", r`Gradient descent on $J(\theta) = 2\theta^2$ converges with $\eta = 0.4$. What happens if you double the learning rate to $\eta = 0.8$?`,
      ["It converges twice as fast", "It converges, but oscillates around the minimum", "It diverges: θ grows in size every step", "It stops at θ = 0 after one step", "Nothing changes: the minimum is the same"], 2,
      r`Each step multiplies θ by $1 - \eta J''(\theta) = 1 - 4\eta$. With η = 0.4 that's −0.6 (shrinks, flipping sign); with η = 0.8 it's −2.2, so $|\theta|$ grows by 2.2× per step. GD on this bowl converges only for $0 < \eta < 2/4 = 0.5$.`),
    mc("reason", "To reason about a knob (learning rate, λ, dropout p, batch size…), the most reliable first move is to…",
      ["memorize the default value", "push it to its extremes (0 and very large) and see what the model does there", "always assume a larger value is better", "look only at the training loss", "change two knobs at once"], 1,
      "At the extremes the behaviour is usually obvious (λ → ∞: all weights 0, a constant model; λ = 0: no regularization). In between, the effect normally changes gradually from one extreme to the other."),
    mc("reason", "A model reaches 99% training accuracy and 70% validation accuracy. Which change is most likely to raise the validation accuracy?",
      ["More hidden units", "A smaller L2 penalty λ", "Training for more epochs", "More training data or data augmentation", "Removing dropout"], 3,
      "The large gap means overfitting (high variance). More (or augmented) data makes memorizing harder. The other four all increase the model's effective capacity, which widens the gap."),
    mc("reason", "A model has 60% training accuracy and 59% validation accuracy on a task where people reach 98%. What is the best next step?",
      ["Add more regularization", "Collect more data with the same model", "Increase the model's capacity (more units/layers or features) or train longer", "Use a smaller learning rate and stop earlier", "Use dropout with p = 0.9"], 2,
      "Training and validation agree, and both are poor: underfitting (high bias). More data or regularization won't help a model that can't even fit the training set."),
    mc("reason", "Which statement must be true of any correct answer to 'what is the softmax output for these logits?'",
      ["Every entry is above 0.5", "The entries are positive and sum to 1", "The largest entry is exactly 1", "Entries can be negative for negative logits", "The entries are the logits divided by their sum"], 1,
      "Use invariants to eliminate options fast: softmax outputs are always in (0, 1) and add up to 1. Losses are ≥ 0, shapes must match, a gradient has the shape of its parameter."),
    mc("reason", r`For the MSE of linear regression, what changes when you multiply every label $y_n$ by 10?`,
      ["Nothing: the optimal parameters are the same", "The optimal weights and bias are all multiplied by 10", "Only the bias changes", "The weights are divided by 10", "The loss becomes non-convex"], 1,
      r`The closed form $\theta^\star = (\tilde{\mathbf X}^\top\tilde{\mathbf X})^{-1}\tilde{\mathbf X}^\top\mathbf y$ is linear in y: scaling y by 10 scales θ* by 10. (The minimal loss is multiplied by 100.)`),
  ],
  // ------------------------------------------------------------ Optimization what-ifs
  "practice/optimization": [
    mc("reason", "The training loss goes up and down wildly from step to step and ends higher than it started. The most likely cause is…",
      ["the learning rate is too small", "the learning rate is too large", "too few epochs", "too much L2 regularization", "the model has too few parameters"], 1,
      "Steps that overshoot the valley land higher up the other side; if each overshoot is larger than the last, the loss climbs. A learning rate that is too small gives a slow but steady decrease."),
    mc("reason", "On a convex loss, what does a learning rate that is far too small (but positive) do?",
      ["The loss diverges", "GD converges to a worse local minimum", "The loss decreases steadily but very slowly", "The loss oscillates", "The gradient becomes zero immediately"], 2,
      "Each step goes downhill, just by a tiny amount. A convex loss has no worse local minimum to get stuck in."),
    mc("reason", "You keep the learning rate fixed and change the minibatch size from 32 to the full training set (N = 3,200). What happens per epoch?",
      ["100× more parameter updates, each noisier", "1 update instead of 100, each with an exact (noise-free) gradient", "The same number of updates, but larger", "The loss can no longer decrease", "Each update uses a random subset of the data"], 1,
      "One epoch = N / B updates: 3200 / 32 = 100 with minibatches, 1 with the full batch. The full-batch gradient is exact; minibatch gradients are noisy estimates of it."),
    mc("reason", "Training data are sorted by class (all class 0 first, then class 1), and the minibatches are taken in that order without shuffling. What do you expect?",
      ["Exactly the same training as with shuffling", "Each minibatch pulls the model towards one class, so the loss oscillates and training is poor", "Training is faster because batches are clean", "The test accuracy is guaranteed to be 50%", "A runtime error"], 1,
      "Every minibatch gradient then only 'sees' one class, so consecutive steps push in opposite directions. Shuffling makes each minibatch a fair sample of the whole data set."),
    mc("reason", r`GD zig-zags across a long, narrow valley. What does momentum with $\rho = 0.9$ do?`,
      ["It makes the zig-zag worse", "Opposite steps across the valley cancel in the velocity, while steps along the valley add up (up to about 10× faster)", "It sets the learning rate to 0.9", "It removes the need for gradients", "It only changes the first step"], 1,
      r`The velocity is a running sum of past gradients. Components that flip sign every step cancel; consistent components accumulate to about $1/(1-\rho) = 10$ times one step.`),
    mc("reason", "One feature is a house's area (values around 1,000) and another its number of bathrooms (values 1–4). Plain GD on linear regression is very slow. Why, and what fixes it?",
      ["The loss is non-convex; use a neural network", "The loss is a very elongated bowl (badly conditioned); standardize the features", "The learning rate is too small for both; increase it 100×", "Too few examples; collect more data", "The bias is missing; add a column of ones"], 1,
      "The curvature along the area weight is far larger (it grows with the square of the feature's values). A learning rate small enough for that direction crawls along the other one. Standardizing gives both features the same scale, which makes the bowl round."),
    mc("reason", "Every layer of a 20-layer MLP uses the sigmoid. The first layers barely change during training. What is the best fix?",
      ["Use sigmoid only in the first layer", "Replace the sigmoids by ReLUs (and/or add skip connections)", "Lower the learning rate", "Add more sigmoid layers", "Use MSE instead of cross-entropy"], 1,
      r`Backprop multiplies by $\sigma'(z) \le 0.25$ at every layer, so the gradient reaching layer 1 shrinks like $0.25^{19}$: vanishing gradients. ReLU's slope is 1 for active units, and skip connections give the gradient a direct path.`),
    mc("reason", "You multiply the loss by 2 (e.g. sum over 2 copies of the data). How does training change with plain SGD and with Adam (same η)?",
      ["Both unchanged", "SGD: like doubling the learning rate. Adam: almost unchanged", "SGD unchanged, Adam's steps double", "Both take steps twice as large", "Both take steps half as large"], 1,
      r`SGD's step is $\eta\mathbf g$, so it doubles. Adam divides the averaged gradient by the square root of the averaged squared gradient: scaling g by 2 scales both by 2, and the ratio stays the same (up to ε).`),
    mc("reason", "A loss uses the sum over a minibatch of 100 examples instead of the mean, and the learning rate was tuned for the mean. What happens?",
      ["Nothing", "The gradients are 100× larger, as if the learning rate were 100× larger: training likely diverges", "The gradients are 100× smaller", "Only the reported loss changes", "The model can only predict one class"], 1,
      "The sum is 100 × the mean, so its gradient is 100× larger too. Unless the learning rate is divided by 100, the steps are 100× too long."),
    mc("reason", r`Gradient clipping with threshold $C = 5$ receives the gradient $\mathbf g = (6, 8)$. What is applied?`,
      ["(5, 5)", "(3, 4)", "(6, 8)", "(0.6, 0.8)", "(5, 8)"], 1,
      r`$\|\mathbf g\| = \sqrt{36 + 64} = 10 > 5$, so g is scaled by $5/10$: (3, 4). The direction is kept; only the length is capped. (Clipping each entry at 5 would give (5, 5), which changes the direction.)`),
    mc("reason", "You train for many more epochs on a small data set, with no early stopping or regularization. What do the curves typically show?",
      ["Training and validation loss both keep decreasing", "Training loss keeps decreasing; validation loss reaches a minimum and then rises", "Training loss rises; validation loss falls", "Both stay constant after epoch 1", "Validation loss always ends below training loss"], 1,
      "Later epochs fit details of the training set that don't generalize. Early stopping keeps the parameters from the epoch with the lowest validation loss."),
    mc("reason", r`What happens with momentum $\rho = 1$ (no decay of the velocity)?`,
      ["The same as plain GD", "The velocity never forgets past gradients, so the parameters overshoot and never settle", "Training stops after one step", "It equals Adam", "The learning rate becomes 0"], 1,
      r`With $\mathbf v_{t+1} = \mathbf v_t - \eta\mathbf g_t$, every gradient ever seen keeps pushing at full strength. Near the minimum the accumulated velocity carries θ straight past it, back and forth, without damping. ρ < 1 makes old gradients fade.`),
    mc("reason", r`For which learning rates does GD converge on $J(\theta) = \frac{h}{2}\theta^2$ (with $h > 0$)?`,
      [r`$\eta < h$`, r`$0 < \eta < 2/h$`, r`$\eta > 2/h$`, r`$0 < \eta < h/2$`, "Any η > 0"], 1,
      r`One step gives $\theta \leftarrow (1 - \eta h)\theta$. That shrinks θ exactly when $|1 - \eta h| < 1$, i.e. $0 < \eta < 2/h$. Steeper curvature (larger h) allows only smaller steps.`),
  ],
  // ------------------------------------------------------------ Generalization what-ifs
  "practice/generalization": [
    mc("reason", "You add much more training data and keep the same flexible model. What typically happens?",
      ["Training loss falls, validation loss rises", "Training loss rises a little (it's harder to memorize), validation loss falls: the gap shrinks", "Both rise", "Nothing changes", "The model underfits more"], 1,
      "With more examples the model can't fit every point exactly, so the training loss goes up slightly, but what it learns carries over better to new data."),
    mc("reason", "On 20 noisy training points, you raise the polynomial degree from 3 to 15. Which pair of changes do you expect?",
      ["Training error ↑, validation error ↓", "Training error ↓, validation error ↑", "Both ↓", "Both ↑", "No change in either"], 1,
      "More capacity always fits the training data at least as well; with only 20 points, the extra freedom goes into fitting the noise, which hurts on new data (overfitting)."),
    mc("reason", r`You increase the L2 penalty $\lambda$ to a huge value (the bias is not penalized). What does the model predict?`,
      ["Exactly the training labels", "Roughly the same value for every input (about the mean label): it underfits", "Random values", "Very large values", "The same as with λ = 0"], 1,
      "The penalty dominates the loss, so all weights are pushed to ≈ 0 and only the unpenalized bias remains: a constant prediction. Too much regularization means underfitting."),
    mc("reason", "Dropout with p = 0.9 (90% of the units dropped) on a medium-sized MLP. What do you expect compared with p = 0.2?",
      ["Less underfitting", "Higher training loss and likely worse validation accuracy: too strong regularization", "Faster convergence to a lower training loss", "No effect, because dropout is off at test time", "Dropout can't be above 0.5"], 1,
      "Only one unit in ten survives each step, so the network can barely fit the training data. Regularization is a dial: too much underfits."),
    mc("reason", "A model with dropout is evaluated without calling model.eval(). What happens to the reported test accuracy?",
      ["Unchanged", "Lower and different from run to run, because units are still dropped at random", "Higher, because of the scaling by 1/(1−p)", "It is exactly 0", "A runtime error"], 1,
      "In training mode dropout keeps zeroing units, so each test prediction comes from a random, weakened sub-network."),
    mc("reason", "You pick the learning rate and the number of epochs that give the best test accuracy, and report that test accuracy. What's wrong?",
      ["Nothing", "The test set was used for choices, so the reported accuracy is too optimistic; use a validation set for the choices", "The test set should be larger than the training set", "Hyperparameters must not be tuned at all", "The training accuracy should be reported instead"], 1,
      "Any choice made by looking at test results fits the test set a little. Choose on validation data, and use the test set once at the end."),
    mc("reason", "A spam filter reaches 95% accuracy, and 95% of the emails are not spam. What should you check first?",
      ["Nothing: 95% is excellent", "Whether it simply predicts 'not spam' for everything (check the confusion matrix, or the recall on spam)", "Whether the learning rate is too small", "Whether the test set is too large", "Whether the model uses a bias"], 1,
      "Always predicting the majority class already scores 95% here. Accuracy hides that; per-class numbers (precision, recall) don't."),
    mc("reason", "Linear regression with D = 500 features and only N = 100 training examples, using the closed form. What goes wrong, and what helps?",
      ["Nothing", r`$\tilde{\mathbf X}^\top\tilde{\mathbf X}$ is not invertible (infinitely many perfect fits); L2 regularization makes it invertible`, "The loss becomes non-convex; use GD", "The bias can't be learned", "The gradient is always zero"], 1,
      r`$\tilde{\mathbf X}^\top\tilde{\mathbf X}$ is 501 × 501 but has rank at most 100, so it's singular: the data can be fitted perfectly in infinitely many ways. Adding $\frac{N\lambda}{2}\mathbf I$ (ridge) makes it invertible and picks the small-weight solution.`),
    mc("reason", "For a classifier of photos of digits, which augmentation is a mistake?",
      ["Shift by 1–2 pixels", "Slight change of brightness", "Rotation by 180°", "Small random crop and resize", "Adding a little noise"], 2,
      "A 6 rotated by 180° looks like a 9: that augmentation changes the label. Good augmentations keep the label correct."),
    mc("reason", "A training set has 30% wrong labels. A very large network reaches 100% training accuracy. What does this tell you?",
      ["The labels must have been fixed during training", "The network memorized the wrong labels too; its test accuracy will be well below 100%", "The network generalizes perfectly", "The learning rate was too small", "Nothing about generalization"], 1,
      "A big enough network can fit random labels. Fitting the wrong labels can't help on new data, so this is overfitting."),
    mc("reason", "In batch normalization, why does the minibatch size matter at training time?",
      ["It doesn't", "The mean and variance are estimated from the minibatch: tiny batches give noisy (with B = 2: ±1 only) normalized values", "Large batches make it non-differentiable", "It only matters at test time", "BN ignores the batch"], 1,
      "With B = 2 every normalized value is ±1, whatever the inputs. Very small batches make the statistics, and so the network's forward pass, unreliable."),
    mc("reason", "You have only 500 labelled images for a new task. Which plan uses them best?",
      ["Train a large CNN from scratch", "Take a CNN pretrained on a large data set, replace its head, and train the head (optionally fine-tune with a small learning rate)", "Train only the first layer", "Use a linear model on raw pixels", "Augment by flipping labels"], 1,
      "Pretrained layers already extract general features (edges, textures, parts). Training only a small head needs far fewer examples."),
  ],
  // ------------------------------------------------------------ PyTorch results
  "practice/pytorch": [
    mc("code", "What does this print?", ["0.0", "1.0", "2.3026", "23.026", "0.1"], 2,
      r`All-zero logits give a uniform softmax: every class gets 1/10. The cross-entropy is $-\ln(1/10) = \ln 10 \approx 2.3026$ for every example, whatever the labels. This is the loss to expect from a classifier with near-zero weights before training.`,
      r`import torch
logits = torch.zeros(4, 10)                 # 4 examples, 10 classes, all logits 0
labels = torch.tensor([3, 0, 9, 1])
loss = torch.nn.CrossEntropyLoss()(logits, labels)
print(round(loss.item(), 4))`),
    num("code", "What does this print?", 0.6931,
      r`A logit of 0 means probability $\sigma(0) = 0.5$, and the BCE is $-\ln 0.5 = \ln 2 \approx 0.6931$ whether the label is 0 or 1.`,
      r`import torch
logits = torch.zeros(3)
y = torch.tensor([1.0, 0.0, 1.0])
print(round(torch.nn.BCEWithLogitsLoss()(logits, y).item(), 4))`),
    mc("code", "What does this print?", ["tensor([1., 2., 3.])", "tensor([2., 4., 6.])", "tensor([1., 4., 9.])", "tensor(14.)", "None"], 1,
      r`$y = \sum_i x_i^2$, so $\partial y/\partial x_i = 2x_i$.`,
      r`import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)`),
    mc("code", "What does this print?", ["tensor([2., 2.])", "tensor([6., 6.])", "tensor([10., -2.])", "tensor([30., -6.])", "an error: backward called twice"], 1,
      "Each backward() adds the gradient of (2x).sum(), which is 2 per entry, to .grad. Three calls: 2 + 2 + 2 = 6. The values of x don't matter.",
      r`import torch
x = torch.tensor([5.0, -1.0], requires_grad=True)
for _ in range(3):
    (2 * x).sum().backward()
print(x.grad)`),
    mc("code", "What does this print?", ["True True", "False False", "False True", "True False", "an error"], 2,
      "Inside no_grad nothing is recorded, so y doesn't track gradients. Outside, z depends on x (which requires gradients), so it does.",
      r`import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    y = x * 2
z = x * 2
print(y.requires_grad, z.requires_grad)`),
    mc("code", "What does this print?", ["tensor([1., 1., 1., 1.])", "tensor([0.5000, 0.5000, 0.5000, 0.5000])", "tensor([2., 2., 2., 2.])", "a random mix of 0. and 2.", "tensor([0., 0., 0., 0.])"], 0,
      "After eval(), dropout does nothing: the input passes through unchanged. (In training mode each entry would be 0 or 1/(1 − 0.5) = 2.)",
      r`import torch
drop = torch.nn.Dropout(p=0.5)
drop.eval()
print(drop(torch.ones(4)))`),
    mc("code", "Which values appear in the output (it prints the sorted distinct values)?", ["[0.0, 1.0]", "[0.5, 1.0]", "[0.0, 2.0]", "[1.0]", "[0.0, 0.5]"], 2,
      "A new module is in training mode. Each entry is dropped (0) with probability 0.5, and the kept ones are divided by 1 − p = 0.5, so they become 2. On average the output still equals the input.",
      r`import torch
torch.manual_seed(0)
drop = torch.nn.Dropout(p=0.5)        # a new module is in training mode
out = drop(torch.ones(1000))
print(sorted(set(out.tolist())))`),
    mc("code", "What does this print?", ["torch.Size([8, 16, 32, 32])", "torch.Size([8, 16, 16, 16])", "torch.Size([8, 3, 16, 16])", "torch.Size([8, 16, 15, 15])", "torch.Size([16, 8, 16, 16])"], 1,
      r`padding=1 with a 3×3 kernel keeps 32 × 32 (32 + 2 − 3 + 1 = 32), the channels become 16, and MaxPool2d(2) halves height and width: (8, 16, 16, 16).`,
      r`import torch
x = torch.zeros(8, 3, 32, 32)
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
pool = torch.nn.MaxPool2d(2)
print(pool(conv(x)).shape)`),
    mc("code", "What does this print?", ["tensor([1, 2])", "tensor([0, 0, 1])", "tensor([2, 1])", "tensor([1, 0, 1])", "tensor(5)"], 1,
      "dim=0 compares down the columns: for each class, which example has the larger score (column 0: 1 > 0 → 0; column 1: 3 > −1 → 0; column 2: 5 > 2 → 1). The predicted class per example would need dim=1, giving tensor([1, 2]).",
      r`import torch
scores = torch.tensor([[1.0, 3.0, 2.0],
                       [0.0, -1.0, 5.0]])   # 2 examples, 3 classes
print(torch.argmax(scores, dim=0))`),
    num("code", "The predictions equal the labels, so you'd expect 0. What does it print?", 0.5,
      "(2, 1) − (2,) broadcasts to (2, 2): [[1−1, 1−2], [2−1, 2−2]] = [[0, −1], [1, 0]]. The mean of the squares is 2/4 = 0.5. PyTorch only prints a warning. Fix: y_hat.squeeze(1).",
      r`import torch
y_hat = torch.tensor([[1.0], [2.0]])   # model output, shape (2, 1)
y = torch.tensor([1.0, 2.0])           # labels, shape (2,)
print(torch.nn.MSELoss()(y_hat, y).item())`),
    num("code", "What does this print?", 67,
      "(4 + 1)·8 + (8 + 1)·3 = 40 + 27 = 67. The ReLU has no parameters.",
      r`import torch
model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
print(sum(p.numel() for p in model.parameters()))`),
    num("code", "What does this print?", 1.4,
      r`The gradient is $2(w - 3) = -4$ at w = 1, so $w \leftarrow 1 - 0.1\cdot(-4) = 1.4$.`,
      r`import torch
w = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
loss = (w - 3) ** 2
loss.backward()
opt.step()
print(round(w.item(), 4))`),
    num("code", "What does this print?", 0.42,
      "PyTorch's momentum: buf ← 0.9·buf + g, then w ← w − lr·buf. Step 1: buf = 2, w = 1 − 0.2 = 0.8. Step 2: buf = 0.9·2 + 2 = 3.8, w = 0.8 − 0.38 = 0.42. (The second step is 1.9× the first: momentum is building up.)",
      r`import torch
w = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1, momentum=0.9)
for _ in range(2):
    opt.zero_grad()
    (2 * w).backward()        # the gradient is always 2
    opt.step()
print(round(w.item(), 4))`),
    mc("code", "What does this print?", ["[2.5, -1.5]", "[3.5, -0.5]", "[2.0, -2.0]", "[4.5, 2.5]", "[1.5, -2.5]"], 0,
      r`Each row computes $x_1 - x_2 + 0.5$: $3 - 1 + 0.5 = 2.5$ and $0 - 2 + 0.5 = -1.5$. nn.Linear computes x @ W.T + b.`,
      r`import torch
lin = torch.nn.Linear(2, 1)
with torch.no_grad():
    lin.weight[:] = torch.tensor([[1.0, -1.0]])
    lin.bias[:] = 0.5
    print(lin(torch.tensor([[3.0, 1.0], [0.0, 2.0]])).squeeze(1).tolist())`),
    mc("code", "What does this print?", ["tensor([[0.3333, 0.3333, 0.3333]])", "tensor([[0.2500, 0.2500, 0.5000]])", "tensor([[0.2000, 0.2000, 0.6000]])", "tensor([[1.0000, 1.0000, 1.6931]])", "tensor([[0.0000, 0.0000, 1.0000]])"], 1,
      r`Only differences of logits matter: the third is larger by $\ln 2$, so $e^{z_3}$ is twice $e^{z_1}$. Probabilities in the ratio 1 : 1 : 2 → 0.25, 0.25, 0.5.`,
      r`import math, torch
z = torch.tensor([[1.0, 1.0, 1.0 + math.log(2)]])
print(torch.softmax(z, dim=1))`),
    num("code", "zero_grad() is missing. What does this print? (With zero_grad it would print 0.36.)", 0.56,
      "Step 1: grad = 2(0 − 1) = −2, w = 0.2. Step 2: the new gradient 2(0.2 − 1) = −1.6 is ADDED to the old −2, giving −3.6, so w = 0.2 + 0.36 = 0.56. The stale gradient makes the step too big.",
      r`import torch
w = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
for _ in range(2):
    loss = (w - 1) ** 2
    loss.backward()           # no zero_grad: gradients add up
    opt.step()
print(round(w.item(), 4))`),
    mc("code", "What does this print?", ["tensor(12.)", "tensor(4.)", "tensor(8.)", "tensor(2.)", "None"], 1,
      r`detach() makes y a constant (4) for autograd, so $z = 4x$ and $dz/dx = 4$. Without detach, $z = x^3$ and the gradient would be $3x^2 = 12$.`,
      r`import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y.detach() * x
z.backward()
print(x.grad)`),
    mc("code", "What does this print?", ["2.3 2.3", "2.3 230.26", "230.26 2.3", "0.0 0.0", "2.3 23.03"], 1,
      "The default reduction is the mean over the 100 examples (ln 10 ≈ 2.30); reduction='sum' adds them up: 100 · 2.3026 = 230.26. That is the midterm handout's sample question 3.",
      r`import torch
logits = torch.zeros(100, 10)
labels = torch.zeros(100, dtype=torch.long)
mean_loss = torch.nn.CrossEntropyLoss()(logits, labels)
sum_loss = torch.nn.CrossEntropyLoss(reduction="sum")(logits, labels)
print(round(mean_loss.item(), 2), round(sum_loss.item(), 2))`),
    mc("code", "What does this print (so what must in_features of the next Linear layer be)?", ["torch.Size([2, 576])", "torch.Size([2, 784])", "torch.Size([2, 676])", "torch.Size([2, 4, 12, 12])", "torch.Size([2, 2304])"], 0,
      "28 − 5 + 1 = 24, pooled to 12: (2, 4, 12, 12), flattened to 4 · 12 · 12 = 576.",
      r`import torch
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, 5),      # no padding
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
)
print(model(torch.zeros(2, 1, 28, 28)).shape)`),
    mc("code", "What does this print?", ["tensor([1., 3.])", "tensor([-1.,  1.])", "tensor([0., 0.])", "tensor([-0.5,  0.5])", "tensor([0.25, 0.75])"], 1,
      "In training mode batch norm standardizes over the minibatch: mean 2, standard deviation 1, so (1 − 2)/1 = −1 and (3 − 2)/1 = 1 (γ = 1, β = 0 initially; ε is negligible). Any two different values give ±1.",
      r`import torch
bn = torch.nn.BatchNorm1d(1)
x = torch.tensor([[1.0], [3.0]])       # a minibatch of 2 examples, 1 feature
print(bn(x).squeeze(1).detach().round(decimals=3))`),
    mc("code", "What does this print?", ["tensor([0, 1, 1])", "tensor([0.5000, 1.0000, 1.5000])", "tensor([0, 1, 2])", "an error: int tensors can't be divided", "tensor([1, 2, 3])"], 1,
      "The / operator is true division, so integer tensors give floats (like Python 3). Integer division would be //.",
      r`import torch
print(torch.tensor([1, 2, 3]) / 2)`),
    mc("code", "What does this print?", ["tensor([0., 0.5, 3.])", "tensor([0., 1., 1.])", "tensor([1., 1., 1.])", "tensor([-2., 0.5, 3.])", "tensor([0., 0., 1.])"], 1,
      "ReLU's slope is 0 for negative inputs and 1 for positive ones. The values 0.5 and 3 don't matter, only their sign.",
      r`import torch
z = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
torch.relu(z).sum().backward()
print(z.grad)`),
  ],
  // ------------------------------------------------------------ Diagnose
  "practice/diagnose": [
    mc("reason", "A 10-class classifier with small random weights starts at a training loss of about 2.30. Is that a problem?",
      ["Yes: it should start near 0", "No: ln 10 ≈ 2.30 is exactly the loss of a uniform guess, as expected before training", "Yes: it should start near 1", "Yes: it means the labels are wrong", "Only if the learning rate is small"], 1,
      "Near-zero logits give probability 1/10 for every class, and −ln(1/10) = 2.30. A start far above that (e.g. 20) would point to huge initial logits or a summed loss."),
    mc("reason", "The training loss stays at exactly 2.3026 for 20 epochs (10 classes). What is the most likely cause?",
      ["The model is overfitting", "The parameters aren't being updated at all (e.g. optimizer.step() missing, learning rate 0, or the optimizer got the wrong parameters)", "The data set is too large", "Too much data augmentation", "The test set is too small"], 1,
      "A loss frozen at the uniform-guess value means the model still outputs uniform probabilities: nothing is changing. Check that gradients exist and that the parameters actually change after a step."),
    mc("reason", "The loss becomes nan after a few hundred steps. Which pair of fixes is most sensible to try first?",
      ["More epochs and more layers", "A smaller learning rate (or gradient clipping), and a numerically safe loss such as BCEWithLogitsLoss instead of log(sigmoid(…))", "Remove the bias", "Use accuracy as the loss", "Switch off shuffling"], 1,
      "nan usually means something blew up (steps too large → huge values → inf − inf) or log(0) was taken. Smaller steps and loss functions that work on logits avoid both."),
    mc("reason", "Validation loss is lower than training loss during the whole training run. The model uses dropout. What's the most likely reason?",
      ["The validation set leaked into training", "Dropout is active when the training loss is measured (during the epoch) but not on validation, so the training numbers are measured on a weakened network", "The model underfits badly", "The learning rate is too high", "Batch norm is broken"], 1,
      "Training-mode dropout makes the network worse on purpose. Also, the training loss is averaged over an epoch in which the model was still improving. Neither indicates a bug."),
    mc("reason", "Training accuracy climbs to 95%, but validation accuracy stays at chance level (10% for 10 classes) from the start. What do you suspect?",
      ["Normal overfitting", "A bug in the validation pipeline: e.g. labels misaligned with inputs, or a different preprocessing (normalization) for validation data", "The learning rate is too small", "Too much regularization", "The model is too small"], 1,
      "Overfitting shows validation accuracy that rises and then falls, not one stuck at chance. Exactly-chance performance means the model's outputs have nothing to do with the validation labels."),
    mc("reason", "A model predicts the same class for every input. 90% of the training examples belong to that class. What's going on, and what helps?",
      ["The model is perfect", "Predicting the majority class is a cheap way to get a low loss; weight the rare classes more in the loss, or resample, and judge it with per-class metrics", "The learning rate is too large", "Use MSE instead of cross-entropy", "Remove the bias"], 1,
      "With heavy class imbalance, the 'always majority' answer already scores 90% accuracy. Reweighting makes mistakes on the rare class cost more."),
    mc("reason", "A degree-9 polynomial fit on 10 points has huge weights with alternating signs. Which explanation and fix fit best?",
      ["Underfitting; raise the degree", "Overfitting: the curve passes through every noisy point by large cancelling terms; add L2 regularization or lower the degree", "The learning rate is too small", "The data must be shuffled", "The bias is too large"], 1,
      "Huge alternating weights are the signature of an overfit polynomial (Lecture 4). Weight decay penalizes exactly those large weights."),
    mc("reason", "Gradient descent's loss decreases, but very slowly, and the parameters zig-zag. Which change is least likely to help?",
      ["Standardizing the features", "Adding momentum", "Using Adam", "Multiplying the learning rate by 100", "Rescaling the badly scaled feature"], 3,
      "The zig-zag means the learning rate is already too large for the steep direction. Increasing it makes the zig-zag diverge. The other four make the valley rounder or damp the zig-zag."),
    mc("reason", "In a deep sigmoid network, the gradients of the first layer are around 10⁻⁸ while the last layer's are around 10⁻². What explains the result?",
      ["A bug in autograd", "Vanishing gradients: each sigmoid layer multiplies the backpropagated gradient by at most 0.25", "Exploding gradients", "Overfitting", "The data isn't shuffled"], 1,
      "Through many layers these factors multiply into tiny numbers. ReLU, skip connections and normalization layers counter it."),
    mc("reason", "Logistic regression on linearly separable data: the training loss keeps shrinking and the weights keep growing, epoch after epoch. Why?",
      ["A bug: the loss should stop at a positive value", "There's no minimum: scaling the weights up makes every probability more confident and correct, lowering the BCE forever", "The learning rate is too small", "The model underfits", "Gradient descent always diverges for logistic regression"], 1,
      "On separable data, larger weights always reduce the loss. L2 regularization (or early stopping) gives the problem a finite solution."),
    mc("reason", "The training loss spikes at the same point of every epoch. 10,002 training examples, batch size 100, a model with batch norm. The cause?",
      ["A bad example in the data", "The last minibatch has only 2 examples, so batch norm maps every feature to ±1 on that step; use drop_last=True for training", "Momentum", "The learning rate is too small", "Dropout"], 1,
      "10002 = 100·100 + 2. With 2 examples, batch norm's (z − μ)/σ is ±1 whatever the inputs, so that step's forward pass is nonsense. It happens once per epoch, at the same point."),
    extra(mc("reason", "A model scores 99% on the test set, but performs badly once deployed. Which is the most likely explanation?",
      ["Too much regularization", "The test set didn't represent real use (distribution shift), or it leaked into training (duplicates, preprocessing fitted on all data)", "The learning rate was too small", "The model has too few parameters", "Dropout was used"], 1,
      "A test score is only meaningful for data like the test set, and only if nothing about the test set influenced training. Both conditions often fail in practice.")),
  ],
  // ------------------------------------------------------------ Short math
  "practice/short-math": [
    num("calc", "The model gives the correct class probability 0.5. What is this example's cross-entropy (to 2 decimals)?", 0.693,
      r`$-\ln 0.5 = \ln 2 \approx 0.69$.`),
    num("calc", "A 4-class model outputs (0.25, 0.25, 0.25, 0.25). What is its cross-entropy?", 1.386,
      r`$-\ln 0.25 = \ln 4 = 2\ln 2 \approx 1.39$, whatever the true class.`),
    num("calc", "Binary cross-entropy for $\\hat y = 0.9$ when the label is $y = 0$?", 2.303,
      r`Only the $(1-y)\ln(1-\hat y)$ term is active: $-\ln(0.1) = \ln 10 \approx 2.30$. Confidently wrong is expensive.`),
    num("calc", r`GD on $J(\theta) = 3\theta^2$ from $\theta_0 = 2$ with $\eta = 0.1$. What is $\theta_1$?`, 0.8,
      r`$J'(\theta) = 6\theta = 12$, so $\theta_1 = 2 - 0.1\cdot12 = 0.8$. (Equivalently $\theta_1 = (1 - 0.6)\theta_0$.)`),
    num("calc", r`GD on $J(\theta) = 5\theta^2$: what is the largest learning rate (the boundary value) for which it still converges?`, 0.2,
      r`$J'' = h = 10$, and GD converges for $0 < \eta < 2/h = 0.2$.`),
    num("calc", r`With momentum $\rho = 0.9$ and a gradient that stays constant, the step size eventually becomes how many times $\eta g$?`, 10,
      r`The velocity tends to $\eta g(1 + \rho + \rho^2 + \dots) = \eta g/(1-\rho) = 10\,\eta g$.`),
    num("calc", "Predictions (1, 2, 3), labels (1, 1, 1). What is the MSE?", 5 / 3,
      r`$(0^2 + 1^2 + 2^2)/3 = 5/3 \approx 1.67$.`),
    num("calc", "How many parameters does a 3-4-2 MLP (with biases) have?", 26,
      r`$(3+1)\cdot4 + (4+1)\cdot2 = 16 + 10 = 26$.`),
    num("calc", "A 1-D convolution: input length D = 20, kernel F = 5, padding P = 2, stride s = 3. Output length?", 7,
      r`$\lfloor(20 + 4 - 5)/3\rfloor + 1 = \lfloor 19/3\rfloor + 1 = 6 + 1 = 7$.`),
    num("calc", "Inverted dropout with p = 0.2 during training. A unit's value is 4 and it is kept. What value is passed on?", 5,
      r`Kept units are divided by $1 - p = 0.8$: $4/0.8 = 5$, so the expected value stays 4.`),
    num("calc", r`Weight decay: $\theta = 2$, $\eta = 0.1$, $\lambda = 0.5$, and the data gradient is 0 at this step. What is the new θ?`, 1.9,
      r`$\theta \leftarrow \theta - \eta(\nabla J + \lambda\theta) = (1 - \eta\lambda)\theta = 0.95\cdot2 = 1.9$.`),
    num("calc", r`Logistic regression, one example $\tilde{\mathbf x} = (2, 1)$ (feature, then the 1), $\hat y = 0.7$, $y = 1$. What is $\partial J/\partial w$?`, -0.6,
      r`Error × input: $(\hat y - y)\,x = (0.7 - 1)\cdot2 = -0.6$. Negative, so GD increases w.`),
    num("calc", r`Fit $\hat y = \theta x$ (no bias) by least squares to $x = (1, 2)$, $y = (2, 3)$. What is $\theta^\star$?`, 1.6,
      r`Setting the derivative of $\sum_n(\theta x_n - y_n)^2$ to 0 gives $\theta^\star = \frac{\sum_n x_ny_n}{\sum_n x_n^2} = \frac{2 + 6}{1 + 4} = 1.6$.`),
    num("calc", r`$J(\theta) = \theta_1^2 + 3\theta_2^2$. What is the condition number of its Hessian?`, 3,
      r`$\mathbf H = \mathrm{diag}(2, 6)$, so $\kappa = 6/2 = 3$.`),
    num("calc", r`One neuron: $\hat y = \sigma(wx)$, loss $(\hat y - t)^2$, with $x = 1$, $w = 0$, $t = 1$. What is $dL/dw$?`, -0.25,
      r`$\hat y = \sigma(0) = 0.5$. Chain rule: $\frac{dL}{d\hat y} = 2(0.5 - 1) = -1$, $\frac{d\hat y}{dz} = 0.5\cdot0.5 = 0.25$, $\frac{dz}{dw} = x = 1$. Product: $-0.25$.`),
    num("calc", r`Softmax of the logits $(\ln 3,\ 0)$: the probability of class 1?`, 0.75,
      r`$\frac{e^{\ln 3}}{e^{\ln 3} + e^0} = \frac{3}{3 + 1} = 0.75$.`),
    num("calc", "Accuracy when 3 of 8 test examples are misclassified?", 0.625,
      r`$5/8 = 0.625$.`),
    num("calc", r`Two GD steps on $J(\theta) = \theta^2$ from $\theta_0 = 1$ with $\eta = 0.25$. What is $\theta_2$?`, 0.25,
      r`Each step multiplies θ by $1 - 2\eta = 0.5$: $1 \to 0.5 \to 0.25$.`),
  ],
  // ------------------------------------------------------------ Beyond the slides
  "practice/beyond": [
    extra(mc("reason", "In the bias–variance picture, what does a very flexible model (e.g. a degree-15 polynomial on 20 points) suffer from most?",
      ["High bias", "High variance: its fit changes a lot from one training sample to another", "Neither", "Both high bias and high variance, equally", "Low variance and high bias"], 1,
      "Bias = error from a too-simple model (underfitting); variance = sensitivity to the particular training sample (overfitting). Flexible models trained on little data have high variance.")),
    extra(num("calc", "A classifier on a test set: 30 true positives, 10 false positives, 20 false negatives. What is its precision?", 0.75,
      r`Precision = TP / (TP + FP) = 30/40 = 0.75: of everything flagged positive, 75% really is.`)),
    extra(num("calc", "Same classifier (TP = 30, FP = 10, FN = 20). What is its recall?", 0.6,
      r`Recall = TP / (TP + FN) = 30/50 = 0.6: it finds 60% of the actual positives.`)),
    extra(num("calc", "Same classifier: precision 0.75, recall 0.6. What is the F1 score?", 2 / 3,
      r`$F_1 = \frac{2PR}{P + R} = \frac{2\cdot0.75\cdot0.6}{1.35} = \frac{0.9}{1.35} \approx 0.667$ (the harmonic mean, pulled towards the smaller of the two).`)),
    extra(num("calc", "5-fold cross-validation on 1,000 examples: how many examples is each of the 5 models trained on?", 800,
      "The data is split into 5 folds of 200; each model trains on 4 folds (800) and is validated on the remaining one. Every example is used for validation exactly once.")),
    extra(mc("reason", "L1 regularization (λ‖θ‖₁) versus L2 (λ‖θ‖²/2): which statement is true?",
      ["They always give the same solution", "L1 tends to set many weights exactly to 0 (sparse models); L2 shrinks all weights but rarely to exactly 0", "L2 gives sparse models, L1 doesn't", "L1 has no effect on large weights", "Only L2 can be used with gradient descent"], 1,
      "L1's gradient has constant size λ·sign(θ) no matter how small θ is, so it pushes small weights all the way to 0. L2's gradient λθ fades as θ shrinks.")),
    extra(mc("reason", "All weights and biases of an MLP are initialized to exactly 0 (tanh hidden units). What happens during training?",
      ["It trains normally", "All hidden units in a layer stay identical forever (symmetry), so the layer acts like a single unit", "The loss is nan", "Only the biases learn", "The network becomes linear"], 1,
      "Identical units receive identical gradients, so they remain identical after every step. Random initialization breaks the symmetry. (With ReLU and exact zeros the gradients would even be 0 everywhere.)")),
    extra(mc("reason", "Why are weights initialized with a variance that depends on the number of inputs (e.g. 2/fan_in for ReLU, 'He initialization')?",
      ["To make the network sparse", "So that activations and gradients keep roughly the same size from layer to layer instead of shrinking or exploding", "To speed up the forward pass", "Because biases must be 0", "It is required by autograd"], 1,
      "A unit sums fan_in inputs; with too-large weights the sums grow layer after layer, with too-small ones they vanish. Scaling the variance by 1/fan_in keeps them stable.")),
    extra(mc("reason", "A learning-rate schedule lowers η during training (e.g. divide it by 10 after some epochs). What's the purpose?",
      ["To overfit faster", "Large steps early make fast progress; smaller steps later let SGD settle into a minimum instead of bouncing around it because of the gradient noise", "To make the loss convex", "To avoid computing gradients", "To change the model's capacity"], 1,
      "With a constant η, SGD's noisy steps keep jittering around the minimum. Shrinking η reduces the jitter.")),
    extra(mc("reason", "You standardize the features using the mean and standard deviation of the whole data set, then split into train and test. What's the problem?",
      ["None", "Data leakage: information from the test set influenced the preprocessing, so the test score is slightly optimistic", "The model can't be trained", "Standardization must happen after training", "The test set becomes larger"], 1,
      "Every learned step, including preprocessing statistics, must come from the training set only (fit the scaler on train, then transform both), as in Lab 4.")),
    extra(num("calc", "Two stacked 3×3 convolutions (stride 1, no pooling). How large, in pixels per side, is the region of the input one output unit sees (its receptive field)?", 5,
      "Each 3×3 layer extends the region by 2 pixels: 1 → 3 → 5. Three such layers see 7×7, which is why stacks of small kernels replace large kernels.")),
    extra(num("calc", "How many parameters does a 1×1 convolution with 64 input channels and 32 output channels have (with biases)?", 2080,
      r`$1\cdot1\cdot64\cdot32 + 32 = 2048 + 32 = 2080$. A 1×1 convolution mixes channels at each pixel; it's a cheap way to reduce the number of channels.`)),
    extra(mc("reason", "Dividing all logits by a temperature T > 1 before the softmax does what to the probabilities?",
      ["Makes them more peaked", "Makes them flatter (closer to uniform), without changing which class is largest", "Changes the predicted class", "Makes them sum to T", "Nothing"], 1,
      "Dividing by T shrinks the differences between logits, and the softmax depends only on those differences. The ordering is unchanged, so argmax is unchanged.")),
    extra(mc("reason", "A rare disease affects 1% of patients. Which loss modification addresses the imbalance directly?",
      ["Lower the learning rate", "A class-weighted cross-entropy that multiplies the rare class's loss terms by a larger weight", "Use MSE", "Remove the bias", "Train for fewer epochs"], 1,
      "Weighting makes each mistake on the rare class cost more, so the model can't ignore it. Resampling (showing rare examples more often) has a similar effect.")),
    extra(mc("reason", "Max pooling makes a CNN's output change little when the input image is shifted by one pixel. This property is called…",
      ["(approximate) translation invariance", "rotation invariance", "overfitting", "weight decay", "batch normalization"], 0,
      "The maximum over a small window often stays the same when the pattern moves by a pixel inside it. Convolutions themselves are translation equivariant (the feature map shifts with the input); pooling adds the invariance.")),
  ],
};
