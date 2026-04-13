# MaxSup Paper Summary (After Proof Sections) + Poster & Implementation Plan
*Conference Booth Project Guide*  
*Download-ready Markdown | 中英结合*

---

# Part I. Summary After Proof Sections（Proof 之后该看什么）

作者完成理论部分（Section 3.1 / 3.2）后，后面重点不是再看证明，而是看：

```text
Does MaxSup actually work?
Why does it work in representation space?
Is it practical?
```

所以重点是 **Section 4 Experiments + Appendix Results**

---

# Section 4.1 Representation Benefits

## Core Claim

MaxSup improves:

* inter-class separability（类间更容易分开）
* intra-class variability（类内保留更多变化）

## Why Important

好的 feature space 应该：

```text
same class not collapse too much
different classes clearly separated
```

## Strong Evidence

### Table 2

Feature quality on ResNet-50:

* higher (d_{within})
* strong (R^2)

Meaning:

* richer class diversity
* still discriminative

---

# Section 4.2 Main Classification Results

## CNN Models

* ResNet18
* ResNet50
* ResNet101
* MobileNetV2

MaxSup consistently best among LS-style methods.

## DeiT-Small (Very Poster-Friendly)

| Method   | Accuracy  |
| -------- | --------- |
| Baseline | 74.39     |
| LS       | 76.08     |
| OLS      | 76.16     |
| MaxSup   | **76.49** |

## Why Important

Shows:

```text
Not CNN-specific
Works for Transformers too
```

---

# Section 4.2 Extra Valuable Results

## Fine-grained Classification

Datasets:

* CUB Birds
* Stanford Cars

MaxSup best.

Meaning:

```text
captures subtle details better
```

---

## Long-tailed Classification

Imbalanced data:

* many-shot
* medium-shot
* low-shot

MaxSup gives better tradeoff.

Meaning:

```text
more robust under imbalance
```

---

## Corrupted / OOD

CIFAR-10-C:

* better calibration
* competitive robustness

Meaning:

```text
confidence more reliable
```

---

# Section 4.3 Semantic Segmentation

ADE20K + UPerNet

| Method   | mIoU     |
| -------- | -------- |
| Baseline | 42.1     |
| LS       | 42.4     |
| MaxSup   | **42.8** |

## Meaning

Backbone features transfer better.

Not only classification.

---

# Section 4.4 Grad-CAM Visualization

LS often attends:

* background
* shortcuts
* spurious regions

MaxSup attends:

* object body
* shape
* meaningful regions

Meaning:

```text
better interpretability
better object-focused reasoning
```

---

# Appendix Results Worth Reading

## MUST SEE

### Table 12

Extended transfer learning.

Very useful for poster.

### Table 13

More baselines comparison.

Answers:

```text
Did they compare enough methods?
```

### Table 16

Compute efficiency.

Shows almost no extra cost.

---

# Final Takeaway

```text
LS regularizes the label target.
MaxSup regularizes the dominant prediction.
```

This small change improves:

* accuracy
* transferability
* segmentation
* calibration
* interpretability

---

# Part II. Poster Plan（Conference Booth Style）

# Poster Layout (Portrait)

```text
┌──────────────────────────┐
│ Title + Names           │
├──────────────────────────┤
│ BIG MAIN VISUAL         │
│ LS vs MaxSup Wrong Case │
├────────────┬────────────┤
│ Problem    │ Method     │
│ Insight    │ Formula    │
├────────────┼────────────┤
│ Results    │ Demo QR    │
│ Tables     │ Laptop     │
├──────────────────────────┤
│ Takeaways + Limitation  │
└──────────────────────────┘
```

---

# Poster Sections

## Title

**MaxSup: Fixing Label Smoothing When the Model Is Wrong**

---

## Main Visual (Largest)

Wrong prediction example:

```text
GT = cat

cat = 1.0
fox = 2.8
dog = 0.4
```

LS:

```text
penalize cat
```

MaxSup:

```text
penalize fox
```

---

## Problem

Label Smoothing is useful, but under misclassified samples it may amplify errors.

---

## Method

\[
L_{LS}=\alpha(z_{gt}-mean(z))
\]

\[
L_{MaxSup}=\alpha(z_{max}-mean(z))
\]

**Only one change:**

\[
z_{gt}\rightarrow z_{max}
\]

---

## Results (Only 3 Blocks)

### Block 1

DeiT-Small:

76.49 > 76.08

### Block 2

Segmentation:

42.8 > 42.4

### Block 3

Grad-CAM:

more focused attention

---

## Bottom Takeaways

* simple
* no architecture change
* low overhead
* stronger representations

---

# Booth Speaking Strategy

## 20 sec version

Label Smoothing can regularize the wrong target when prediction is wrong. MaxSup instead suppresses the dominant wrong logit.

## 1 min version

Use formula + benchmark.

## Deep question version

Talk about:

* error amplification
* feature collapse
* transfer learning

---

# Part III. Implementation Plan (Highly Recommended)

# Best Demo Choice

## Interactive Logit Simulator

User adjusts logits:

| Class | Logit  |
| ----- | ------ |
| cat   | slider |
| fox   | slider |
| dog   | slider |

Then compare:

* softmax probs
* LS gradient direction
* MaxSup gradient direction

---

# Why Powerful

Instantly explains:

```text
Why MaxSup works
```

Much stronger than static poster.

---

# Tech Stack

## Streamlit (Recommended)

```bash
pip install streamlit numpy matplotlib
streamlit run app.py
```

---

# Demo UI Layout

## Left Panel

Sliders:

* cat
* fox
* dog

Choose GT label.

---

## Middle Panel

Bar chart:

* probabilities

---

## Right Panel

Compare:

### LS

penalizes GT logit

### MaxSup

penalizes top-1 logit

---

## Bottom

One-step update animation.

---

# If More Time

## Add Image Upload Demo

Upload confusing image.

Show:

* predicted label
* LS attention
* MaxSup attention

---

# What To Say During Demo

Not:

```text
We coded the paper.
```

Say:

```text
We built an interactive tool to visualize why MaxSup fixes Label Smoothing under wrong predictions.
```

---

# Final Recommended Team Split

## Person 1

Greeter + overview

## Person 2

Theory + formulas

## Person 3

Experiments + tables

## Person 4

Demo operator + Q&A

---

# Last Advice

At conference booths:

```text
People stop for visuals,
stay for insight,
remember simplicity.
```

MaxSup's strength is exactly that.

Use it.
