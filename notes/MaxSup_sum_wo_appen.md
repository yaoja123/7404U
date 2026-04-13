# MaxSup 论文精读最终版 Markdown 笔记（中英结合）

> Paper: **Max Suppression Regularization (MaxSup)**  
> Theme: **Why Label Smoothing fails on misclassified samples, and how MaxSup fixes it by penalizing the top-1 logit instead of the ground-truth logit.**

---

# 0. 一句话总结 / One-line Summary

```text
Label Smoothing penalizes the true class.
MaxSup penalizes the dominant class.
```

```text
LS 压真实类，
MaxSup 压当前最强类。
```

---

# 1. 论文想解决什么问题 / Problem Statement

## Background

Label Smoothing (LS) is widely used in classification because it usually helps:

- reduce overconfidence
- improve generalization
- improve calibration

在多分类任务中，传统 one-hot 标签默认认为各类别彼此完全分离，但现实里类别之间往往共享低层特征或高层语义，因此 LS 被广泛用来 soften target distribution。

## Authors' Core Concern

作者指出：LS 虽然有用，但它并不是纯收益。  
它有两个关键问题：

1. **On misclassified samples, LS may amplify wrong predictions.**  
   在错分样本上，LS 可能放大错误预测的置信度。

2. **LS compresses feature representations into overly tight clusters.**  
   LS 会让同类特征挤得太紧，降低类内多样性。

---

# 2. Section 1 Introduction

## Main Idea

Introduction 的主线是：

- one-hot labels are too rigid
- LS is useful but imperfect
- LS causes both overconfidence under misclassification and feature collapse
- authors propose **MaxSup**
- MaxSup penalizes the top-1 logit rather than the ground-truth logit

## Human Interpretation / 人话版

LS 的原本动机是：

> 正确类不要高到 100%，其他类也别被压到绝对 0。

但作者认为：

> 这在“预测对”的时候大多是好事；  
> 可在“预测错”的时候，LS 可能继续强化当前那个错误 top-1。

所以他们提出：

```text
不要固定去压真实类，
而是压模型当前最自信的那个类。
```

---

# 3. Section 2 Related Work

## 3.1 Regularization

### Traditional regularization methods

- **L2 regularization**: penalize large weights  
- **L1 regularization**: encourage sparse weights  
- **Dropout**: randomly deactivate neurons

### Loss-level regularization

- Label Smoothing
- Confidence Penalty
- Logit Penalty

## Authors' Positioning

作者想强调：

- 以前很多 regularization 方法都很有用
- 但它们通常要么约束参数，要么围绕 ground-truth logit 做事
- MaxSup 的关键不同在于：**它只关注当前最高 logit**

即：

\[
z_{gt} \quad \rightarrow \quad z_{\max}
\]

---

## 3.2 Studies on Label Smoothing

作者回顾了 LS 在以下领域的研究：

- image classification
- calibration
- knowledge distillation
- teacher-student learning

但 prior work 也发现 LS 的副作用：

- tight intra-class clusters
- reduced feature variability
- worse transfer learning
- rigid feature geometry

## Key Position

别人常做的是：

```text
改 soft label 怎么构造
```

比如：

- OLS
- Zipf-LS

而作者认为真正的问题不是：

```text
标签怎么平滑
```

而是：

```text
错分时 regularization 压错了对象
```

---

# 4. Section 3 Method

---

# 4.1 Revisiting Label Smoothing

## Definition 3.1

标准 Label Smoothing 定义为：

\[
s_k = (1-\alpha) y_k + \frac{\alpha}{K}
\]

where:

- \(K\): number of classes
- \(\alpha\): smoothing strength
- \(y\): one-hot label
- \(s\): softened target

## Intuition / 直觉

LS 做的是：

- 从真实类拿走一部分概率质量
- 平均分给所有类

例如 3 分类时，真实类为第 1 类，one-hot 是：

\[
[1,0,0]
\]

如果 \(\alpha = 0.1\)，则 soft label roughly 变成：

\[
[0.9,0.05,0.05]
\]

---

## Lemma 3.2: Decomposition of Cross-Entropy with Soft Labels

作者把 soft-label cross entropy 写成：

\[
H(s,q)=H(y,q)+L_{LS}
\]

其中：

\[
L_{LS}=\alpha\left(H\left(\frac{1}{K}\mathbf{1},q\right)-H(y,q)\right)
\]

### Meaning

这很重要，因为它说明：

```text
LS 不是单纯换标签
而是等价于：
CE loss + 一个额外 regularization-like 项
```

---

## From Eq. (3) to Eq. (4)

作者进一步把 LS 写成 logit 形式：

\[
L_{LS}=\alpha\left(z_{gt}-\frac1K\sum_{k=1}^K z_k\right)
\]

### How it comes out / 怎么来的

从

\[
L_{LS}=\alpha\left(H\left(\frac{1}{K}\mathbf{1},q\right)-H(y,q)\right)
\]

出发，利用：

- cross-entropy definition
- one-hot property
- softmax form of \(q_k\)

即：

\[
q_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
\]

和

\[
\log q_k = z_k - \log\sum_j e^{z_j}
\]

代入后，公共的 \(\log\sum_j e^{z_j}\) 项会抵消，最终得到：

\[
L_{LS}=\alpha\left(z_{gt}-\frac1K\sum_k z_k\right)
\]

### Intuition

LS 本质上是在惩罚：

```text
真实类 logit 比平均 logit 高太多
```

因此它能抑制过度自信。

---

## Corollary 3.4: Two Components of LS

作者进一步把 LS 分成两项：

1. **Regularization term**
2. **Error amplification term**

### Regularization term

作用在比 \(z_{gt}\) 小的 logits 上。  
它会缩小真实类与更小 logits 的 gap，抑制 overconfidence。

### Error amplification term

作用在比 \(z_{gt}\) 大的 logits 上。  
当存在错误类比真实类更高时，这一项可能进一步扩大 gap，导致错误更难纠正。

## Most Important Insight

```text
预测正确时：
LS mostly behaves like regularization

预测错误时：
LS may amplify the error
```

这就是全文最核心的理论结论。

---

## Ablation Study / 什么叫消融实验

Ablation study 的意思是：

> 把方法拆成几个部分，分别测试每个部分的作用。

这篇论文里，作者把 LS 拆成：

- regularization part
- error amplification part

然后分别测试：

- baseline
- full LS
- regularization only
- error amplification only
- MaxSup

### Result

- regularization only works
- error amplification only hurts
- MaxSup works best

这说明作者的理论拆解是有实验支持的。

---

# 4.2 Max Suppression Regularization

## Core Change

LS 的 logit-level form 是：

\[
L_{LS}=\alpha\left(z_{gt}-\frac1K\sum_k z_k\right)
\]

MaxSup 改成：

\[
L_{MaxSup}=\alpha\left(z_{\max}-\frac1K\sum_k z_k\right)
\]

也就是把：

\[
z_{gt} \rightarrow z_{\max}
\]

---

## Definition

作者定义：

\[
H(s,q)=H(y,q)+L_{MaxSup}
\]

并构造：

\[
L_{MaxSup}=\alpha\left(H\left(\frac{1}{K}\mathbf{1},q\right)-H(y',q)\right)
\]

其中 \(y'\) 不是 ground-truth one-hot，  
而是 **current top-1 one-hot**。

即：

\[
y'_k = \mathbf{1}\{k=\arg\max(q)\}
\]

---

## Why this matters

### If prediction is correct

\[
z_{gt}=z_{\max}
\]

这时 MaxSup 和 LS 很像，都能抑制过度自信。

### If prediction is wrong

\[
z_{gt}\neq z_{\max}
\]

这时 LS 还在围绕真实类做 regularization，  
而 MaxSup 直接压当前错误最大类，因而更对症。

---

## Consistent vs Inconsistent Regularization

### Inconsistent regularization

指的是：

> 同一个 regularization 公式，在“预测对”和“预测错”两种情况下，起到的效果不一致。

LS 就是这样：

- 预测对时：效果合理
- 预测错时：可能压错对象，甚至帮倒忙

### Consistent regularization

MaxSup 的目标始终是：

```text
压当前最强 logit
```

所以不论对错，它都更 consistent。

---

## Gradient Analysis

作者给出梯度：

\[
\frac{\partial L_{MaxSup}}{\partial z_k}
=
\begin{cases}
\alpha\left(1-\frac{1}{K}\right), & k=\arg\max(q) \\
-\frac{\alpha}{K}, & \text{otherwise}
\end{cases}
\]

### Intuition

- top-1 logit gets pushed down
- other logits get slightly lifted

所以 MaxSup 的总体效果是：

- 防止某个类过度 dominant
- 但不会像 LS 那样在错分时压制真实类

---

## Collapse 是什么意思

这篇论文里，collapse 主要指：

```text
intra-class collapse
```

即：

- 同一类样本被挤得过于紧
- 类内多样性消失
- 表示变 rigid

MaxSup 的一个核心好处就是：

```text
preserve intra-class variation
```

---

# 5. Section 4 Experiments

---

# 5.1 Analysis of Broader Benefits

作者先不急着讲 accuracy，而是看 representation quality。

重点分析两个性质：

- **inter-class separability**
- **intra-class variability**

## Ideal feature geometry

理想表示空间应当满足：

```text
类间分得开
+
类内不塌缩
```

---

## Table 2: Feature quality

指标包括：

- \(d_{within}\): within-class distance
- \(R^2\): class separability related score

### Reading

- larger \(d_{within}\) → richer intra-class variation
- higher \(R^2\) → better class separation

作者发现：

- LS reduces intra-class diversity
- MaxSup retains richer within-class structure while keeping good separability

---

## Connection to Logit Penalty

Logit Penalty 压的是所有 logits 的整体 \(\ell_2\) norm。  
它像“全局收缩”，容易把特征也压坏。

而 MaxSup 只压：

\[
z_{\max}
\]

因此更 selective，不会过度压缩整个 feature space。

---

## Transfer Accuracy

在线性 probe 实验里，MaxSup 明显优于 LS / Logit Penalty，说明：

```text
MaxSup learns better transferable features.
```

---

# 5.2 Evaluation on ImageNet Classification

## Experimental setup

作者在 ImageNet-1K 上比较：

- Baseline
- LS
- OLS
- Zipf-LS
- MaxSup
- Logit Penalty

Backbones include:

- ResNet-18
- ResNet-50
- ResNet-101
- MobileNetV2
- DeiT-Small

---

## CNN results

Across ConvNets, MaxSup consistently achieves the best or near-best top-1 accuracy.

### Why this matters

这说明它不是某个 backbone-specific trick，  
而是更普遍的训练改进。

---

## DeiT results

在 DeiT-Small 上：

- Baseline: 74.39
- Label Smoothing: 76.08
- OLS: 76.16
- **MaxSup: 76.49**

### Interpretation

MaxSup 不仅在 CNN 上有效，在 transformer 上也有效。

---

## Fine-grained classification

Datasets:

- CUB-200-2011
- Stanford Cars

MaxSup achieves the best performance.

### Meaning

细粒度分类非常依赖 subtle visual cues，如：

- texture
- part-level details
- fine boundaries

MaxSup 在这里更强，说明它保留了更丰富、更细腻的表示。

---

## Long-tailed classification

Dataset:

- LT CIFAR-10

Compared with:

- Focal Loss
- LS
- MaxSup

### Observation

MaxSup 在 many-shot / medium-shot 上取得更好的 trade-off，overall 表现最好。

### Note

作者也很诚实：它并没有彻底解决 low-shot 类别问题，但整体 direction 是积极的。

---

## Corrupted / OOD classification

Dataset:

- CIFAR-10-C

Metrics:

- NLL
- ECE

### Observation

LS 在某些 NLL 指标上更强，  
但 MaxSup 的 ECE 更低，说明 calibration 更可靠。

---

## Alpha schedule ablation

作者还测试了不同 \(\alpha\) schedule，发现 MaxSup 对 schedule 比较稳。

这说明：

```text
method is robust, not too sensitive to hyperparameters
```

---

# 5.3 Evaluation on Semantic Segmentation

## Setup

- Dataset: ADE20K
- Framework: UPerNet
- Backbone: DeiT-Small pretrained on ImageNet

比较：

- Baseline
- LS
- MaxSup

## Result

mIoU:

- Baseline: 42.1
- LS: 42.4
- **MaxSup: 42.8**

## Meaning

MaxSup 学到的特征不只适合 classification，  
也更适合 dense prediction tasks。

这说明它的 backbone representation 更 transferable。

---

# 5.4 Visualization via Class Activation Maps

作者用 Grad-CAM 比较 LS 和 MaxSup 的关注区域。

## Observation

### LS often attends to background cues

例如：

- 鸟类任务关注树枝
- 船类任务关注水面
- 动物任务关注背景

### MaxSup attends more to the object itself

例如：

- 鸟身
- 船体
- 动物轮廓

## Meaning

MaxSup 的注意力更 focused，  
模型更少依赖 spurious shortcuts，解释性更好。

---

# 6. Section 5 Conclusion

## What the paper claims

1. LS unintentionally amplifies errors on misclassified samples
2. MaxSup fixes this by penalizing the top-1 logit
3. MaxSup improves:
   - classification
   - transferability
   - segmentation
   - interpretability

---

## Limitations

作者提到的限制包括：

- influence on knowledge distillation needs further study
- still a logit-level regularizer, may not be the final answer
- combination with other regularizers remains open

---

## Practical impact

MaxSup 的工程优势：

- simple
- lightweight
- minimal overhead
- easy to integrate into existing training pipelines

---

# 7. 这篇论文的优点 / Strengths

## 1. Problem is real

LS is widely used, so fixing its weakness matters.

## 2. Insight is elegant

只改：

\[
z_{gt}\rightarrow z_{\max}
\]

但理论和实验都很完整。

## 3. Experiments are comprehensive

不仅有：

- top-1 accuracy

还有：

- feature analysis
- transfer
- segmentation
- long-tail
- corrupted data
- Grad-CAM

---

# 8. 这篇论文的不足 / Critique

## 1. Improvement is conceptually small

方法很 elegant，但从“创新规模”看，核心变化只有一个 target replacement。

## 2. More theory could be explored

虽然解释了 LS 的 hidden bad term，但与 margin theory、optimization stability 的关系还可更深入。

## 3. Low-shot issue remains unsolved

在 long-tailed setting 下，对 minority classes 的改善仍有限。

---

# 9. 汇报版 / Presentation Version

## Slide 1 — Problem

Label Smoothing reduces overconfidence, but may amplify wrong predictions when the model is already wrong.

## Slide 2 — Key theorem

\[
L_{LS}=\alpha\left(z_{gt}-\frac1K\sum_k z_k\right)
\]

LS can be decomposed into:

- regularization term
- error amplification term

## Slide 3 — MaxSup

\[
L_{MaxSup}=\alpha\left(z_{\max}-\frac1K\sum_k z_k\right)
\]

Replace:

\[
z_{gt}\to z_{\max}
\]

## Slide 4 — Why it works

- correct prediction: suppress overconfidence
- wrong prediction: suppress wrong dominant class

## Slide 5 — Evidence

MaxSup improves:

- ImageNet classification
- DeiT
- fine-grained classification
- long-tailed classification
- segmentation
- Grad-CAM interpretability

---

# 10. 最终 takeaway / Final Takeaway

```text
The class that should be regularized
is not always the labeled class,
but the class the model currently trusts most.
```

```text
训练时真正该被约束的，
不一定是标签指定的类，
而是模型当前最相信的类。
```
