# Maximum Likelihood Estimation and Gradient Descent

> Machine Learning 课程笔记｜最大似然估计与梯度下降  
> 依据课件：`2a-MLE-gradient-descent.pdf`（33 页，更新于 03/10/2026）。  
> 延续上一节 Logistic Regression：本节回答“如何从训练数据中学习参数”。重复展示页按知识点合并；额外推导与严谨性说明标为“补充”。

## 目录

- [1. 本节主线与符号](#s1)
- [2. Maximum Likelihood Estimation](#s2)
- [3. Likelihood：单个样本与整个训练集](#s3)
- [4. Probability 与 Likelihood](#s4)
- [5. 从 Likelihood 到 Log-Likelihood](#s5)
- [6. Negative Log-Likelihood 与 Loss](#s6)
- [7. Binary Cross-Entropy](#s7)
- [8. 怎样理解 Loss 的大小](#s8)
- [9. Derivative 与 Partial Derivative](#s9)
- [10. Gradient 与 Steepest Descent](#s10)
- [11. Logistic Regression 的梯度推导](#s11)
- [12. Batch Gradient Descent](#s12)
- [13. 一轮参数更新的完整例子](#s13)
- [14. Learning Rate 与 Hyperparameter](#s14)
- [15. Local Minima 与 Convexity](#s15)
- [16. 易错点与公式速查](#s16)
- [17. 复习问答](#s17)
- [18. 课件页码索引](#s18)

<a id="s1"></a>
## 1. 本节主线与符号

**对应课件：第 1–4、18、33 页。**

上一节确定了模型形式，但尚未确定参数：

$$
p(y=1\mid\mathbf x,\mathbf w)=\sigma(z),\qquad
z=w_0+\sum_{j=1}^{d}w_jx_j,\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

本节的逻辑是：**选择最能解释已观察标签的参数 → 最大化 Likelihood → 最大化 Log-Likelihood → 最小化 Cross-Entropy Loss → 用 Gradient Descent 迭代更新参数。**

| 符号 | 含义 |
| --- | --- |
| $N$ | 训练样本数量 |
| $d$ | 原始输入特征数量 |
| $\mathcal T$ | 训练集 $\{(\mathbf x^{(i)},y^{(i)})\}_{i=1}^N$ |
| $y^{(i)}\in\{0,1\}$ | 第 $i$ 个样本的真实标签 |
| $\tilde{\mathbf x}^{(i)}=(1,x_1^{(i)},\ldots,x_d^{(i)})^T$ | 加入 bias 特征 $x_0=1$ 后的输入 |
| $\mathbf w=(w_0,w_1,\ldots,w_d)^T$ | 待学习的 $d+1$ 个参数 |
| $z_i=\mathbf w^T\tilde{\mathbf x}^{(i)}$ | 第 $i$ 个样本的线性得分 |
| $q_i=\sigma(z_i)$ | 模型预测该样本属于 class 1 的概率 |
| $\mathcal L(\mathbf w)$ | Likelihood，作为参数的函数 |
| $\ell(\mathbf w)=\ln\mathcal L(\mathbf w)$ | Log-Likelihood |
| $E(\mathbf w)$ | 本课使用的总 Loss，即 Negative Log-Likelihood |
| $\nabla E(\mathbf w)$ | Loss 对全部参数的 Gradient |
| $\eta>0$ | Learning Rate |

本笔记使用 $q_i$ 简写课件中的 $p(1\mid\mathbf x^{(i)},\mathbf w)$。课件加入 $x_0=1$ 后仍写 $\mathbf x$；这里用 $\tilde{\mathbf x}$ 明确表示增广输入。

**区分：** $E(\mathbf w)$ 在本课是整个训练集上的总 Loss；上一节 $E(g(\mathbf x),y)$ 表示单次预测的 error function。符号相似，参数与语境不同。

<a id="s2"></a>
## 2. Maximum Likelihood Estimation

**对应课件：第 4、8 页。**

**Maximum Likelihood Estimation（MLE，最大似然估计）**：将训练数据视为已知，选择使这些观察结果的 Likelihood 最大的参数。

在 Logistic Regression 中，我们已经知道输入和正确标签。对不同候选 $\mathbf w$，模型给这些正确标签分配的概率不同。MLE 选择使整组真实标签最可能出现的参数。

$$
\boxed{\hat{\mathbf w}_{\mathrm{MLE}}=\arg\max_{\mathbf w}\mathcal L(\mathbf w)}.
$$

| 表达式 | 返回的是什么 |
| --- | --- |
| $\max_{\mathbf w}\mathcal L(\mathbf w)$ | 最大的 Likelihood 数值（若最大值存在） |
| $\arg\max_{\mathbf w}\mathcal L(\mathbf w)$ | 取得最大值的参数（若存在，可能不唯一） |

**MLE 给出“应该优化什么”的原则；Gradient Descent 给出“怎样迭代求解”的方法。两者不是同一种概念。**

<a id="s3"></a>
## 3. Likelihood：单个样本与整个训练集

**对应课件：第 5–8 页。**

### 3.1 单个样本的 Conditional Likelihood

对样本 $(\mathbf x^{(i)},y^{(i)})$，需要的是模型给**实际观察到的标签**的概率：

$$
p_{y^{(i)}}=p(y^{(i)}\mid\mathbf x^{(i)},\mathbf w).
$$

| 真实标签 | 应采用的概率 |
| --- | --- |
| $y^{(i)}=1$ | $q_i=p(1\mid\mathbf x^{(i)},\mathbf w)$ |
| $y^{(i)}=0$ | $1-q_i=p(0\mid\mathbf x^{(i)},\mathbf w)$ |

**不能不看真实标签，一律使用 $q_i$。** 例如真实标签是 0，而模型预测 $q_i=0.9$，正确标签的概率只有 $1-q_i=0.1$。

用 Bernoulli 的统一写法：

$$
p(y^{(i)}\mid\mathbf x^{(i)},\mathbf w)
=q_i^{y^{(i)}}(1-q_i)^{1-y^{(i)}}.
$$

当 $y=1$，第二个因子的指数为 0，结果为 $q$；当 $y=0$，第一个因子的指数为 0，结果为 $1-q$。

### 3.2 Joint Conditional Likelihood

在本课的 i.i.d. 数据与逐样本条件模型设定下，整个训练集标签的 Conditional Likelihood 是这些概率的乘积：

$$
\boxed{\mathcal L(\mathbf w)
=p(\mathbf y\mid X,\mathbf w)
=\prod_{i=1}^{N}p(y^{(i)}\mid\mathbf x^{(i)},\mathbf w)
=\prod_{i=1}^{N}q_i^{y^{(i)}}(1-q_i)^{1-y^{(i)}}}.
$$

这里将输入 $X$ 视为已给定，优化的是标签的条件概率；不是要同时学习 $p(\mathbf x)$。

### 3.3 三个样本的比较例子

**补充例子：** 真实标签为 $(1,0,1)$。

| 模型 | 三个样本的 class 1 概率 | 真实标签的概率 | Likelihood |
| --- | --- | --- | --- |
| A | $(0.8,0.2,0.7)$ | $(0.8,0.8,0.7)$ | $0.8\times0.8\times0.7=0.448$ |
| B | $(0.6,0.4,0.6)$ | $(0.6,0.6,0.6)$ | $0.6^3=0.216$ |

MLE 在这两个候选模型之间偏向 A，因为 A 对整组观察标签给出的 Likelihood 更高。两者在阈值 0.5 下都分类正确，但给正确标签的概率不同。

<a id="s4"></a>
## 4. Probability 与 Likelihood

**对应课件：第 5–6 页。**

它们可以使用同一个概率表达式，但**固定什么、研究什么**不同。

| 视角 | 固定对象 | 研究对象 | 问题 |
| --- | --- | --- | --- |
| Probability | 模型参数 $\mathbf w$ 与输入 | 可能出现的标签 | 采用这组参数，会出现什么输出？ |
| Likelihood | 已观察的数据 | 候选参数 $\mathbf w$ | 哪组参数更能解释这些数据？ |

课件将 Probability 描述为采用模型参数预测结果，将 Likelihood 描述为判断参数解释观察数据的好坏。更精确的区别是：**Likelihood 是固定数据后，作为参数函数来看待的概率表达式。**

**补充：** $\mathcal L(\mathbf w)$ 不是“参数为 $\mathbf w$ 的概率”。MLE 没有直接计算 $p(\mathbf w\mid X,\mathbf y)$；也不要求 Likelihood 对所有参数积分或求和为 1。

<a id="s5"></a>
## 5. 从 Likelihood 到 Log-Likelihood

**对应课件：第 9 页。**

### 5.1 为什么取 logarithm

很多小于 1 的概率相乘，会得到非常小的数。例如 $0.1^{1000}=10^{-1000}$，直接计算可能出现 **Underflow**。乘积在求导时也较繁琐。

对乘积取自然对数，用 $\ln(ab)=\ln a+\ln b$ 将乘法转成加法：

$$
\ell(\mathbf w)=\ln\mathcal L(\mathbf w)
=\ln\left(\prod_{i=1}^{N}p_{y^{(i)}}\right)
=\sum_{i=1}^{N}\ln p_{y^{(i)}}.
$$

这方便计算与求导，并避免显式计算整组极小的概率乘积。

### 5.2 为什么最优参数不变

$\ln$ 在正数范围上严格单调递增：$a>b>0\Rightarrow\ln a>\ln b$。它保留了候选 Likelihood 的大小顺序，因此：

$$
\boxed{\arg\max_{\mathbf w}\mathcal L(\mathbf w)
=\arg\max_{\mathbf w}\ln\mathcal L(\mathbf w)}.
$$

这是**相同的最优参数**，不是两个函数的数值相同。上一节例子中，$\ln0.448\approx-0.803$，$\ln0.216\approx-1.532$；A 的 Log-Likelihood 仍然更大。

**补充：** 应直接累加 log probabilities，而不是先算完乘积再取 log；后者可能在取 log 之前已经 Underflow。

<a id="s6"></a>
## 6. Negative Log-Likelihood 与 Loss

**对应课件：第 10–15 页。**

为使用最小化形式，定义：

$$
\boxed{E(\mathbf w)=-\ell(\mathbf w)
=-\ln\mathcal L(\mathbf w)
=-\sum_{i=1}^{N}\ln p_{y^{(i)}}}.
$$

于是：

$$
\arg\max_{\mathbf w}\mathcal L(\mathbf w)
=\arg\max_{\mathbf w}\ell(\mathbf w)
=\arg\min_{\mathbf w}E(\mathbf w).
$$

| 函数 | 希望优化的方向 |
| --- | --- |
| Likelihood $\mathcal L$ | 越大越好 |
| Log-Likelihood $\ell$ | 越大越好 |
| Negative Log-Likelihood $E$ | 越小越好 |

单个样本的 loss 为 $-\ln p_{y^{(i)}}$：真实标签的预测概率越接近 1，loss 越接近 0；越接近 0，loss 越大。

在二分类且有限得分时，概率严格在 $(0,1)$ 中，因此每项 loss 为正，总 loss 非负。理想情况下每个正确标签概率趋近 1，总 loss 趋近 0。

<a id="s7"></a>
## 7. Binary Cross-Entropy

**对应课件：第 16–18 页。**

### 7.1 由 Bernoulli Likelihood 得到 Cross-Entropy

### 定义
```
Cross-entropy is a measure of discrepancy between two probability distributions.
它用来衡量`真实`（目标）分布 p(y | x) 与`基于训练样本`估计的学习分布 p(y | x, w) 之间的差异。
```
在逻辑回归中，找到能min Cross-entropy的parameter，相当于找到parameters that best explain the observed data

将统一概率形式代入 Negative Log-Likelihood：

$$
\ln p(y^{(i)}\mid\mathbf x^{(i)},\mathbf w)
=y^{(i)}\ln q_i+(1-y^{(i)})\ln(1-q_i).
$$

所以 Logistic Regression 的总 **Binary Cross-Entropy Loss（BCE）** 为：

$$
\boxed{E(\mathbf w)
=-\sum_{i=1}^{N}\left[y^{(i)}\ln q_i+(1-y^{(i)})\ln(1-q_i)\right]}.
$$

单个样本：

$$
E_i=-\left[y^{(i)}\ln q_i+(1-y^{(i)})\ln(1-q_i)\right].
$$

| 真实标签 | 单样本 loss |
| --- | --- |
| $y=1$ | $-\ln q$ |
| $y=0$ | $-\ln(1-q)$ |

括号中的两个部分由真实标签选择，并不是每个样本同时加上完整的 $-\ln q$ 与 $-\ln(1-q)$。

### 7.2 为什么叫 Cross-Entropy

课件将 Cross-Entropy 描述为衡量两个概率分布的不相似程度。这里预测分布为 $(1-q_i,q_i)$，观察标签对应的分布为 $(1-y^{(i)},y^{(i)})$。

**补充理解：** 每个样本的标签提供一次观察；训练集上的 Cross-Entropy 是对真实条件分布与预测分布之间 expected cross-entropy 的经验估计。它本身不是对称距离，不要把“dissimilarity”理解成严格的距离函数。

最小化这项 BCE，在本课未加入额外正则项的设定下，就是最大化观察标签的 Likelihood。

### 7.3 Sum Loss 与 Mean Loss

课件使用求和；实现中也经常使用平均：

$$
\bar E(\mathbf w)=\frac1N E(\mathbf w).
$$

对于固定训练集，乘以正数 $1/N$ 不改变最优参数，但会改变数值和梯度：

$$
\nabla\bar E(\mathbf w)=\frac1N\nabla E(\mathbf w).
$$

**补充：** 在相同 learning rate 下，两种写法的一步更新幅度不同。使用公式时必须让 loss、gradient 与 learning rate 的约定保持一致。

<a id="s8"></a>
## 8. 怎样理解 Loss 的大小

**对应课件：第 12–15 页。**

### 8.1 真实标签为 1

| $q=p(y=1\mid\mathbf x,\mathbf w)$ | $-\ln q$ | 解释 |
| --- | --- | --- |
| 0.99 | $\approx0.010$ | 给正确标签很高的概率，惩罚很小 |
| 0.5 | $\approx0.693$ | 两类概率相同 |
| 0.01 | $\approx4.605$ | 给正确标签很低的概率，惩罚很大 |

### 8.2 真实标签为 0

| $q$ | 正确标签概率 $1-q$ | $-\ln(1-q)$ |
| --- | --- | --- |
| 0.01 | 0.99 | $\approx0.010$ |
| 0.5 | 0.5 | $\approx0.693$ |
| 0.99 | 0.01 | $\approx4.605$ |

**Cross-Entropy 特别惩罚“对错误类别非常自信”。** 正确标签的概率趋近 0 时，loss 趋近 $+\infty$。

**补充：Accuracy 与 Loss 不同。** 真实标签为 1 时，$q=0.51$ 和 $q=0.99$ 都预测 class 1，但 loss 分别约为 $0.673$ 与 $0.010$。同样的分类准确率，可以对应不同的概率质量。

<a id="s9"></a>
## 9. Derivative 与 Partial Derivative

**对应课件：第 21–23 页。**

### 9.1 Derivative：一维变化率

$df/dx$ 描述 $x$ 增加时 $f(x)$ 在当前位置的变化率。

| 导数 | 如果希望降低 $f$，局部应怎样移动 |
| --- | --- |
| $df/dx>0$ | 减小 $x$ |
| $df/dx<0$ | 增大 $x$ |
| $df/dx=0$ | 一阶变化率为 0，是 stationary point 的条件 |

统一更新规则：

$$
x^{(t+1)}=x^{(t)}-\eta\frac{df}{dx}\bigg|_{x=x^{(t)}},\qquad\eta>0.
$$

**补充例子：** $f(x)=x^2$，$f'(x)=2x$。当 $x=3,\eta=0.1$，新值为 $3-0.1\times6=2.4$；当 $x=-3$，新值为 $-3-0.1\times(-6)=-2.4$。两个位置都向最小点 0 移动。

### 9.2 Partial Derivative：多维变化率

对于 $E(w_0,\ldots,w_d)$，$\partial E/\partial w_j$ 表示固定其他参数时，改变 $w_j$ 带来的局部变化率。

$$
w_j^{(t+1)}=w_j^{(t)}-\eta\frac{\partial E}{\partial w_j}\bigg|_{\mathbf w=\mathbf w^{(t)}}.
$$

**注意：求导对象是参数 $w_j$，不是输入特征 $x_j$。** 学习过程中训练数据固定，调整的是模型参数。

导数为 0 不一定意味着最小值；一般函数也可能在最大值或 Saddle Point 处梯度为 0。课件的碗形示意图只帮助理解方向，不代表 Logistic Regression 的 loss 是 quadratic function。

<a id="s10"></a>
## 10. Gradient 与 Steepest Descent

**对应课件：第 20、24、27–29 页。**

将全部 Partial Derivatives 放入向量：

$$
\nabla E(\mathbf w)=
\begin{bmatrix}
\partial E/\partial w_0\\
\partial E/\partial w_1\\
\vdots\\
\partial E/\partial w_d
\end{bmatrix}.
$$

Gradient 在通常的 Euclidean 几何下指向**局部最陡上升方向**，因此 $-\nabla E$ 是局部最陡下降方向。

$$
\boxed{\mathbf w^{(t+1)}=\mathbf w^{(t)}-\eta\nabla E(\mathbf w^{(t)})}.
$$

**补充解释：** 对足够小的变化 $\Delta\mathbf w$，一阶近似为：

$$
E(\mathbf w+\Delta\mathbf w)
\approx E(\mathbf w)+\nabla E(\mathbf w)^T\Delta\mathbf w.
$$

令 $\Delta\mathbf w=-\eta\nabla E$，则近似变化为：

$$
-\eta\Vert\nabla E\Vert_2^2\leq0.
$$

这说明为什么减去 gradient 能局部降低 loss。**“Steepest”说的是当前位置、固定小步长下的方向，不表示任意大的更新都保证降低 loss。**

课件三维图的水平坐标是参数 $w_0,w_1$，竖直坐标是 $E(\mathbf w)$；它展示的是 Parameter Space 中的优化，而不是 Input Space 中的分类边界。

<a id="s11"></a>
## 11. Logistic Regression 的梯度推导

**对应课件：第 26 页给出结果；以下链式求导为补充。**

### 11.1 Sigmoid 的导数

$$
q=\sigma(z)=\frac1{1+e^{-z}},\qquad
\frac{dq}{dz}=\frac{e^{-z}}{(1+e^{-z})^2}=q(1-q).
$$

### 11.2 单样本 loss 对概率求导

为简洁省略样本上标：

$$
E_i=-[y\ln q+(1-y)\ln(1-q)].
$$

$$
\frac{\partial E_i}{\partial q}
=-\frac yq+\frac{1-y}{1-q}
=\frac{q-y}{q(1-q)}.
$$

第二项是正号，因为对 $-\ln(1-q)$ 求导时，内层 $1-q$ 还会贡献一个负号。

### 11.3 对 score 求导

利用 Chain Rule：

$$
\frac{\partial E_i}{\partial z}
=\frac{\partial E_i}{\partial q}\frac{dq}{dz}
=\frac{q-y}{q(1-q)}\,q(1-q)
=\boxed{q-y}.
$$

### 11.4 对参数求导

因为 $z=\mathbf w^T\tilde{\mathbf x}$，有 $\partial z/\partial w_j=\tilde x_j$。因此：

$$
\frac{\partial E_i}{\partial w_j}=(q_i-y^{(i)})\tilde x_j^{(i)}.
$$

对全部样本求和：

$$
\boxed{\frac{\partial E}{\partial w_j}
=\sum_{i=1}^{N}(q_i-y^{(i)})\tilde x_j^{(i)}}.
$$

向量形式：

$$
\boxed{\nabla E(\mathbf w)
=\sum_{i=1}^{N}(q_i-y^{(i)})\tilde{\mathbf x}^{(i)}}.
$$

**记忆：Gradient = 预测概率与真实标签之差 × 输入，再对样本求和。** 这里使用概率 $q_i$，不是经过 0.5 阈值后的类别 $\hat y^{(i)}$。

### 11.5 Bias 的梯度

因为 $\tilde x_0^{(i)}=1$：

$$
\frac{\partial E}{\partial w_0}=\sum_{i=1}^{N}(q_i-y^{(i)}).
$$

### 11.6 Matrix Form 与维度

**补充：** 令 $\tilde X$ 为加入全 1 列的 Design Matrix，$\mathbf q=\sigma(\tilde X\mathbf w)$，Sigmoid 逐元素作用。

$$
\boxed{\nabla E=\tilde X^T(\mathbf q-\mathbf y)}.
$$

| 对象 | 形状 |
| --- | --- |
| $\tilde X$ | $N\times(d+1)$ |
| $\mathbf w$ | $(d+1)\times1$ |
| $\mathbf q,\mathbf y$ | $N\times1$ |
| $\tilde X^T(\mathbf q-\mathbf y)$ | $(d+1)\times1$ |

所得 gradient 必须与 $\mathbf w$ 形状相同，才能相减。

<a id="s12"></a>
## 12. Batch Gradient Descent

**对应课件：第 25–26 页。**

**Batch Version**：每次更新使用整个训练集的 gradient。

1. 将 $\mathbf w$ 初始化为全 0，或接近 0 的随机数。
2. 用当前参数为所有训练样本计算 $q_i$。
3. 求所有样本的 gradient contribution 并相加。
4. 更新 $\mathbf w\leftarrow\mathbf w-\eta\nabla E(\mathbf w)$。
5. 重复上述过程，直到满足停止条件。

课件停止条件是达到给定 iteration 数，或 gradient 成为零向量。**补充：实际数值计算通常用 gradient norm 小于某个 tolerance、loss 变化足够小或迭代上限，不要求浮点数恰好等于 0。**

```text
输入：增广训练矩阵 X_tilde，标签 y，learning rate eta，迭代上限 T
初始化：w = 全零向量

重复最多 T 次：
    z = X_tilde @ w
    q = sigmoid(z)
    gradient = X_tilde.T @ (q - y)    # 本课使用 sum loss
    如果 gradient 的范数小于 tolerance：停止
    w = w - eta * gradient

输出：w
```

所有参数应使用**同一组旧参数**算出的 gradient 同时更新。若在算其他参数的 gradient 前已经更改某个权重，就不是本课给出的标准 vector update。

初始化为全 0 时，每个样本 $z_i=0$，因此 $q_i=0.5$。Logistic Regression 可以这样初始化；不应把其他模型中关于初始化的要求机械套入本课。

<a id="s13"></a>
## 13. 一轮参数更新的完整例子

**以下为补充练习，使用课件的 sum loss。**

只有一个原始特征，两个训练样本：

$$
(x^{(1)},y^{(1)})=(0,0),\qquad(x^{(2)},y^{(2)})=(1,1).
$$

增广数据与初始化：

$$
\tilde X=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad
\mathbf y=\begin{bmatrix}0\\1\end{bmatrix},\quad
\mathbf w^{(0)}=\begin{bmatrix}0\\0\end{bmatrix},\quad\eta=0.1.
$$

### 13.1 预测与初始 loss

两个样本的得分都为 0，因此 $q_1=q_2=0.5$：

$$
E(\mathbf w^{(0)})=-\ln(1-0.5)-\ln0.5=2\ln2\approx1.3863.
$$

### 13.2 计算 gradient

第一个样本贡献：

$$
(q_1-y^{(1)})\tilde{\mathbf x}^{(1)}
=(0.5-0)\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0.5\\0\end{bmatrix}.
$$

第二个样本贡献：

$$
(q_2-y^{(2)})\tilde{\mathbf x}^{(2)}
=(0.5-1)\begin{bmatrix}1\\1\end{bmatrix}
=\begin{bmatrix}-0.5\\-0.5\end{bmatrix}.
$$

求和：

$$
\nabla E(\mathbf w^{(0)})=\begin{bmatrix}0\\-0.5\end{bmatrix}.
$$

### 13.3 更新参数并重新评估

$$
\mathbf w^{(1)}
=\begin{bmatrix}0\\0\end{bmatrix}
-0.1\begin{bmatrix}0\\-0.5\end{bmatrix}
=\begin{bmatrix}0\\0.05\end{bmatrix}.
$$

| 样本 | 新 score | 新 class 1 概率 | 真实标签 |
| --- | --- | --- | --- |
| $x=0$ | 0 | 0.5 | 0 |
| $x=1$ | 0.05 | $\approx0.5125$ | 1 |

新 loss：

$$
E(\mathbf w^{(1)})=-\ln0.5-\ln\sigma(0.05)\approx1.3616.
$$

Loss 从 $1.3863$ 降到 $1.3616$。第一轮 bias 的 gradient 为 0 是因为两项恰好抵消，不意味着 bias 永远不会更新。

**注意：** $x=0$ 的样本仍在 0.5 边界上；按本课相等时预测 1 的规则，它此时分类错误。Gradient Descent 降低的是连续的 Cross-Entropy，单轮更新不保证立即提高 Accuracy。

<a id="s14"></a>
## 14. Learning Rate 与 Hyperparameter

**对应课件：第 30–31 页。**

$\eta$ 控制沿负 gradient 方向的步长：

$$
\Delta\mathbf w=-\eta\nabla E.
$$

| Learning Rate | 可能发生什么 |
| --- | --- |
| 太大 | 越过最优区域，震荡、loss 增加，甚至不收敛 |
| 太小 | 每次移动很少，需要很多轮才能收敛 |
| 合适 | 在保持稳定的同时逐步降低 loss |

**Parameters** 是通过训练更新的 $\mathbf w$；**Hyperparameters** 是控制训练过程的设置，如本课的 learning rate 与 iteration 上限。$\eta$ 不是输入特征的权重。

**补充例子：** 对 $f(w)=w^2$，更新为 $w^{(t+1)}=(1-2\eta)w^{(t)}$：

| $\eta$ | 更新行为 |
| --- | --- |
| 0.1 | $w$ 每轮变成原来的 0.8 倍，逐步靠近 0 |
| 0.75 | $w$ 每轮变成原来的 $-0.5$ 倍，来回但逐步靠近 0 |
| 1 | $w$ 每轮反号，非零初始值不会靠近 0 |
| 1.1 | 振幅扩大，发散 |

这个例子展示步长的作用，**不是 Logistic Regression 的通用 learning rate 推荐值**；可用步长与数据尺度、loss 定义及曲率有关。

<a id="s15"></a>
## 15. Local Minima 与 Convexity

**对应课件：第 32 页。**

### 15.1 一般函数的 Local Minimum

**Local Minimum**：在附近区域内没有更低的点；**Global Minimum**：在整个参数域内没有更低的点。对 Non-convex Function，一般 Gradient Descent 可能停在局部最小点或其他 stationary point，结果可能依赖初始化。

### 15.2 Logistic Regression 的核心结论

标准 Logistic Regression 的 Cross-Entropy 对参数是 **Convex**，因此没有“比全局最小值更差的局部最小值”。这使它比一般 Non-convex Optimisation 更容易分析。

> **课件严谨性说明：** 第 32 页写为 “strictly convex … single unique minimum”。这个结论需要额外条件，不应理解成任何数据集都有唯一且有限的最优参数。一般保证是 convex；strict convexity 与有限最小值的存在需要分别讨论。

**补充：** 其 Hessian 为：

$$
\nabla^2E(\mathbf w)=\tilde X^TD\tilde X,\qquad
D=\operatorname{diag}(q_i(1-q_i)).
$$

有限参数下 $D$ 的对角元素为正，Hessian 为 Positive Semidefinite，因此 loss 是 convex。若增广矩阵 $\tilde X$ 满列秩，Hessian 为 Positive Definite，loss 才具有对应的 strict convexity。

仍需区分两个问题：

- **参数是否唯一：** 重复或线性相关的特征可能使多组权重给出同样预测，最优参数不一定唯一。
- **有限最小值是否存在：** 完全线性可分数据下，未正则化 Logistic Regression 可以不断放大某组参数，让正确标签概率趋近 1、loss 趋近 0，却没有有限参数真正达到 0。因此有限 MLE 可能不存在。

第 13 节的两个样本就是可分数据的简单例子。训练前期仍可通过有限更新降低 loss，但不能据此推断最后一定得到唯一、有限的 optimum。

**复习要点：Convex Loss 消除了较差的 Local Minima 问题；它不自动保证任意 learning rate 都收敛，也不自动保证唯一、有限的最优参数。**

<a id="s16"></a>
## 16. 易错点与公式速查

### 16.1 易错点

| 易错说法或操作 | 正确理解 |
| --- | --- |
| Likelihood 是参数为某值的概率 | 它是固定观察数据后，关于参数的函数 |
| 所有样本都把 $q_i$ 相乘 | 标签为 1 用 $q_i$，标签为 0 用 $1-q_i$ |
| 取 log 改变了最优参数 | 严格单调递增的 log 保留最优参数 |
| 最小化 Log-Likelihood | 最大化 Log-Likelihood；最小化其负值 |
| 正确标签概率越小，loss 越小 | 越小越应惩罚，$-\ln p$ 越大 |
| 用分类结果 $\hat y$ 计算本课 gradient | 使用预测概率 $q_i$ |
| Gradient 指向下降方向 | Gradient 指向局部最陡上升，负 gradient 才下降 |
| 更新 $\mathbf w\leftarrow\mathbf w+\eta\nabla E$ | 最小化 loss 应减去 gradient |
| 每更新一个参数，再用新值算下一个 gradient | 一轮所有 gradient 都基于同一组旧参数 |
| 忘记 bias | 使用增广输入，bias 特征恒为 1 |
| 混用 sum loss 与 mean gradient | 两者必须匹配；mean gradient 额外除以 $N$ |
| Learning Rate 越大越好 | 太大可能震荡或发散 |
| 导数等于 0 总是最小值 | 一般函数中也可能是最大值或 Saddle Point |
| 课件碗形图说明 BCE 是二次函数 | 图只是方向示意，Logistic BCE 不是 quadratic function |
| Loss 降低一定表示 Accuracy 提升 | 概率可以改善而类别不变 |

### 16.2 核心公式

| 名称 | 公式 |
| --- | --- |
| 模型概率 | $q_i=\sigma(\mathbf w^T\tilde{\mathbf x}^{(i)})$ |
| 单样本 Likelihood | $q_i^{y^{(i)}}(1-q_i)^{1-y^{(i)}}$ |
| 训练集 Likelihood | $\mathcal L=\prod_iq_i^{y^{(i)}}(1-q_i)^{1-y^{(i)}}$ |
| Log-Likelihood | $\ell=\sum_i[y^{(i)}\ln q_i+(1-y^{(i)})\ln(1-q_i)]$ |
| 总 Cross-Entropy | $E=-\ell$ |
| 优化目标 | $\hat{\mathbf w}=\arg\min_{\mathbf w}E(\mathbf w)$，若最小值存在 |
| Sigmoid 导数 | $\sigma'(z)=\sigma(z)(1-\sigma(z))$ |
| 单样本 score 导数 | $\partial E_i/\partial z_i=q_i-y^{(i)}$ |
| 总梯度 | $\nabla E=\sum_i(q_i-y^{(i)})\tilde{\mathbf x}^{(i)}$ |
| Matrix Gradient | $\nabla E=\tilde X^T(\mathbf q-\mathbf y)$ |
| Bias Gradient | $\partial E/\partial w_0=\sum_i(q_i-y^{(i)})$ |
| Gradient Descent | $\mathbf w^{(t+1)}=\mathbf w^{(t)}-\eta\nabla E(\mathbf w^{(t)})$ |
| Mean Loss Gradient | $\nabla\bar E=\nabla E/N$ |

<a id="s17"></a>
## 17. 复习问答

### 17.1 What is maximum likelihood estimation?

Maximum likelihood estimation selects the parameters that maximise the likelihood of the observed training data.

中文：选择最能解释已观察训练数据的参数。在本课中，输入固定，最大化的是观察标签的条件 Likelihood。

### 17.2 Why maximise Log-Likelihood instead of Likelihood?

The logarithm is strictly increasing, so it preserves the maximiser. It also converts products into sums, avoiding the explicit product of many small probabilities and simplifying differentiation.

中文：最优参数不变，同时把连乘变成求和，便于计算与求导。

### 17.3 How is MLE related to Cross-Entropy?

For binary Logistic Regression, the negative conditional Log-Likelihood is the sum of Binary Cross-Entropy losses. Therefore, maximising the likelihood is equivalent to minimising this loss.

中文：本课 BCE 就是 Negative Log-Likelihood；一个最大化，一个最小化，优化方向相反但目标一致。

### 17.4 Why does Gradient Descent subtract the gradient?

The gradient points in the direction of steepest local increase. Its negative points in the direction of steepest local decrease under the Euclidean norm.

中文：希望下降，因此沿负 gradient 方向更新；步长仍需合适。

### 17.5 What is Batch Gradient Descent?

Batch Gradient Descent computes the gradient using the entire training set before each parameter update.

中文：每一步用全部训练样本计算 gradient，再统一更新参数。

### 17.6 What happens if the learning rate is too large or too small?

A large learning rate may cause overshooting, oscillation or divergence. A small learning rate may require many iterations to converge.

中文：太大可能不稳定，太小可能收敛缓慢。

### 17.7 Are poor local minima a problem for standard Logistic Regression?

Its Cross-Entropy loss is convex in the parameters, so a local minimum, if attained, is global. A unique finite optimum requires further conditions.

中文：标准 BCE 是 convex，没有较差的局部最小点；唯一性和有限最优解的存在仍需额外条件。

<a id="s18"></a>
## 18. 课件页码索引

| 课件页码 | 内容 | 笔记位置 |
| --- | --- | --- |
| 1–3 | 标题、学习目标 | [第 1 节](#s1) |
| 4 | 如何学习权重、MLE 原则 | [第 2 节](#s2) |
| 5–6 | 单样本与整体 Likelihood、Probability vs Likelihood | [第 3–4 节](#s3) |
| 7–8 | 记号、MLE 优化问题 | [第 2–3 节](#s2) |
| 9 | Log-Likelihood | [第 5 节](#s5) |
| 10–11 | Negative Log-Likelihood 与 Loss | [第 6 节](#s6) |
| 12–15 | 正确标签概率与 loss 大小 | [第 8 节](#s8) |
| 16–17 | Cross-Entropy 与 MLE 等价 | [第 7 节](#s7) |
| 18–19 | 优化问题总结、进入 Gradient Descent | [第 1、7 节](#s1) |
| 20 | Gradient Descent 的基本思想 | [第 10 节](#s10) |
| 21–23 | Derivatives、Partial Derivatives | [第 9 节](#s9) |
| 24 | 向量更新与 Gradient | [第 10 节](#s10) |
| 25 | Batch Gradient Descent | [第 12 节](#s12) |
| 26 | Logistic Regression 的 gradient | [第 11–12 节](#s11) |
| 27–29 | Steepest Descent 图示 | [第 10 节](#s10) |
| 30–31 | Learning Rate 的作用 | [第 14 节](#s14) |
| 32 | Local Minima、Convexity | [第 15 节](#s15) |
| 33 | 本课总结与下一节问题 | [第 1、16 节](#s1) |

**本课范围：** 用 MLE 建立 Logistic Regression 的优化目标，再用 Batch Gradient Descent 求解。其他优化算法的比较属于后续内容。
