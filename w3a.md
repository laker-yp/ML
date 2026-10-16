## 目录

- [1. Logistic Regression 与泛化](#sec-1)
- [2. Logistic Regression 的 Overfitting](#sec-2)
- [3. L2 Regularisation / Weight Decay](#sec-3)
- [4. Underfitting 与非线性变换的必要性](#sec-4)
- [5. 如何构造 Polynomial Features](#sec-5)
- [6. 课件重点例题：为什么有两个决策边界？](#sec-6)
- [7. 在 Logistic Regression 中应用非线性变换](#sec-7)
- [8. 变换后还是 Linear Model 吗？](#sec-8)
- [9. Nonlinear Transformations 的风险](#sec-9)
- [10. 考前重点公式](#sec-10)

---

<a id="sec-1"></a>

# 1. Logistic Regression 与泛化（PPT 4–5）

<a id="sec-1-1"></a>

## 1.1 Logistic Regression 的预测（PPT 4）

输入向量为 $\mathbf{x}$，权重向量为 $\mathbf{w}$，通常令 $x_0=1$ 来表示常数项（bias）。

**Score：**

$$
z=\mathbf{w}^T\mathbf{x}
$$

**Class 1 的预测概率：**

$$
p_1=\frac{e^z}{1+e^z}=\frac{1}{1+e^{-z}}
$$

**Log-odds：**

$$
\ln\frac{p_1}{1-p_1}=\mathbf{w}^T\mathbf{x}
$$

- $z\geq 0$：预测 Class 1；$z<0$：预测 Class 0。
- **Decision Boundary（决策边界）**：$\mathbf{w}^T\mathbf{x}=0$。
- $p_0=1-p_1$。

<a id="sec-1-2"></a>

## 1.2 Cross-Entropy Loss（PPT 4）

对于 $N$ 个训练样本：

$$
E(\mathbf{w})=-\sum_{i=1}^{N}\left[y^{(i)}\ln p_1^{(i)}+(1-y^{(i)})\ln(1-p_1^{(i)})\right]
$$

其中 $p_1^{(i)}$ 是模型对第 $i$ 个训练样本预测为 Class 1 的概率。

- 真实标签是 $1$：损失为 $-\ln p_1^{(i)}$，模型希望 $p_1^{(i)}$ 趋近 $1$。
- 真实标签是 $0$：损失为 $-\ln(1-p_1^{(i)})$，模型希望 $p_1^{(i)}$ 趋近 $0$。
- **Cross-Entropy 在此页使用求和形式**，并且是在训练集上计算的。

<a id="sec-1-3"></a>

## 1.3 Training Error 与 Generalisation Error（PPT 5）

**Training Error / Empirical Risk（训练误差 / 经验风险）：**

$$
\hat{E}(\mathbf{w})=\frac{1}{N}\sum_{i=1}^{N}F(g(\mathbf{x}^{(i)}),y^{(i)})
$$

- $g$：预测函数；$F$：衡量预测与标签差异的误差函数。
- **用 Training Set 计算，不是 Test Set。**

**Generalisation Error / Population Risk（泛化误差 / 总体风险）：**

$$
E(\mathbf{w})=\mathrm{E}_{(\mathbf{x},y)\sim p(\mathbf{x},y)}[F(g(\mathbf{x}),y)]
$$

- 表示在真实数据分布上的**期望误差**。
- 泛化误差越小，模型对未见数据的预测通常越好。

**Generalisation Gap（泛化差距）：**

$$
G=E(\mathbf{w})-\hat{E}(\mathbf{w})
$$

训练误差小、泛化误差大时，$G$ 可能很大。

> 注意：PPT 第 4 页的 $E(\mathbf{w})$ 是训练集 Cross-Entropy 总和；第 5 页复用了 $E(\mathbf{w})$ 表示总体风险。**应按所在公式区分，不要把两者当作同一个量。**

**Inductive Bias（归纳偏置）**：学习算法为从有限样本推断未见数据而采用的假设。Logistic Regression 的重要假设是：**在当前特征空间中，决策边界是线性的。**

---

<a id="sec-2"></a>

# 2. Logistic Regression 的 Overfitting（PPT 6–10、13）

<a id="sec-2-1"></a>

## 2.1 为什么线性模型也会过拟合？（PPT 6）

**Linearly Separable（线性可分）**：存在一条直线（二维）或超平面（高维），能把训练数据中的两类完全分开。

即使已经正确分类所有训练样本，Cross-Entropy 仍可能继续下降：

- 对 Class 1 样本，模型希望预测概率无限接近 $1$。
- 对 Class 0 样本，模型希望预测概率无限接近 $0$。
- 模型可能通过放大权重，使概率变得越来越极端，而不需要改变分类边界。

因此，**训练集完全分对，不代表泛化一定好**。

<a id="sec-2-2"></a>

## 2.2 增大权重的效果（PPT 7–10）

课件依次展示：

| PPT | 权重 $\mathbf{w}$ | 预测概率的变化 |
|---|---|---|
| 7 | $(0.5,0.5,0.5)^T$ | Sigmoid 比较平缓，概率过渡区较宽 |
| 9 | $(10,10,10)^T$ | 曲线接近 Step Function（阶跃函数） |

**概率却可以越来越极端**BUT！！！**决策边界不变**。
- 如果权重整体乘负数，score 符号反转，预测类别也会交换。

**结论：** w越大，虽然说最小化了 Cross-Entropy，但Sigmoid 越像阶跃函数，模型给出的概率就越极端。 可能使权重持续增大，需要 Regularisation 约束权重。

---

<a id="sec-3"></a>

# 3. L2 Regularisation / Weight Decay（PPT 11–13）

<a id="sec-3-1"></a>

## 3.1 正则化后的目标函数（PPT 11）

在 Cross-Entropy 后面加上惩罚项：

$$
\tilde{E}(\mathbf{w})=E(\mathbf{w})+\frac{\lambda}{2}\|\mathbf{w}\|_2^2
$$

其中：

$$
\|\mathbf{w}\|_2^2=\mathbf{w}^T\mathbf{w}=\sum_{j=0}^{d}w_j^2
$$

展开为：

$$
\tilde{E}(\mathbf{w})=-\sum_{i=1}^{N}\left[y^{(i)}\ln p_1^{(i)}+(1-y^{(i)})\ln(1-p_1^{(i)})\right]+\frac{\lambda}{2}\sum_{j=0}^{d}w_j^2
$$

- **第一项 $E(\mathbf{w})$**：让预测尽量符合训练集。
- **第二项**：让过大的权重受到惩罚。
- $\lambda$ 是 **Hyperparameter（超参数）**，控制惩罚强度。

| $\lambda$ | 含义 |
|---|---|
| $0$ | 不使用正则化 |
| 较小 | 惩罚较弱，更重视拟合训练集 |
| 较大 | 惩罚较强，可能抑制过拟合；过大则可能欠拟合 |

**易错点：** $\|\mathbf{w}\|_2$ 才是 L2 Norm，目标函数实际上使用其**平方**。课件把 $w_0$ 也纳入求和，本节推导沿用这一约定。

<a id="sec-3-2"></a>

## 3.2 加入 L2 后的 Gradient（PPT 12）

原始 Cross-Entropy 的 Gradient：

$$
\nabla E(\mathbf{w})=\sum_{i=1}^{N}(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)}
$$

正则项的一阶导数：

$$
\nabla\left(\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}\right)=\lambda\mathbf{w}
$$

所以：

$$
\nabla\tilde{E}(\mathbf{w})=\sum_{i=1}^{N}(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)}+\lambda\mathbf{w}
$$

**记忆：新 Gradient = 原 Gradient + $\lambda\mathbf{w}$。**

使用 Gradient Descent 时：

$$
\mathbf{w}_{new}=\mathbf{w}_{old}-\eta\nabla\tilde{E}(\mathbf{w}_{old})
$$

这里 $\eta$ 是 Learning Rate。该更新式是根据课件梯度公式直接写出的。

<a id="sec-3-3"></a>

## 3.3 加入 L2 后的 Hessian（PPT 12）

原始 Hessian：

$$
H_E(\mathbf{w})=\sum_{i=1}^{N}p_1^{(i)}(1-p_1^{(i)})\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^T
$$

L2 正则项的二阶导数为 $\lambda I$。因此：

$$
H_{\tilde{E}}(\mathbf{w})=\sum_{i=1}^{N}p_1^{(i)}(1-p_1^{(i)})\mathbf{x}^{(i)}(\mathbf{x}^{(i)})^T+\lambda I
$$

- $I$：Identity Matrix（单位矩阵），主对角线为 $1$，其余为 $0$。
- **Gradient Descent** 使用 Gradient；**Newton–Raphson / IRLS** 还需要 Hessian。
- **记忆：新 Hessian = 原 Hessian + $\lambda I$。**

> 原课件第 12 页梯度中有一处 $y^{(t)}$，结合该页的样本求和符号，这里统一写为 $y^{(i)}$。

---

<a id="sec-4"></a>

# 4. Underfitting 与非线性变换的必要性（PPT 15–20）

<a id="sec-4-1"></a>

## 4.1 Linearly / Nonlinearly Separable（PPT 15–18）

两个输入变量时，普通 Logistic Regression 的决策边界为：

$$
w_0+w_1x_1+w_2x_2=0
$$

- **PPT 15**：二维数据可以用直线分开，线性模型足够。
- **PPT 16**：两类样本分布复杂，单条直线难以分开，可能 **Underfit（欠拟合）**。
- **PPT 17**：一维蓝点在左、橙点在右，一个阈值即可分开。
- **PPT 18**：一维蓝点在左右、橙点在中间，需要两个阈值；单个线性边界不够。

这里的核心问题是**模型的表示能力不足**，并不是把 Gradient Descent 跑得更久就一定能解决。

<a id="sec-4-2"></a>

## 4.2 Feature Transform / Basis Expansion（PPT 19–20）

给原始变量 $x_1$ 增加平方项：

$$
\phi(\mathbf{x})=(x_1,x_1^2)^T
$$

这里：

- $\phi$：Feature Transform（特征变换）函数。
- $x_1$ 和 $x_1^2$：由原始输入构造的新特征（Basis Functions 的结果）。
- 新的二维空间：Feature Space / Embedding（特征空间 / 嵌入空间）。

**PPT 20 的图：** 原本在一维数轴上无法被一个阈值分开的数据，映射到坐标 $(x_1,x_1^2)$ 后，可以在新空间用**一条直线**分开。

假设该直线表示为：

$$
w_0+w_1\phi_1(\mathbf{x})+w_2\phi_2(\mathbf{x})=0
$$

把 $\phi_1(\mathbf{x})=x_1$、$\phi_2(\mathbf{x})=x_1^2$ 代入：

$$
w_0+w_1x_1+w_2x_1^2=0
$$

**最重要的结论：在新 Feature Space 中是线性边界，在原始空间中却可以是非线性边界。**

---

<a id="sec-5"></a>

# 5. 如何构造 Polynomial Features（PPT 21–22、26–27）

<a id="sec-5-1"></a>

## 5.1 最高 $p$ 次多项式（PPT 21–22）

如果需要原始空间中的最高 $p$ 次多项式决策边界，构造 $\phi(\mathbf{x})$ 时包含**所有次数不超过 $p$ 的项**（包含常数项）。

例：$x_1^2$ 和 $x_1x_2$ 都是二次项；$x_1^2x_2$ 是三次项。

**一个变量，最高二次（PPT 21）：**

$$
\mathbf{x}=(1,x_1)^T
$$

$$
\phi(\mathbf{x})=(1,x_1,x_1^2)^T
$$

**两个变量，最高二次（PPT 21）：**

$$
\mathbf{x}=(1,x_1,x_2)^T
$$

$$
\phi(\mathbf{x})=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)^T
$$

对应决策边界：

$$
w_0+w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2+w_5x_1x_2=0
$$

**两个变量，最高三次（PPT 22）：**

$$
\phi(\mathbf{x})=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2,x_1^3,x_2^3,x_1x_2^2,x_1^2x_2)^T
$$

注意三次多项式中也保留一次项、二次项和常数项。

**原理：** 在 Feature Space 中，模型只是对各个变换后的特征作加权求和；代入原始变量后，就能表示非线性多项式边界。

<a id="sec-5-2"></a>

## 5.2 不一定使用多项式（PPT 26）

课件给出的另一种变换：

$$
\mathbf{x}=(1,x_1)^T
$$

$$
\phi(\mathbf{x})=(1,x_1,e^{x_1})^T
$$

因此变换也可以包含指数函数等非多项式特征。

<a id="sec-5-3"></a>

## 5.3 不一定增加维度（PPT 27）

例如：

$$
(1,x_1,x_2)^T\ \longrightarrow\ (1,x_1,x_2^2)^T
$$

前后都只有三个分量。实际中往往不知道应保留哪些特征，所以会构造较高维的 Feature Space。

---

<a id="sec-6"></a>

# 6. 课件重点例题：为什么有两个决策边界？（PPT 23–25）

已知权重和特征变换：

$$
\mathbf{w}=(10,-7,1)^T
$$

$$
\phi(\mathbf{x})=(1,x_1,x_1^2)^T
$$

**Step 1：在 Feature Space 中写出线性边界。**

$$
\mathbf{w}^T\phi(\mathbf{x})=0
$$

$$
10-7\phi_1(\mathbf{x})+\phi_2(\mathbf{x})=0
$$

**Step 2：用原始变量替换。**

$$
10-7x_1+x_1^2=0
$$

**Step 3：解二次方程。**

$$
x_1=\frac{7\pm\sqrt{49-40}}{2}=\frac{7\pm3}{2}
$$

$$
x_1=2,\qquad x_1=5
$$

因此在原始一维空间中，决策边界有两个：**$x_1=2$ 和 $x_1=5$**。

还可以因式分解：

$$
\mathbf{w}^T\phi(\mathbf{x})=(x_1-2)(x_1-5)
$$

按照 $\mathbf{w}^T\phi(\mathbf{x})\geq0$ 预测 Class 1 的规则：

| 原始输入范围 | 预测类别 |
|---|---|
| $x_1\leq2$ | Class 1 |
| $2<x_1<5$ | Class 0 |
| $x_1\geq5$ | Class 1 |

**对应 PPT 24 的图：** 左边是 Feature Space 中的一条直线，右边却是原始 $x_1$ 轴上的两个分类阈值。**PPT 25 强调：在变换中加入 $x_1^2$，原始决策边界就能出现二次项。**

---

<a id="sec-7"></a>

# 7. 在 Logistic Regression 中应用非线性变换（PPT 28–31）

<a id="sec-7-1"></a>

## 7.1 完整训练流程（PPT 28）

1. **Choose a Feature Transform**：选择 $\phi(\mathbf{x})$，例如 $(1,x_1,x_1^2)^T$。
2. **Transform the Training Set**：每个输入 $\mathbf{x}^{(i)}$ 替换为 $\phi(\mathbf{x}^{(i)})$，标签 $y^{(i)}$ 不变。
3. **Train a Linear Model**：使用之前的 Cross-Entropy、Gradient Descent 或 Newton–Raphson / IRLS 训练。
4. **Substitute Back**：将学到的 $\phi_j(\mathbf{x})$ 代回原始变量，得到原空间中的边界。

变换后的训练集：

$$
T_{\phi}=\{(\phi(\mathbf{x}^{(1)}),y^{(1)}),\ldots,(\phi(\mathbf{x}^{(N)}),y^{(N)})\}
$$

<a id="sec-7-2"></a>

## 7.2 Hypothesis / Score（PPT 29）

**原来：**

$$
p_1=\frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}
$$

**变换后：**

$$
p_1=\frac{1}{1+e^{-\mathbf{w}^T\phi(\mathbf{x})}}
$$

对应的 Log-odds：

$$
\ln\frac{p_1}{1-p_1}=\mathbf{w}^T\phi(\mathbf{x})
$$

**Sigmoid 没有换；变化的只是输入由 $\mathbf{x}$ 变成 $\phi(\mathbf{x})$。**

<a id="sec-7-3"></a>

## 7.3 Cross-Entropy（PPT 30）

为简化记号，设：

$$
p_{\phi}^{(i)}=\frac{1}{1+e^{-\mathbf{w}^T\phi(\mathbf{x}^{(i)})}}
$$

那么变换后的训练损失是：

$$
E_{\phi}(\mathbf{w})=-\sum_{i=1}^{N}\left[y^{(i)}\ln p_{\phi}^{(i)}+(1-y^{(i)})\ln(1-p_{\phi}^{(i)})\right]
$$

**Cross-Entropy 形式不变**，只是预测概率现在基于变换后的特征计算。

<a id="sec-7-4"></a>

## 7.4 Gradient 与 Hessian（PPT 31）

**Gradient：**

$$
\nabla E_{\phi}(\mathbf{w})=\sum_{i=1}^{N}(p_{\phi}^{(i)}-y^{(i)})\phi(\mathbf{x}^{(i)})
$$

**Hessian：**

$$
H_{E_{\phi}}(\mathbf{w})=\sum_{i=1}^{N}p_{\phi}^{(i)}(1-p_{\phi}^{(i)})\phi(\mathbf{x}^{(i)})\phi(\mathbf{x}^{(i)})^T
$$

**联系 2a / 2c：** 计算结构与之前完全一致，只是把每个样本的 $\mathbf{x}^{(i)}$ 替换成 $\phi(\mathbf{x}^{(i)})$。

<a id="sec-7-5"></a>

## 7.5 同时使用 L2（将 PPT 11–12 与 30–31 结合）

$$
\tilde{E}_{\phi}(\mathbf{w})=E_{\phi}(\mathbf{w})+\frac{\lambda}{2}\|\mathbf{w}\|_2^2
$$

$$
\nabla\tilde{E}_{\phi}(\mathbf{w})=\sum_{i=1}^{N}(p_{\phi}^{(i)}-y^{(i)})\phi(\mathbf{x}^{(i)})+\lambda\mathbf{w}
$$

$$
H_{\tilde{E}_{\phi}}(\mathbf{w})=\sum_{i=1}^{N}p_{\phi}^{(i)}(1-p_{\phi}^{(i)})\phi(\mathbf{x}^{(i)})\phi(\mathbf{x}^{(i)})^T+\lambda I
$$

**非线性变换提高表达能力；L2 正则化限制权重过大。** 这三式为前后两部分公式的组合，课件没有单独列出。

---

<a id="sec-8"></a>

# 8. 变换后还是 Linear Model 吗？（PPT 32–33）

**是。课件所说的线性，主要指 Linear in Parameters（对参数线性）。**

例如：

$$
z=w_0+w_1x_1+w_2x_1^2
$$

- 对原始输入 $x_1$：包含平方项，所以是非线性的。
- 对参数 $w_0,w_1,w_2$：每个参数都只以一次方出现，因此 score 仍然对参数线性。

一般形式：

$$
z=\mathbf{w}^T\phi(\mathbf{x})=\sum_{j=0}^{d}w_j\phi_j(\mathbf{x})
$$

Logistic Regression 在变换后的 Feature Space 中仍在学习线性边界，即使该边界在原始输入空间中可能不是直线。

**补充区分：** 这里是说 score / log-odds 对参数线性；最后的概率 $p_1=\sigma(z)$ 本身仍是非线性的 Sigmoid。

Linear Models 的优点（PPT 33）：学习算法常较高效，并可能具有较好的稳健性和泛化能力。

---

<a id="sec-9"></a>

# 9. Nonlinear Transformations 的风险（PPT 34–35）

**PPT 34：Linear Fit 与 4th Order Polynomial 对比。**

- 更高阶的 Polynomial 能拟合更弯曲的边界。
- **训练集拟合得更好，不等于在新数据上泛化得更好。**
- 高阶变换也可能导致 Feature Space 的维度很高。

**PPT 35 总结：**

1. Regularisation 有助于减少 Overfitting。
2. Nonlinear Transformations 可使原空间中非线性可分的问题，在新的 Feature Space 中变得线性可分。
3. 可以继续使用 Logistic Regression，在变换后的空间学习线性决策边界。
4. 相同思路也适用于其他 Linear Models。
5. 下一部分将讨论：怎样在不显式计算高维嵌入的情况下使用它（本课件没有展开）。

---

<a id="sec-10"></a>

# 10. 考前重点公式

| 知识点 | 公式 | PPT |
|---|---|---|
| Score | $z=\mathbf{w}^T\mathbf{x}$ | 4 |
| Sigmoid | $p_1=1/(1+e^{-z})$ | 4 |
| 决策边界 | $\mathbf{w}^T\mathbf{x}=0$ | 4、15 |
| Generalisation Gap | $G=E(\mathbf{w})-\hat{E}(\mathbf{w})$ | 5 |
| L2 Loss | $\tilde{E}=E+(\lambda/2)\|\mathbf{w}\|_2^2$ | 11 |
| L2 Gradient | $\nabla\tilde{E}=\nabla E+\lambda\mathbf{w}$ | 12 |
| L2 Hessian | $H_{\tilde{E}}=H_E+\lambda I$ | 12 |
| Feature Transform | $\mathbf{x}\rightarrow\phi(\mathbf{x})$ | 19–21 |
| 变换后 Score | $z=\mathbf{w}^T\phi(\mathbf{x})$ | 29 |

**记住三句话：**

- **Overfitting**：即使决策边界是线性的，权重也可能过大；用 **Regularisation** 缓解。
- **Underfitting**：原来的线性边界不够灵活；用 **Nonlinear Transformations** 增强表达能力。
- **变换后**：仍然可以使用之前学过的 Logistic Regression、Cross-Entropy、Gradient Descent 和 Newton–Raphson / IRLS。

