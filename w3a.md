
# ML 3a — Regularisation and Nonlinear Transformations

## 0. 本节到底在解决什么问题？

这节课围绕两个**看似相反**的问题：

1. **Overfitting（过拟合）**：模型在训练集上表现很好，却不一定能推广到新数据。即使 Logistic Regression 的决策边界是线性的，它仍可能出现过拟合。课件给出的主要解决思路是 **Regularisation（正则化）**，尤其是 **L2 / Weight Decay**。
2. **Underfitting（欠拟合）**：模型本身过于简单，无法表达数据所需的决策边界。课件给出的主要解决思路是 **Nonlinear Transformations（非线性特征变换）**，把数据映射到新的 Feature Space，再使用熟悉的线性模型。

**一条主线记住：**

- 正则化：**约束模型**，避免权重过大、过度自信。
- 非线性变换：**增强特征表达能力**，使原空间中的复杂边界可以在新空间中用线性边界表示。
- 两者可以一起使用：先变换特征，再在变换后的特征上使用带正则化的 Logistic Regression。

---

## 1. 复习

### 1.1 Logistic Regression 的预测（PPT 第 4 页）

设特征向量为 $\mathbf{x}$，权重为 $\mathbf{w}$；通常把偏置项吸收到向量中，即 $x_0=1$。

模型先计算一个 **score（分数）**：

$$
z=\mathbf{w}^{T}\mathbf{x}.
$$

再用 Sigmoid 把分数变成 Class 1 的预测概率：

$$
p_1=p(y=1\mid\mathbf{x},\mathbf{w})
=\sigma(z)
=\frac{e^z}{1+e^z}
=\frac{1}{1+e^{-z}}.
$$

因此：

$$
\operatorname{logit}(p_1)=\ln\frac{p_1}{1-p_1}=\mathbf{w}^{T}\mathbf{x},
\qquad p_0=1-p_1.
$$

如果用 $0.5$ 作为分类阈值：

- $\mathbf{w}^{T}\mathbf{x}\geq 0$：预测 Class 1。
- $\mathbf{w}^{T}\mathbf{x}<0$：预测 Class 0。
- **Decision Boundary（决策边界）**：$\mathbf{w}^{T}\mathbf{x}=0$。

### 1.2 Cross-Entropy Loss（PPT 第 4 页）

给定 $N$ 个训练样本 $\{(\mathbf{x}^{(i)},y^{(i)})\}_{i=1}^{N}$，其中 $y^{(i)}\in\{0,1\}$，课件使用的**总损失（求和形式）**为：

$$
E(\mathbf{w})=-\sum_{i=1}^{N}
\left[
y^{(i)}\ln p_1^{(i)}+
(1-y^{(i)})\ln(1-p_1^{(i)})
\right],
$$

其中 $p_1^{(i)}=p(y=1\mid\mathbf{x}^{(i)},\mathbf{w})$。

**直觉：** 正确标签的预测概率越低，损失越大。例如真实标签 $y=1$ 时，单个样本的损失就是 $-\ln p_1$。训练模型的目标是降低**训练集**上的损失。

> **易混点：** 第 4 页的 Cross-Entropy 是求和；下一页的 Empirical Risk 写成平均值 $1/N$。两种归一化方式不同，但对固定训练集、没有其他项时，最小化的位置相同；加入正则项时，要留意 $\lambda$ 与损失的尺度。

### 1.3 Training Error / Empirical Risk（PPT 第 5 页）

$$
\widehat E(\mathbf{w})=
\frac{1}{N}\sum_{i=1}^{N}
F\!\left(g(\mathbf{x}^{(i)}),y^{(i)}\right).
$$

- $g(\mathbf{x})$：模型的预测函数。
- $F(\cdot,\cdot)$：用于衡量预测误差的 Error Function，例如某种分类损失。
- $\widehat E$：在**现有训练样本**上计算出来的平均误差。

**特别注意：** Training Error / Empirical Risk 用的是 **Training Set，不是 Test Set**。

### 1.4 Generalisation Error / Population Risk（PPT 第 5 页）

$$
E(\mathbf{w})=
\mathbb{E}_{(\mathbf{x},y)\sim p(\mathbf{x},y)}
\left[F(g(\mathbf{x}),y)\right].
$$

它衡量的是模型在真实数据分布中的**期望误差**，而不是只在已见过的训练集上的误差。课件指出：**Generalisation Error 小，说明模型泛化得好。**

Generalisation Gap（泛化差距）：

$$
\operatorname{Generalisation\ Gap}
=E(\mathbf{w})-\widehat E(\mathbf{w}).
$$

当训练误差很小、真实分布上的误差却很大时，泛化差距可能很大。

### 1.5 Inductive Bias（归纳偏置）（PPT 第 5–6 页）

**Inductive Bias** 指学习算法为了从有限训练数据推断未见过的数据，而采用的假设或偏好。

Logistic Regression 的一个强归纳偏置是：**在当前输入特征空间中，Decision Boundary 是线性的。**

这既是优点（模型相对简单），也可能成为局限（无法直接表达复杂的非线性边界）。

---

## 2. Logistic Regression 为什么也会 Overfit？（PPT 第 6–10、13 页）

### 2.1 线性可分时会出现什么？（PPT 第 6 页）

**Linearly Separable（线性可分）**：可以找到一条直线（二维情形）或超平面（更高维），把训练数据中的两类分开。

课件强调：**即使 Logistic Regression 具有线性决策边界，也依然可能出现过拟合。**

原因是 Cross-Entropy 不仅关心“分类对没对”，也关心预测概率：

- 对真实 Class 1 的样本，模型希望 $p_1$ 越接近 $1$ 越好。
- 对真实 Class 0 的样本，模型希望 $p_1$ 越接近 $0$ 越好。
- 在线性可分的训练集上，即使已经分对所有点，继续增大权重的绝对值，仍可以让预测概率更极端，并继续降低 Cross-Entropy。

### 2.2 权重越大，Sigmoid 会怎样？（PPT 第 7–10 页）

原课件展示了以下四组权重：

| PPT 页 | 权重 $\mathbf{w}$ | 图上展示的现象 |
|---|---|---|
| 第 7 页 | $(0.5,0.5,0.5)^T$ | Sigmoid 变化相对平滑；概率从 0 到 1 的过渡区域较宽。 |
| 第 8 页 | $(3,3,3)^T$ | 过渡区域变窄，预测更接近 0 或 1。 |
| 第 9 页 | $(10,10,10)^T$ | 过渡极陡，Sigmoid 更接近 Step Function（阶跃函数）。 |
| 第 10 页 | $(-10,-10,-10)^T$ | 概率方向反转；边界仍然很陡。 |

**理解补充：** 固定某个输入 $\mathbf{x}$，设原来的 score 是 $z=\mathbf{w}^{T}\mathbf{x}$。如果把所有权重乘以正数 $c$，则：

$$
z'=(c\mathbf{w})^T\mathbf{x}=cz,
\qquad p_1'=\sigma(cz).
$$

- 当 $c>1$，且 $z>0$ 时，$p_1'$ 更接近 $1$。
- 当 $c>1$，且 $z<0$ 时，$p_1'$ 更接近 $0$。
- $c>0$ 时，$cz=0$ 与 $z=0$ 表示**同一条决策边界**。因此概率可以变得越来越极端，**即使分类线没有动**。
- 当 $c<0$，score 的符号反转，预测两侧也会互换。第 10 页的例子就是这种情况。

### 2.3 为什么这有风险？（PPT 第 10、13 页）

Sigmoid 越接近阶跃函数，模型就越容易给出接近 $0$ 或 $1$ 的极端概率。课件的表述是：这会使预测概率失去应有的意义。

**理解补充：** 这里并不意味着“所有大权重都必然过拟合”；重点是：在课件的线性可分场景中，**仅靠最小化训练 Cross-Entropy，不能有效约束权重的规模**。这可能损害模型在未见过的数据上的表现与概率可信度。

---

## 3. Regularisation：L2 / Weight Decay（PPT 第 11–13 页）

### 3.1 核心思想（PPT 第 11 页）

为了不让权重变得过大，在原始 Cross-Entropy 后面加一个惩罚项：

$$
\boxed{\widetilde E(\mathbf{w})
=E(\mathbf{w})+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2}
$$

其中：

$$
\lVert\mathbf{w}\rVert_2^2
=\mathbf{w}^T\mathbf{w}
=\sum_{j=0}^{d}w_j^2.
$$

展开成课件的形式：

$$
\widetilde E(\mathbf{w})
=-\sum_{i=1}^{N}\left[
 y^{(i)}\ln p_1^{(i)}+
 (1-y^{(i)})\ln(1-p_1^{(i)})
\right]
+\frac{\lambda}{2}\sum_{j=0}^{d}w_j^2.
$$

两个部分各管一件事：

| 目标函数部分 | 作用 |
|---|---|
| $E(\mathbf{w})$ | 让模型尽量拟合训练数据（减小 Cross-Entropy）。 |
| $\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$ | 惩罚权重过大，限制模型过度自信。 |

**$\lambda$（Lambda）是 Hyperparameter（超参数）**，控制正则化的强度：

- $\lambda=0$：没有 L2 正则化。
- $\lambda$ 较小：主要关注训练损失，对权重限制较弱。
- $\lambda$ 较大：更强烈地惩罚大权重，但太大也可能使模型欠拟合。

**公式记号注意：** 原 PPT 第 11 页在文字里把 $\mathbf{w}^T\mathbf{w}$ 称作“L2 norm”；严格地说，$\lVert\mathbf{w}\rVert_2=\sqrt{\sum_j w_j^2}$ 才是 L2 范数，而公式用的是**L2 范数的平方**。这里按原目标函数保留平方惩罚的数学含义。

> **理解补充：** 课件的求和 $\sum_{j=0}^{d}$ 包括 $w_0$。实际实现中也经常不对偏置项 $w_0$ 做正则化，但这属于实现选项，**本节推导遵循 PPT 的 $\lambda\mathbf{w}$ 和 $\lambda I$ 形式**。

### 3.2 Gradient（梯度）怎样改变？（PPT 第 12 页）

你在 2a 学过，原始 Cross-Entropy 的梯度是：

$$
\nabla E(\mathbf{w})
=\sum_{i=1}^{N}
(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)}.
$$

因为：

$$
\nabla_{\mathbf{w}}
\left(\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}\right)
=\lambda\mathbf{w},
$$

所以加入 L2 后：

$$
\boxed{\nabla\widetilde E(\mathbf{w})
=\sum_{i=1}^{N}
(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)}
+\lambda\mathbf{w}.}
$$

理解为：**原来的梯度 + 正则项自己的梯度**。

因此 Gradient Descent 更新变为：

$$
\mathbf{w}_{\text{new}}
=\mathbf{w}_{\text{old}}
-\eta\left[
\sum_{i=1}^{N}(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)}
+\lambda\mathbf{w}_{\text{old}}
\right],
$$

其中 $\eta$ 是 Learning Rate（学习率）。上式是根据 PPT 梯度写出的**理解补充**。

### 3.3 Hessian（海森矩阵）怎样改变？（PPT 第 12 页）

原来的 Hessian 是：

$$
H_E(\mathbf{w})
=\sum_{i=1}^{N}
 p_1^{(i)}(1-p_1^{(i)})
 \mathbf{x}^{(i)}\mathbf{x}^{(i)T}.
$$

L2 惩罚项的二阶导数是：

$$
H_{\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2}
=\lambda I.
$$

所以：

$$
\boxed{H_{\widetilde E}(\mathbf{w})
=\sum_{i=1}^{N}
 p_1^{(i)}(1-p_1^{(i)})
 \mathbf{x}^{(i)}\mathbf{x}^{(i)T}
+\lambda I.}
$$

- $I$：Identity Matrix（单位矩阵），对角线为 $1$，其余元素为 $0$。
- **Gradient Descent** 主要使用 Gradient。
- **Newton–Raphson / IRLS** 需要 Gradient 和 Hessian。

**课件排版备注：** 第 12 页梯度公式出现 $y^{(t)}$，但同页和前后文都以第 $i$ 个样本为求和对象；本笔记统一写为 $y^{(i)}$，并在此注明该处记号差异。

### 3.4 这一部分的结论（PPT 第 13 页）

**Cross-Entropy 单独使用时，线性可分训练数据也可能带来过拟合问题；Weight Decay Regularisation 可以帮助缓解。**

---

## 4. 为什么需要 Nonlinear Transformations？（PPT 第 15–20 页）

### 4.1 线性可分与非线性可分（PPT 第 15–18 页）

在原始二维输入空间 $(x_1,x_2)$ 中，Logistic Regression 的分界线是：

$$
w_0+w_1x_1+w_2x_2=0.
$$

**第 15 页：** 两类点可以被一条直线分开，线性模型就足够。

**第 16 页：** 数据的分类形状比较复杂，一条直线不能把两类很好地分开；若坚持只用这类边界，就可能 **Underfit（欠拟合）**。

**第 17 页：** 在一维 $x_1$ 上，蓝点在左、橙点在右，只需**一个阈值**就能分开。

**第 18 页：** 在一维 $x_1$ 上，蓝点在左右两侧、橙点在中间，一个阈值不够，因为分类需要两个分界点。

核心问题不是“优化算法不够厉害”，而是：**当前特征下，模型能表示的决策边界不够灵活。**

### 4.2 把一维变成二维：Basis Expansion（PPT 第 19–20 页）

第 18 页的例子是一维非线性可分问题。课件引入第二个特征：

$$
x_2=x_1^2.
$$

把一个原始输入 $x_1$ 转换为：

$$
\boxed{\boldsymbol\phi(\mathbf{x})
=\begin{pmatrix}x_1\\x_1^2\end{pmatrix}.}
$$

这里：

- $\boldsymbol\phi$（Phi）叫 **Feature Transform / Nonlinear Transformation（特征变换）**。
- $x_1$、$x_1^2$ 是构造出来的 **Basis Functions（基函数）** 对应的特征。
- 新的坐标空间叫 **Feature Space / Embedding（特征空间 / 嵌入空间）**。

原来所有点都在一条数轴上；变换后，每个点变成二维坐标 $(x_1,x_1^2)$，落在一条抛物线上。

**第 20 页的关键图：** 在新二维空间中，可以画一条**直线**把两类点分开。因此，新的空间里使用线性模型，仍能解决原空间中需要非线性分界的问题。

### 4.3 最容易理解错的一句话

> **模型在 Feature Space 中线性，不等于它在原始输入 $x$ 中也是线性的。**

例如，新空间内的直线：

$$
w_0+w_1\phi_1(\mathbf{x})+w_2\phi_2(\mathbf{x})=0
$$

代入 $\phi_1(\mathbf{x})=x_1$、$\phi_2(\mathbf{x})=x_1^2$，就变成：

$$
w_0+w_1x_1+w_2x_1^2=0.
$$

**这是原始变量 $x_1$ 的二次方程，不再是 $x_1$ 的一次线性边界。**

---

## 5. Polynomial Features：如何设计 $\boldsymbol\phi(\mathbf{x})$？（PPT 第 21–25 页）

### 5.1 一般原则（PPT 第 21–22 页）

如果希望原始空间中的决策边界可以是**最高 $p$ 次多项式（Polynomial of Order $p$）**，则构造 $\boldsymbol\phi(\mathbf{x})$ 时，包括能从原始输入变量生成的**所有次数不超过 $p$ 的项**。

这里的次数是**一项中各个变量指数的总和**。

例如：

- $x_1^2$ 是 2 次项。
- $x_1x_2$ 是 2 次项。
- $x_1^2x_2$ 是 3 次项。

### 5.2 只有一个输入变量，最高 2 次（PPT 第 21 页）

原始输入（包含常数特征）：

$$
\mathbf{x}=(1,x_1)^T.
$$

变换后：

$$
\boxed{\boldsymbol\phi(\mathbf{x})=(1,x_1,x_1^2)^T.}
$$

模型的分界式：

$$
\mathbf{w}^T\boldsymbol\phi(\mathbf{x})=0
\iff w_0+w_1x_1+w_2x_1^2=0.
$$

### 5.3 两个输入变量，最高 2 次（PPT 第 21 页）

$$
\mathbf{x}=(1,x_1,x_2)^T
$$

$$
\boxed{\boldsymbol\phi(\mathbf{x})
=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)^T.}
$$

- 常数项：$1$
- 一次项：$x_1,x_2$
- 二次项：$x_1^2,x_2^2,x_1x_2$

得到的原空间决策边界是：

$$
w_0+w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2+w_5x_1x_2=0.
$$

这可以表达比一条直线丰富得多的曲线。

### 5.4 两个输入变量，最高 3 次（PPT 第 22 页）

$$
\boxed{\boldsymbol\phi(\mathbf{x})=
(1,x_1,x_2,x_1^2,x_2^2,x_1x_2,
 x_1^3,x_2^3,x_1x_2^2,x_1^2x_2)^T.}
$$

比 2 次多出的 3 次项为：

$$
x_1^3,\quad x_2^3,\quad x_1x_2^2,\quad x_1^2x_2.
$$

**课件结论：** 原始空间里次数不超过 $p$ 的多项式决策边界，在包含相应多项式项的 $\boldsymbol\phi(\mathbf{x})$ 空间中，对变换后的特征而言是线性的。

### 5.5 课件完整数值例子：边界 $2$ 和 $5$ 从哪里来？（PPT 第 23–25 页）

给定：

$$
\mathbf{w}^T=(10,-7,1),
\qquad\boldsymbol\phi(\mathbf{x})=(1,x_1,x_1^2)^T.
$$

在变换后的 Feature Space 里，线性边界为：

$$
\mathbf{w}^T\boldsymbol\phi(\mathbf{x})=0.
$$

**第 1 步：代入权重。**

$$
10\cdot1-7\phi_1(\mathbf{x})+\phi_2(\mathbf{x})=0.
$$

**第 2 步：把基函数换回原来的 $x_1$。**

$$
10-7x_1+x_1^2=0.
$$

**第 3 步：解二次方程。**

$$
x_1=\frac{7\pm\sqrt{(-7)^2-4\cdot1\cdot10}}{2\cdot1}
=\frac{7\pm3}{2}.
$$

因此：

$$
\boxed{x_1=2\quad\text{或}\quad x_1=5.}
$$

这意味着**新空间中的一条直线**，在**原始一维空间中对应两个决策阈值**：

$$
\underbrace{x_1<2}_{\text{一侧}}
\quad\big|\quad
\underbrace{2<x_1<5}_{\text{另一侧}}
\quad\big|\quad
\underbrace{x_1>5}_{\text{一侧}}.
$$

若按 $\mathbf{w}^T\boldsymbol\phi(\mathbf{x})\geq0$ 预测 Class 1，那么 $x_1\leq2$ 和 $x_1\geq5$ 处预测 Class 1，$2<x_1<5$ 预测 Class 0（等号按本课前面约定归为 Class 1）。

**第 24 页图像对应：** 左图在 $(\phi_1,\phi_2)$ 空间画直线；右图在原始 $x_1$ 轴上看到两个分界位置 $2$、$5$。

**第 25 页强调：** 如果变换里包含某个基函数，原始空间的边界表达式就可以包含与之对应的项。例如加入 $x_1^2$，边界就可以出现二次项。

---

## 6. 非线性变换不一定是多项式，也不一定升维（PPT 第 26–27 页）

### 6.1 Other Nonlinear Transformations（PPT 第 26 页）

多项式只是一种选择。课件给出的非多项式例子：

$$
\mathbf{x}=(1,x_1)^T
\longrightarrow
\boldsymbol\phi(\mathbf{x})=(1,x_1,e^{x_1})^T.
$$

也就是说，原始输入变量可以变成 $x_1^2$、$x_1^3$，也可以变成 $e^{x_1}$ 等其他非线性函数。

**重点不是固定使用平方，而是选择能帮助模型表示目标边界的 Feature Transform。**

### 6.2 Dimensionality of the Feature Space（PPT 第 27 页）

通常 Nonlinear Transformation 会让特征维度增加，但**并不一定**。

课件举例：

$$
\mathbf{x}=(1,x_1,x_2)^T
\longrightarrow
\boldsymbol\phi(\mathbf{x})=(1,x_1,x_2^2)^T.
$$

原输入与新特征都是 3 个分量，只是把 $x_2$ 替换成 $x_2^2$。如果问题不需要 $x_1^2$ 或 $x_1x_2$，就没必要强制包含这些特征。

不过课件也指出：**在实际问题中，通常事先不知道哪些基函数最有用，所以经常会构造更高维的嵌入。**

---

## 7. 真正训练时该怎么用？（PPT 第 28–31 页）

### 7.1 四步流程（PPT 第 28 页）

**Step 1 — Choose a Nonlinear Transformation（选择变换）**

例如：

$$
\boldsymbol\phi(\mathbf{x})=(1,x_1,x_1^2)^T.
$$

**Step 2 — Transform the Training Set（转换训练集）**

原数据：

$$
\mathcal{T}=\{(\mathbf{x}^{(1)},y^{(1)}),\ldots,
(\mathbf{x}^{(N)},y^{(N)})\}.
$$

转换后：

$$
\boxed{\mathcal{T}_{\phi}=
\{(\boldsymbol\phi(\mathbf{x}^{(1)}),y^{(1)}),\ldots,
(\boldsymbol\phi(\mathbf{x}^{(N)}),y^{(N)})\}.}
$$

**注意：** 变的是输入特征；样本标签 $y^{(i)}$ 不变。

**Step 3 — Train a Linear Model in the New Space（在新空间训练线性模型）**

使用之前学过的学习算法，最小化：

$$
\underset{\mathbf{w}}{\operatorname{argmin}}\ E(\mathbf{w})
\quad\text{或}\quad
\underset{\mathbf{w}}{\operatorname{argmin}}\ \widetilde E(\mathbf{w}).
$$

可用 Gradient Descent 或 Newton–Raphson / IRLS 等方法。

**Step 4 — Express the Boundary Using the Original Variables（换回原始变量）**

$$
w_0+w_1\phi_1(\mathbf{x})+w_2\phi_2(\mathbf{x})=0
$$

$$
\Downarrow
$$

$$
w_0+w_1x_1+w_2x_1^2=0.
$$

### 7.2 Logistic Regression 的 Hypothesis 怎样变？（PPT 第 29 页）

**变换前：**

$$
\operatorname{logit}(p_1)=\mathbf{w}^T\mathbf{x},
\qquad
p_1=\frac{e^{\mathbf{w}^T\mathbf{x}}}
{1+e^{\mathbf{w}^T\mathbf{x}}}.
$$

**变换后：**

$$
\boxed{\operatorname{logit}(p_1)=\mathbf{w}^T\boldsymbol\phi(\mathbf{x})}
$$

$$
\boxed{p_1=p(1\mid\boldsymbol\phi(\mathbf{x}),\mathbf{w})
=\frac{e^{\mathbf{w}^T\boldsymbol\phi(\mathbf{x})}}
{1+e^{\mathbf{w}^T\boldsymbol\phi(\mathbf{x})}}.}
$$

**记忆：** 把原来所有代表输入特征的 $\mathbf{x}$，换成新的 $\boldsymbol\phi(\mathbf{x})$。**Sigmoid 本身没有变。**

### 7.3 Cross-Entropy Loss 怎样变？（PPT 第 30 页）

原损失：

$$
E(\mathbf{w})=-\sum_{i=1}^{N}
\left[y^{(i)}\ln p(1\mid\mathbf{x}^{(i)},\mathbf{w})
+(1-y^{(i)})\ln(1-p(1\mid\mathbf{x}^{(i)},\mathbf{w}))\right].
$$

变换后的损失：

$$
\boxed{E_{\phi}(\mathbf{w})=-\sum_{i=1}^{N}
\left[y^{(i)}\ln p(1\mid\boldsymbol\phi(\mathbf{x}^{(i)}),\mathbf{w})
+(1-y^{(i)})\ln(1-p(1\mid\boldsymbol\phi(\mathbf{x}^{(i)}),\mathbf{w}))\right].}
$$

**损失函数的逻辑没变**；改变的是每个样本送入模型的特征。

### 7.4 Gradient 和 Hessian 怎样变？（PPT 第 31 页）

原模型的梯度和 Hessian：

$$
\nabla E(\mathbf{w})
=\sum_{i=1}^{N}(p_1^{(i)}-y^{(i)})\mathbf{x}^{(i)},
$$

$$
H_E(\mathbf{w})
=\sum_{i=1}^{N}p_1^{(i)}(1-p_1^{(i)})
\mathbf{x}^{(i)}\mathbf{x}^{(i)T}.
$$

变换后（此处 $p_{1,\phi}^{(i)}=p(1\mid\boldsymbol\phi(\mathbf{x}^{(i)}),\mathbf{w})$）：

$$
\boxed{\nabla E_{\phi}(\mathbf{w})
=\sum_{i=1}^{N}
(p_{1,\phi}^{(i)}-y^{(i)})\boldsymbol\phi(\mathbf{x}^{(i)}).}
$$

$$
\boxed{H_{E_{\phi}}(\mathbf{w})
=\sum_{i=1}^{N}
p_{1,\phi}^{(i)}(1-p_{1,\phi}^{(i)})
\boldsymbol\phi(\mathbf{x}^{(i)})
\boldsymbol\phi(\mathbf{x}^{(i)})^T.}
$$

**这就是本节和 2a / 2c 的联系：** Gradient Descent、Newton–Raphson / IRLS **算法框架仍然可用**，只是训练时使用的特征从 $\mathbf{x}$ 换成 $\boldsymbol\phi(\mathbf{x})$。

### 7.5 同时使用 Nonlinear Transformation 和 L2（综合理解）

这是把课件的两个部分组合后的**理解补充**：

$$
\widetilde E_{\phi}(\mathbf{w})
=E_{\phi}(\mathbf{w})+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2.
$$

$$
\nabla\widetilde E_{\phi}(\mathbf{w})
=\sum_{i=1}^{N}
(p_{1,\phi}^{(i)}-y^{(i)})\boldsymbol\phi(\mathbf{x}^{(i)})
+\lambda\mathbf{w}.
$$

$$
H_{\widetilde E_{\phi}}(\mathbf{w})
=\sum_{i=1}^{N}
p_{1,\phi}^{(i)}(1-p_{1,\phi}^{(i)})
\boldsymbol\phi(\mathbf{x}^{(i)})\boldsymbol\phi(\mathbf{x}^{(i)})^T
+\lambda I.
$$

直觉：**变换让模型有能力学复杂边界；正则化帮助避免复杂边界过度适应训练集。**

---

## 8. 变换以后，Logistic Regression 还是 Linear Model 吗？（PPT 第 32–33 页）

**答案：按照课件使用的定义，仍然是。**

课件强调，“线性模型”通常指的是**对参数 $\mathbf{w}$ 线性（Linear in Parameters）**，而不是一定要对原始变量 $\mathbf{x}$ 线性。

例如：

$$
z=w_0+w_1x_1+w_2x_1^2.
$$

- 对 **$x_1$**：式子包含 $x_1^2$，是非线性的。
- 对 **$w_0,w_1,w_2$**：每个参数只以一次方出现，且没有参数之间的乘积，所以 score 对参数仍是线性的。

更一般地：

$$
z=\mathbf{w}^T\boldsymbol\phi(\mathbf{x})
=\sum_{j}w_j\phi_j(\mathbf{x}).
$$

因此新模型可以说：**Linear in Parameters, Nonlinear in Original Inputs**。

> **容易混淆的细节：** 这里说 Logistic Regression “线性”，指的是其 **logit / score 对参数线性、特征空间中的决策边界线性**；最终输出概率 $\sigma(z)$ 本身是 Sigmoid 非线性函数。这两句话并不矛盾。

课件第 33 页还指出 Linear Models 的优点：学习算法通常相对高效；它们也可能具有较好的 Robustness（稳健性）与 Generalisation（泛化）表现。

---

## 9. Caveats：Nonlinear Transformations 的风险（PPT 第 34–35 页）

### 9.1 拟合得更好，不等于泛化得更好（PPT 第 34 页）

课件对比了 **Linear Fit（线性拟合）** 与 **4th Order Polynomial（四次多项式）**：

- 更高阶的多项式能够画出更弯曲的边界。
- 它可能把训练集分得更精细，却不一定对新样本预测得更好。
- 因此，提升表达能力也可能提高 Overfitting 风险。

**理解补充：** 评估是否泛化得好，需要依靠没有参与拟合的数据（例如 Validation / Test 的相应评估），而不能只看训练损失。

### 9.2 特征空间可能非常高维（PPT 第 34 页）

当输入变量很多、最高多项式次数 $p$ 又很高时，构造出的基函数可能非常多。课件强调这会带来维度问题。

**理解补充（非课件公式）：** 如果有 $d$ 个原始变量，包含所有次数 $\leq p$ 的多项式项（包括常数项），特征数是：

$$
\binom{d+p}{p}.
$$

例如 $d=2,p=3$ 时，共有 $\binom{5}{3}=10$ 个特征，正好与第 22 页列出的 10 项一致。

### 9.3 最后一页总结（PPT 第 35 页）

1. **Regularisation** 可以帮助减轻 Overfitting。
2. **Nonlinear Transformations** 可把原本非线性可分的问题变成在新 Feature Space 中线性可分的问题。
3. 在新空间里继续使用原来的 **Logistic Regression** 学习线性决策边界。
4. 这种思路也可以用于其他线性模型。
5. 课件最后预告：**如何使用高维嵌入，而不必显式计算它？**（本份课件未展开方法。）

---

## 10. 核心公式总表（复习时看这一页）

| 内容 | 记忆重点 | PPT |
|---|---|---|
| Logistic Score | $z=\mathbf{w}^T\mathbf{x}$ | 4 |
| 预测概率 | $p_1=\sigma(z)=1/(1+e^{-z})$ | 4 |
| 决策边界 | $\mathbf{w}^T\mathbf{x}=0$ | 4、15 |
| Generalisation Gap | $E(\mathbf{w})-\widehat E(\mathbf{w})$ | 5 |
| L2 Regularisation | $\widetilde E=E+(\lambda/2)\lVert\mathbf{w}\rVert_2^2$ | 11 |
| L2 Gradient | $\nabla\widetilde E=\nabla E+\lambda\mathbf{w}$ | 12 |
| L2 Hessian | $H_{\widetilde E}=H_E+\lambda I$ | 12 |
| Feature Transform | $\mathbf{x}\mapsto\boldsymbol\phi(\mathbf{x})$ | 19–21 |
| 新的 Score | $z=\mathbf{w}^T\boldsymbol\phi(\mathbf{x})$ | 29 |
| 新的 Gradient | $\sum_i(p_{1,\phi}^{(i)}-y^{(i)})\boldsymbol\phi(\mathbf{x}^{(i)})$ | 31 |
| 新的 Hessian | $\sum_i p_{1,\phi}^{(i)}(1-p_{1,\phi}^{(i)})\boldsymbol\phi_i\boldsymbol\phi_i^T$ | 31 |

其中最后一行的 $\boldsymbol\phi_i=\boldsymbol\phi(\mathbf{x}^{(i)})$。

## 11. 考前最容易混淆的 6 组概念

| 容易混淆的概念 | 正确区分 |
|---|---|
| **Training Error vs Generalisation Error** | 前者在训练集计算；后者是真实分布上的期望误差。 |
| **Overfitting vs Underfitting** | 前者过于适应训练集；后者表达能力不足，难以拟合数据规律。 |
| **Regularisation vs Nonlinear Transformation** | 前者限制权重；后者改变特征表达能力。 |
| **Decision Boundary vs Probability** | 同一分界线也可以对应不同的概率陡峭程度（见第 7–9 页的正比例权重）。 |
| **Original Space vs Feature Space** | 原空间中可能需要非线性曲线；新空间中可用线性边界。 |
| **Linear in Parameters vs Linear in Inputs** | $\mathbf{w}^T\boldsymbol\phi(\mathbf{x})$ 对参数线性，但对原输入不一定线性。 |

## 12. 自测题（根据课件知识整理）

**Q1.** 线性可分的数据中，为什么 Logistic Regression 即使全分对了，还可能继续降低训练 Cross-Entropy？

**Q2.** L2 Regularisation 为什么能抑制权重变大？加入后 Gradient 和 Hessian 分别增加什么？

**Q3.** 原始输入为 $(1,x_1,x_2)^T$，如果要求最高 2 次的多项式边界，$\boldsymbol\phi(\mathbf{x})$ 包含哪 6 项？

**Q4.** $\boldsymbol\phi(\mathbf{x})=(1,x_1,x_1^2)^T$，$\mathbf{w}^T=(10,-7,1)$：在原始 $x_1$ 空间中，边界在哪里？

**Q5.** 使用了 $x_1^2$ 后，Logistic Regression 为什么仍可称为 Linear Model？

**Q6.** 如果想把第 31 页的梯度用于变换后的训练集，原本的 $\mathbf{x}^{(i)}$ 应替换成什么？

### 参考答案

1. 因为损失不仅取决于分类正确与否，还取决于预测概率；增大某些权重可以让正确样本的预测概率更接近 $1$ 或 $0$。
2. 在损失中增加 $\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$，使大权重受到更高惩罚；Gradient 增加 $\lambda\mathbf{w}$；Hessian 增加 $\lambda I$。
3. $(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)^T$。
4. $x_1^2-7x_1+10=(x_1-2)(x_1-5)=0$，所以在 $2$ 和 $5$。
5. 因为 $w_0+w_1x_1+w_2x_1^2$ 仍对参数 $w_0,w_1,w_2$ 线性；模型只是在原始输入上非线性。
6. 替换成 $\boldsymbol\phi(\mathbf{x}^{(i)})$，并同时用变换后特征计算 $p_1^{(i)}$。

---

**一句话复习：** **Regularisation 控制复杂度；Nonlinear Transformations 增强表达能力；在新的 Feature Space 里，我们仍然可以使用之前学过的 Logistic Regression、Cross-Entropy、Gradient Descent 和 Newton–Raphson / IRLS。**

*注：本笔记依据用户提供的 35 页课件整理；原课件内容与额外学习解释已通过“理解补充”等标识区分。*
