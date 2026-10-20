# 4a · Lagrange Relaxation and Duality

拉格朗日松弛与对偶性 | Machine Learning

## 目录

- [1. 从 SVM 看原始优化问题](#svm-primal)
- [2. Lagrange Relaxation：把约束放进目标函数](#lagrange-relaxation)
- [3. Minimax Primal：用 Lagrangian 表达原问题](#minimax-primal)
- [4. Dual Formulation：交换优化顺序](#dual-formulation)
- [5. Strong Duality：什么时候两者相等](#strong-duality)
- [6. KKT Conditions：四组最优性条件](#kkt-conditions)
- [7. 用 Stationarity 消去原变量](#eliminate-primal-variables)
- [8. 完整例题：两个变量、一个约束](#worked-example)
- [9. 核心公式与易错点](#revision)

<a id="svm-primal"></a>

## 1. 从 SVM 看原始优化问题

### Maximum Margin Classifier

在前一讲中，Hard-margin SVM 的决策边界是：

$$
h(x)=w^Tx+b=0.
$$

寻找最大间隔，可以写成下面的约束优化问题：

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad
y^{(n)}\bigl(w^Tx^{(n)}+b\bigr)\ge1,
\quad n=1,\ldots,N.
$$

- **目标函数（Objective function）**：让 $\frac12\|w\|^2$ 尽可能小，从而增大几何间隔。
- **约束（Constraints）**：每个训练样本都要分类正确，并达到规定的间隔。
- $y^{(n)}\in\{-1,+1\}$，每个样本对应一个不等式约束。

### 加入 Basis Expansion

使用特征变换 $\phi(x)$ 后：

$$
h(x)=w^T\phi(x)+b,
$$

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to}\quad
y^{(n)}\bigl(w^T\phi(x^{(n)})+b\bigr)\ge1.
$$

取 $\phi(x)=x$，就回到原来的线性模型。若变换后的空间维度很高，显式计算 $\phi(x)$、存储并优化 $w$ 都可能很昂贵。

**本讲的目标：将 Primal problem 改写成 Dual problem，让优化变量从原来的参数变成 Lagrange multipliers。** 在后续 SVM 推导中，这能帮助我们消去显式的 $w$ 和 $b$，并发现更方便的计算结构。

<a id="lagrange-relaxation"></a>

## 2. Lagrange Relaxation：把约束放进目标函数

### 统一写成小于等于零

先考虑一般形式：

$$
\min_x F(x)
\quad\text{subject to}\quad
f_i(x)\le0,\quad i=1,\ldots,N.
$$

这里的 $x$ 是**待优化的变量**，可以是向量；它不一定是 SVM 中的输入样本。$F(x)$ 是目标函数，$f_i(x)$ 是第 $i$ 个约束函数，二者不要混淆。

例如，把 SVM 的约束改成标准形式：

$$
y^{(n)}\bigl(w^T\phi(x^{(n)})+b\bigr)\ge1
\iff
1-y^{(n)}\bigl(w^T\phi(x^{(n)})+b\bigr)\le0.
$$

满足全部约束的点叫 **Feasible solution（可行解）**；这些点组成 **Feasible set（可行域）**。

### Lagrangian 与 Lagrange multipliers

单个约束时，引入一个非负乘子 $a\ge0$：

$$
L(x,a)=F(x)+af(x).
$$

多个约束时，每个约束各有一个乘子：

$$
\boxed{
L(x,a)=F(x)+\sum_{i=1}^{N}a_i f_i(x),
\qquad a_i\ge0.
}
$$

- $L(x,a)$：**Lagrangian（拉格朗日函数）**。
- $a_i$：**Lagrange multiplier（拉格朗日乘子）**，控制第 $i$ 个约束项的影响。
- $a=(a_1,\ldots,a_N)^T$；$a\ge0$ 表示每个分量都非负。
- 固定 $a$ 后，Lagrange relaxation 求的是 $\min_x L(x,a)$，原来的不等式约束已被移入目标函数。

### 为什么要求乘子非负

由于我们在做最小化，目标值越大越不受欢迎。

| 约束状态 | 约束项 $a_i f_i(x)$ | 对最小化的影响 |
|---|---|---|
| $f_i(x)>0$：违反约束 | 若 $a_i>0$，则为正 | 提高目标值，形成惩罚 |
| $f_i(x)=0$：恰好在边界 | 为零 | 不改变目标值 |
| $f_i(x)<0$：严格满足约束 | 若 $a_i>0$，则为负 | 降低目标值，形成“奖励” |

若允许 $a_i<0$，违反约束反而可能降低目标值。因此，**对于这里的 $f_i(x)\le0$ 形式，要求 $a_i\ge0$**。

### 固定乘子，不保证得到原问题的最优解

课件中，取单个乘子 $a=1$，比较两个候选点：

| 候选点 | $F(x)$ | $f(x)$ | $L(x,1)=F(x)+f(x)$ | 是否可行 |
|---|---:|---:|---:|---|
| A | 1 | 10 | 11 | 否 |
| B | 20 | 0 | 20 | 是 |

最小化 $L$ 会偏好 A，因为 $11<20$，但 A 违反约束。**有惩罚不等于惩罚足够大。**

即使两个点都可行，固定乘子也可能改变排序：

| 候选点 | $F(x)$ | $f(x)$ | $L(x,1)$ |
|---|---:|---:|---:|
| C | 1 | 0 | 1 |
| D | 20 | $-30$ | $-10$ |

原目标更偏好 C，但松弛后的目标更偏好 D。

对任何可行点和任何非负乘子，都有：

$$
\boxed{L(x,a)\le F(x).}
$$

这只是因为每个 $a_i f_i(x)\le0$；**不能据此认定两个优化问题等价**。

<a id="minimax-primal"></a>

## 3. Minimax Primal：用 Lagrangian 表达原问题

### 核心想法：对乘子再做一次最大化

固定乘子有问题，于是对每个固定的 $x$，先最大化 $L(x,a)$，再选择 $x$：

$$
\boxed{\min_x\max_{a\ge0}L(x,a).}
$$

**计算顺序是从里往外：先固定 $x$，对 $a$ 做 max；再对 $x$ 做 min。**

### 若违反任意一个约束

假设某个 $f_j(x)>0$，让对应乘子 $a_j\to+\infty$，并把其他乘子取零：

$$
L(x,a)=F(x)+a_j f_j(x)\to+\infty.
$$

所以，**不可行点会得到无穷大的值**，外层的最小化不会选择它。

### 若满足全部约束

此时 $f_i(x)\le0$，任何非负乘子都满足 $L(x,a)\le F(x)$。取全部 $a_i=0$，就能达到 $F(x)$，因此：

$$
\max_{a\ge0}L(x,a)=F(x).
$$

更具体地说：

- 若 $f_i(x)<0$，最大化该项时要取 $a_i=0$，避免出现负值。
- 若 $f_i(x)=0$，该项始终为零，任何 $a_i\ge0$ 都可以。

两种情况合起来：

$$
\boxed{
\max_{a\ge0}L(x,a)=
\begin{cases}
F(x), & f_i(x)\le0\text{ for all }i,\\
+\infty, & \text{otherwise}.
\end{cases}
}
$$

因此，原问题可以等价改写为 **Minimax Primal Formulation**：

$$
\boxed{
p^*=\min_{x:\,f_i(x)\le0}F(x)
=\min_x\max_{a\ge0}L(x,a).
}
$$

**注意：可行点处的 $\max_{a\ge0}L(x,a)=F(x)$，不代表任意固定乘子下都有 $L(x,a)=F(x)$。**

这里形式上消去了对 $x$ 的显式约束，但引入了嵌套优化，而且 $a\ge0$ 仍然是约束。

<a id="dual-formulation"></a>

## 4. Dual Formulation：交换优化顺序

### 从 Minimax 到 Maxmin

Primal 的 Lagrangian 表达为：

$$
\min_x\max_{a\ge0}L(x,a).
$$

把两层优化的顺序交换，得到 **Dual problem（对偶问题）**：

$$
\boxed{\max_{a\ge0}\min_x L(x,a).}
$$

定义 **Dual objective / Dual function（对偶函数）**：

$$
\boxed{G(a)=\inf_x L(x,a).}
$$

当最小值能取到时，可以写成 $G(a)=\min_x L(x,a)$，这也是课件采用的写法。使用 $\inf$ 是为了兼容最小值取不到，或下方无界的情况。

于是 Dual 简化为：

$$
\boxed{d^*=\max_{a\ge0}G(a).}
$$

| 比较项 | Minimax Primal | Maxmin Dual |
|---|---|---|
| 先固定什么 | 固定 $x$ | 固定 $a$ |
| 内层操作 | 对 $a$ 最大化 | 对 $x$ 最小化 |
| 外层操作 | 对 $x$ 最小化 | 对 $a$ 最大化 |
| 希望利用的结构 | 排除不可行点 | 先解内层 min，消去原变量 |

### Weak Duality：对偶值给出下界

**交换 min 与 max 并不自动保持最优值不变。** 始终成立的是：

$$
\boxed{
\max_{a\ge0}\inf_x L(x,a)
\le
\inf_x\sup_{a\ge0}L(x,a),
\qquad d^*\le p^*.
}
$$

这叫 **Weak duality（弱对偶性）**。

一个简短推导：对任意原问题的可行点 $\bar x$，以及任意 $a\ge0$：

$$
G(a)=\inf_x L(x,a)
\le L(\bar x,a)
\le F(\bar x).
$$

所以每个 $G(a)$ 都是原问题最优值的下界；Dual 要找其中最大的下界。

**Weak duality 允许等号成立，不是说一定严格小于。** 当两者最优值有限时，差值 $p^*-d^*\ge0$ 叫 **Duality gap（对偶间隙）**。

<a id="strong-duality"></a>

## 5. Strong Duality：什么时候两者相等

### Strong Duality 的含义

如果 Dual 与 Primal 的最优值相同，就有 **Strong duality（强对偶性）**：

$$
\boxed{d^*=p^*.}
$$

最优值能够取到时，也就是课件中的：

$$
\boxed{
\min_x\max_{a\ge0}L(x,a)
=
\max_{a\ge0}\min_x L(x,a).
}
$$

**相等的是最优目标值。** 两个问题的优化变量不同，$x^*$ 和 $a^*$ 不是同一种东西。

### 本讲使用的充分条件

对于这里仅有不等式约束的问题，一个常用保证是：

1. $F(x)$ 是凸函数（Convex）。
2. 每个 $f_i(x)$ 都是凸函数。
3. 满足 **Slater's condition（Slater 条件）**：在函数定义域内存在一个严格可行点 $\tilde x$，使所有约束同时严格成立：

$$
f_i(\tilde x)<0,\qquad i=1,\ldots,N.
$$

注意，**是同一个点满足全部严格不等式**，而不是每个约束分别找到不同的点。严格可行点也不必是最优点；最优点可以在边界上。

对固定的 $a\ge0$，$L(x,a)$ 是凸函数的非负加权和，因此关于 $x$ 是凸的。对固定的 $x$，$L(x,a)$ 关于 $a$ 是仿射函数（Affine），因而既凸又凹（Concave）。

Slater 条件是保证强对偶的**充分条件之一**；没有找到严格可行点，不等于已经证明强对偶不成立。

### Saddle Point：一个方向最小，另一个方向最大

课件用鞍面解释 Minimax 与 Maxmin 相等。若存在 Lagrangian 鞍点 $(x^*,a^*)$，则：

$$
\boxed{
L(x^*,a)\le L(x^*,a^*)\le L(x,a^*),
\qquad \forall x,\ a\ge0.
}
$$

- 固定 $a^*$，$x^*$ 使 $L$ 最小，对应右侧不等式。
- 固定 $x^*$，$a^*$ 使 $L$ 最大，对应左侧不等式。

这里的鞍点位于 $(x,a)$ 的联合空间；并不是说原目标函数 $F(x)$ 本身出现了鞍点。

<a id="kkt-conditions"></a>

## 6. KKT Conditions：四组最优性条件

### 从无约束最优性到有约束最优性

对可微凸函数，无约束问题的全局最优性条件是：

$$
\nabla F(x^*)=0.
$$

有约束时，最优点可能位于边界，$\nabla F(x^*)$ 不一定为零，因此需要 **Karush-Kuhn-Tucker（KKT）conditions**。

以下考虑可微的凸目标与凸不等式约束。在 Slater 等适当约束资格条件下，原问题最优解可以通过下面四组条件刻画；对凸问题，找到满足它们的一对 $(x^*,a^*)$，就能确认其分别是 Primal 和 Dual 的全局最优解。

### ① Stationarity：驻点条件

$$
\boxed{
\nabla_x L(x^*,a^*)
=\nabla F(x^*)+
\sum_{i=1}^{N}a_i^*\nabla f_i(x^*)=0.
}
$$

含义：固定乘子 $a^*$ 后，$x^*$ 要最小化内层的 $L$。求导对象是 **$L$ 对原变量 $x$ 的梯度**。

### ② Complementary Slackness：互补松弛

$$
\boxed{a_i^*f_i(x^*)=0,\qquad i=1,\ldots,N.}
$$

每个约束都满足“乘子”和“约束值”的乘积为零：

| 已知情况 | 可以推出 | 直观含义 |
|---|---|---|
| $f_i(x^*)<0$ | $a_i^*=0$ | 约束有余量，该乘子必须为零 |
| $a_i^*>0$ | $f_i(x^*)=0$ | 非零乘子只能对应边界上的约束 |
| $f_i(x^*)=0$ | 允许 $a_i^*=0$，也允许 $a_i^*>0$ | 在边界上不保证乘子非零 |
| $a_i^*=0$ | 单靠乘积为零，不能推出 $f_i(x^*)=0$ | 仍需另外检查原始可行性 |

$f_i(x^*)=0$ 称为 **Active constraint（活跃约束）**；$f_i(x^*)<0$ 称为 **Inactive constraint（非活跃约束）**。

最容易记住的两个单向关系是：

$$
\boxed{
f_i(x^*)<0\Rightarrow a_i^*=0,
\qquad
a_i^*>0\Rightarrow f_i(x^*)=0.
}
$$

**上面两条都不能直接反过来。**

### ③ Primal Feasibility：原始可行性

$$
\boxed{f_i(x^*)\le0,\qquad i=1,\ldots,N.}
$$

原变量必须满足所有原始约束。

### ④ Dual Feasibility：对偶可行性

$$
\boxed{a_i^*\ge0,\qquad i=1,\ldots,N.}
$$

乘子必须满足非负约束。

**四组条件缺一不可。** 只令 $\nabla_x L=0$，还不能直接认定得到了原问题的最优解；也不能把 KKT 替换成“对 $x$ 和 $a$ 的所有偏导都等于零”。乘子有 $a\ge0$ 的边界限制，其最优性需要结合可行性和互补松弛判断。

<a id="eliminate-primal-variables"></a>

## 7. 用 Stationarity 消去原变量

Dual 的形式是：

$$
\max_{a\ge0}\underbrace{\min_x L(x,a)}_{G(a)}.
$$

我们希望先解掉内层最小化，使外层只剩 $a$。

**第一步：固定 $a$，对 $x$ 求导。**

$$
\nabla_xL(x,a)=0.
$$

**第二步：解出内层最小点。** 如果能够用乘子表示它，就写成：

$$
x=g(a).
$$

这里的 $g(a)$ 是一个向量函数，表示“给定乘子时，哪个 $x$ 最小化 $L$”。它不是训练出来的分类器。

**第三步：代回完整的 Lagrangian。**

$$
\boxed{
G(a)=L(g(a),a)
=F(g(a))+\sum_{i=1}^{N}a_i f_i(g(a)).
}
$$

**第四步：解剩下的对偶问题。**

$$
\max_{a\ge0}G(a).
$$

若已知强对偶成立，可结合 KKT 从 $a^*$ 恢复原问题的 $x^*$。

注意三点：

- **代回 $L$，不能只代回 $F$**，否则会漏掉约束项。
- 固定 $a\ge0$ 时，若 $L$ 关于 $x$ 可微且凸，找到驻点就找到了全局最小点，但它不一定唯一。
- 不是所有内层问题都有闭式解。若某些 $a$ 使内层下方无界，则 $G(a)=-\infty$；推导时也可能由此产生额外的对偶可行性条件。

另一个有用性质是：**$G(a)$ 总是凹函数**，即使原问题并不凸；因为它是关于 $a$ 的一族仿射函数的逐点下确界。不过，这不代表任何情况下都一定容易求解。

<a id="worked-example"></a>

## 8. 完整例题：两个变量、一个约束

### 题目与符号

课件给出的 Lagrangian 为：

$$
L(x,a)=x_1^2+\frac12x_2^2+10a_1x_1-4,
\qquad a_1\ge0.
$$

这里有两个原变量 $x_1,x_2$，以及一个乘子 $a_1$。**最后一项是常数 $-4$，不是 $-4a_1$**；下面严格按这个公式求解。

它可以对应以下原问题：

$$
\min_{x_1,x_2}
\left(x_1^2+\frac12x_2^2-4\right)
\quad\text{subject to}\quad 10x_1\le0.
$$

### 第一步：Stationarity

固定 $a_1$，对两个原变量分别求偏导：

$$
\nabla_xL(x,a)=
\begin{bmatrix}
2x_1+10a_1\\
x_2
\end{bmatrix}
=\begin{bmatrix}0\\0\end{bmatrix}.
$$

所以：

$$
\boxed{x_1=-5a_1,\qquad x_2=0.}
$$

关于 $x$ 的 Hessian 是：

$$
\nabla_x^2L=
\begin{bmatrix}2&0\\0&1\end{bmatrix}.
$$

它是正定矩阵，因此上述驻点确实是固定 $a_1$ 时唯一的全局最小点。

### 第二步：代回 Lagrangian

$$
\begin{aligned}
G(a_1)
&=L((-5a_1,0),a_1)\\
&=(-5a_1)^2+\frac12(0)^2
+10a_1(-5a_1)-4\\
&=25a_1^2-50a_1^2-4\\
&=\boxed{-25a_1^2-4}.
\end{aligned}
$$

于是 Dual problem 变成：

$$
\boxed{\max_{a_1\ge0}\left(-25a_1^2-4\right).}
$$

### 第三步：求对偶最优解

因为 $a_1^2\ge0$：

$$
-25a_1^2-4\le-4.
$$

在 $a_1^*=0$ 时达到最大值，因此：

$$
\boxed{a_1^*=0,\qquad d^*=-4.}
$$

代回 $x_1=-5a_1$、$x_2=0$：

$$
\boxed{x^*=(0,0)^T,\qquad p^*=-4.}
$$

### 第四步：检查强对偶与 KKT

原目标是凸函数，约束 $10x_1\le0$ 是仿射约束。取 $\tilde x=(-1,0)^T$，有 $10\tilde x_1=-10<0$，满足 Slater 条件，因此强对偶成立。

对 $(x^*,a_1^*)=((0,0)^T,0)$，四组 KKT 条件也全部成立：

| 条件 | 代入检查 |
|---|---|
| Stationarity | $2(0)+10(0)=0$，且 $x_2=0$ |
| Complementary slackness | $a_1^*(10x_1^*)=0\times0=0$ |
| Primal feasibility | $10x_1^*=0\le0$ |
| Dual feasibility | $a_1^*=0\ge0$ |

这个例子还说明：**约束可以恰好位于边界，但对应乘子仍然为零。** 因此，“Active constraint 一定对应正乘子”是错误的。

<a id="revision"></a>

## 9. 核心公式与易错点

### 本讲的完整路线

先将所有约束写成 $f_i(x)\le0$，构造 Lagrangian；再用 Minimax 表达原问题，交换顺序得到 Dual；最后利用 Stationarity 消去原变量，并结合强对偶和其余 KKT 条件确认最优解。

| 名称 | 核心公式 |
|---|---|
| Primal problem | $\min_x F(x)$，满足 $f_i(x)\le0$ |
| Lagrangian | $L(x,a)=F(x)+\sum_i a_i f_i(x)$，$a_i\ge0$ |
| Minimax Primal | $\min_x\max_{a\ge0}L(x,a)$ |
| Dual function | $G(a)=\inf_xL(x,a)$ |
| Maxmin Dual | $\max_{a\ge0}G(a)$ |
| Weak duality | $d^*\le p^*$ |
| Strong duality | $d^*=p^*$ |
| Stationarity | $\nabla_xL(x^*,a^*)=0$ |
| Complementary slackness | $a_i^*f_i(x^*)=0$ |
| Primal / Dual feasibility | $f_i(x^*)\le0$，$a_i^*\ge0$ |

### Primal 与 Dual 优化的变量不同

若原变量 $x\in\mathbb R^d$，共有 $N$ 个不等式约束：

- Primal 优化的是 $d$ 维原变量 $x$。
- Dual 优化的是 $N$ 维乘子 $a$。
- 在 SVM 中，每个训练样本产生一个间隔约束，因此每个样本对应一个乘子。

**Dual 不一定维度更低，也不一定更容易解。** 它的价值在于可能暴露更好用的结构，而不只是减少变量个数。

### 重点检查的六件事

1. **符号配套**：本讲采用 $f_i(x)\le0$、$a_i\ge0$，以及 $L=F+\sum_i a_i f_i$。
2. **固定乘子不保证可行**：Lagrange relaxation 不能直接当成与原问题等价的求解。
3. **分清优化顺序**：Minimax 内层是 max，Maxmin 内层是 min；不能随意交换并声称相等。
4. **分清弱对偶和强对偶**：前者总给出下界，后者表示下界达到原问题最优值。
5. **互补松弛不可乱反推**：$a_i>0$ 推出约束在边界，但约束在边界不保证 $a_i>0$。
6. **消元后仍需检查条件**：Stationarity 用于求内层最小点；最终答案还要满足其余 KKT 条件。

下一讲会把这套方法应用于 SVM，推导它的具体 Dual formulation。
