# 4c · SVM Dual 笔记

## Table of Contents

- [1. 本节的问题与符号](#1-本节的问题与符号)
- [2. 从 Primal 到 Maxmin](#2-从-primal-到-maxmin)
- [3. 用 KKT 消去 w 和 b](#3-用-kkt-消去-w-和-b)
- [4. 得到 SVM 的 Dual](#4-得到-svm-的-dual)
- [5. Kernel Trick 与 Polynomial Kernel](#5-kernel-trick-与-polynomial-kernel)
- [6. Support Vectors 与预测](#6-support-vectors-与预测)
- [7. 相似度与 Mercer 条件](#7-相似度与-mercer-条件)
- [8. Kernel 的组合规则](#8-kernel-的组合规则)
- [9. Gaussian 和 RBF Kernel](#9-gaussian-和-rbf-kernel)
- [10. 例题与复习重点](#10-例题与复习重点)

## 1. 本节的问题与符号

本节接着 SVM Primal 和 Lagrange Duality：把直接优化权重的 Primal，改写成优化每个训练样本对应系数的 Dual。改写后会发现： **训练和预测都只需要特征向量之间的内积，因此可以引入 Kernel。**

本节推导的是 **Hard-margin SVM** ，假设训练数据在所选 Feature Space 中线性可分。这里没有 Soft-margin 的松弛变量，也没有参数 C。

| 符号 | 含义 |
| --- | --- |
| `x_n` | 第 n 个训练样本；对应课件的上标编号写法 |
| `y_n` | 第 n 个标签，只取 +1 或 -1 |
| `N` | 训练样本数 |
| `φ(x)` | 特征变换，把输入映射到 Feature Space |
| `w, b` | 权重向量和偏置 |
| `a_n` | 第 n 个约束对应的 Lagrange multiplier；也常写作 α |
| `h(x)` | 分类打分，符号决定预测类别 |

分类打分与 Primal 问题分别是：

$$
h(x)=w^T\phi(x)+b
$$

$$
\min_{w,b}\frac12\|w\|^2
\qquad y_n\bigl(w^T\phi(x_n)+b\bigr)\geq1,\quad n=1,\ldots,N
$$

**几何意义：** 决策边界是 `h(x) = 0`，两条间隔边界是 `h(x) = +1` 和 `h(x) = -1`。在 Feature Space 中，两条间隔边界之间的距离为 `2 / ‖w‖`，因此最小化权重范数等价于最大化间隔。

注意：`h(x)` 是打分；点到决策超平面的几何距离是 `|h(x)| / ‖w‖`。`y_n h(x_n)` 是 Functional Margin，不能直接当成几何距离。

## 2. 从 Primal 到 Maxmin

### Step 1：Lagrange Relaxation

先把所有约束写成“某个表达式 ≤ 0”的形式：

$$
f_n(w,b)=1-y_n\bigl(w^T\phi(x_n)+b\bigr)\leq0
$$

每个约束配一个非负乘子 `a_n ≥ 0`，构造 Lagrangian：

$$
L(w,b,a)=\frac12\|w\|^2+
\sum_{n=1}^{N}a_n\bigl(1-y_n(w^T\phi(x_n)+b)\bigr)
$$

其中 `a = (a_1, …, a_N)`。 **乘子非负，是因为我们采用了“约束 ≤ 0，再加上乘子乘约束”的写法。** 符号必须成套对应。

### Step 2：Minimax Primal

只构造 Lagrangian 还不够；要用下面的形式恢复原约束问题：

$$
\min_{w,b}\max_{a\geq0}L(w,b,a)
$$

阅读顺序是 **先内层 max，再外层 min** 。固定一组 `w, b` 后：

- 若所有约束都满足，每个 `f_n ≤ 0`，内层最大值就是原目标值，因为可以令所有乘子为 0。
- 若某个约束被违反，即 `f_n > 0`，增大对应的 `a_n` 就能使 L 无上界。

因此，外层最小化会排除违反约束的方案。这解释了为什么这里需要内层 **max** 。

### Step 3：Maxmin Dual

交换优化顺序，得到 Dual：

$$
\max_{a\geq0}\min_{w,b}L(w,b,a)
$$

**min 与 max 不能随意交换。** 一般只有 Weak Duality：Dual 最优值不大于 Primal 最优值。本节在凸性和 Slater 条件下满足 Strong Duality，两者最优值相同。

- 固定 `a ≥ 0`，L 对 `w, b` 是凸函数：二次范数项是凸的，其余项对它们是仿射的。
- 固定 `w, b`，L 对 a 是仿射函数，因此也是凹函数。
- **Slater’s Condition：** 存在一组 `w, b`，使所有不等式都严格成立，即所有 `y_n h(x_n) > 1`。

为什么可分的 Hard-margin SVM 满足 Slater 条件？对于有限训练集，先找一个正确分开所有点的超平面，使所有 `y_n h(x_n) > 0`，再把 `w, b` 同时乘以足够大的正数，就能让所有值都大于 1。

注意：Slater 要求 **存在一个严格可行解** ，并不要求最优解处所有约束都严格成立。最优解处，部分点恰好落在间隔边界上。

## 3. 用 KKT 消去 w 和 b

现在先固定 a，求内层的最小值。因为内层对 `w, b` 无约束且是凸问题，可以利用 Stationarity 条件。

### 对 w 求导

$$
\nabla_w L=w-\sum_{n=1}^{N}a_ny_n\phi(x_n)=0
$$

因此：

$$
w=\sum_{n=1}^{N}a_ny_n\phi(x_n)
$$

**关键含义：最优权重是训练样本经过特征变换后的线性组合。** 系数是 `a_n y_n`。这是后面只用训练样本进行预测的原因。

### 对 b 求导

$$
\frac{\partial L}{\partial b}=-\sum_{n=1}^{N}a_ny_n=0
$$

因此，Dual 必须满足额外的等式约束：

$$
\sum_{n=1}^{N}a_ny_n=0
$$

这里没有直接解出 b。该条件会让 L 中与 b 有关的项消失；训练完成后，再利用支持向量求 b。

为什么这个约束不能省？L 中包含 `-b Σ a_n y_n`。如果该求和不为 0，就能沿某个方向改变 b，使内层最小化无下界，Dual function 的值为负无穷。

因为标签是 ±1，这个约束也等价于： **正类乘子的总和 = 负类乘子的总和。** 它不表示两类样本数量相同。

## 4. 得到 SVM 的 Dual

先展开 L，便于观察各项：

$$
L=\frac12\|w\|^2+\sum_{n=1}^{N}a_n
-\sum_{n=1}^{N}a_ny_nw^T\phi(x_n)
-b\sum_{n=1}^{N}a_ny_n
$$

代入上一节的两个结果：最后一项为 0，而第三项中的求和满足：

$$
\sum_{n=1}^{N}a_ny_nw^T\phi(x_n)
=w^T\sum_{n=1}^{N}a_ny_n\phi(x_n)
=w^Tw
$$

所以，`1/2 ‖w‖² - ‖w‖²` 合并为 `-1/2 ‖w‖²`，得到只关于 a 的 Dual function：

$$
G(a)=\sum_{n=1}^{N}a_n-\frac12\|w\|^2
$$

上式中的 w 必须继续替换成 a 的表达式。平方范数展开后：

$$
\|w\|^2=\sum_{n=1}^{N}\sum_{m=1}^{N}
a_na_my_ny_m\phi(x_n)^T\phi(x_m)
$$

这里是 **所有样本对之间的双重求和** ，包含 `n = m` 的情况，不能只保留每个样本与自己的内积。

定义 Kernel：

$$
k(x,z)=\phi(x)^T\phi(z)
$$

**最终的 Hard-margin SVM Dual：**

$$
\max_a G(a)
$$

$$
G(a)=\sum_{n=1}^{N}a_n-\frac12
\sum_{n=1}^{N}\sum_{m=1}^{N}a_na_my_ny_mk(x_n,x_m)
$$

$$
a_n\geq0\quad(n=1,\ldots,N),
\qquad\sum_{n=1}^{N}a_ny_n=0
$$

现在优化的变量只有 `a_1, …, a_N`。`w, b` 已从优化目标中消去，但最终分类器仍然需要它们对应的作用。

补充区分：`max` 给出最大的目标值，`argmax` 给出达到最大值的参数 a。求训练后的乘子时，严格来说是在求 `argmax`。

## 5. Kernel Trick 与 Polynomial Kernel

### 为什么能绕过显式升维

Dual 中没有单独使用 `φ(x)` 的地方，只需要 `φ(x_n)^T φ(x_m)`。如果能直接用原始输入算出这个内积，就无需先构造完整的高维特征向量。这就是 **Kernel Trick** 。

**特征变换仍然存在于数学解释中；计算时可以直接求 Kernel。** 在 Feature Space 中，SVM 的决策边界仍是线性超平面；映回 Input Space 后，边界可以是曲线。

### 二次多项式 Kernel 的推导

对于二维输入，采用下面的特征变换：

$$
\phi(x)=(1,\sqrt2x_1,\sqrt2x_2,x_1^2,x_2^2,\sqrt2x_1x_2)^T
$$

对另一个点 z 做相同变换，两者内积为：

$$
k(x,z)=1+2x_1z_1+2x_2z_2+x_1^2z_1^2+x_2^2z_2^2+2x_1x_2z_1z_2
$$

把它识别成完全平方：

$$
k(x,z)=(1+x_1z_1+x_2z_2)^2=(1+x^Tz)^2
$$

**为什么出现 √2？** 内积会把对应特征相乘，`√2 × √2 = 2`，刚好产生完全平方展开式需要的系数。

如果直接使用未缩放的 `(1, x_1, x_2, x_1², x_2², x_1x_2)`，内积中的这些项系数都是 1，便不是这里的 `(1 + x^Tz)²`。两种变换包含相同种类的单项式，但特征的缩放不同。

**数值核对：** 取 `x = (1, 2)`、`z = (3, 4)`，有 `x^Tz = 11`，因此 Kernel 值为 `12² = 144`。按六维特征逐项相乘，也得到 `1 + 6 + 16 + 9 + 64 + 48 = 144`。

### 推广到 p 次

$$
k(x,z)=(1+x^Tz)^p
$$

这里 p 是正整数。其特征变换包含 **所有总次数不超过 p 的单项式** ，并按展开式系数进行缩放；常数项的次数是 0。

“总次数”要把各变量的指数相加。例如 `x_1²x_2` 的总次数是 3，因此不属于二次特征变换。

### Primal 与 Dual 的计算差别

| 对比 | Primal | Dual |
| --- | --- | --- |
| 优化变量 | d 维的 w，以及一个 b | N 个乘子 a_n |
| 变量数量 | d + 1 | N |
| 特征的使用方式 | 直接使用 φ(x) | 使用样本之间的 Kernel |
| 主要优势 | 特征维数较小时可以直接优化 | 可以隐式处理很高维甚至无限维的特征 |

这里 d 是 **Feature Space 的维数** ，并非原始输入维数。Dual 不保证总是更快；N 很大时，完整训练 Gram matrix 有 N² 个元素。

## 6. Support Vectors 与预测

### 从 Primal 预测式到 Dual 预测式

把 w 的表达式代入 `h(x) = w^Tφ(x) + b`，得到：

$$
h(x)=\sum_{n=1}^{N}a_ny_nk(x_n,x)+b
$$

`h(x) > 0` 预测 +1，`h(x) < 0` 预测 -1。`h(x) = 0` 的点位于决策边界上，需要事先约定如何处理平局。

每个训练样本对预测的贡献，由三项共同决定：乘子 `a_n`、标签 `y_n`、以及与新样本的 Kernel 值。

### 为什么只需要 Support Vectors

若 `a_n = 0`，该点的贡献就是 0；若 `a_n > 0`，该点称为 **Support Vector（支持向量）** 。令 S 为这些点的索引集合：

$$
S=\{n:a_n>0\},
\qquad h(x)=\sum_{n\in S}a_ny_nk(x_n,x)+b
$$

因此，使用 Dual 形式预测时，只需保留支持向量、对应的乘子和标签，以及 b。

### KKT Complementary Slackness

在最优解处，每个样本都满足：

$$
a_n\bigl(1-y_nh(x_n)\bigr)=0
$$

如果 `a_n > 0`，括号必须为 0，因此：

$$
y_nh(x_n)=1
$$

**在本节的 Hard-margin SVM 中，所有支持向量都位于间隔边界上。** 正类支持向量满足 `h(x_n) = +1`，负类支持向量满足 `h(x_n) = -1`。

| 样本情况 | 由 KKT 得到的结论 |
| --- | --- |
| `a_n > 0` | 必须有 `y_n h(x_n) = 1` |
| `y_n h(x_n) > 1` | 必须有 `a_n = 0` |
| `y_n h(x_n) = 1` | 仅凭这个条件，不能保证 `a_n > 0`；退化情形下可以为 0 |

### 怎样计算 b

选一个支持向量，利用 `y_n h(x_n) = 1`。因为 `y_n² = 1`，两边乘以 `y_n` 后，就得到 `h(x_n) = y_n`。所以：

$$
b_n=y_n-\sum_{m\in S}a_my_mk(x_m,x_n)
$$

理论上，各支持向量算出的 b 相同。数值计算时，可以对所有支持向量的结果取平均，提高稳定性。设支持向量数量为 `N_S`：

$$
b=\frac1{N_S}\sum_{n\in S}b_n
$$

**求和索引不要混淆：** 外面选择第 n 个支持向量作为参照；里面用 m 遍历全部支持向量。

### 四类 KKT 条件放在一起看

| 条件 | 在本节中的内容 |
| --- | --- |
| Primal feasibility | 每个样本都满足 `y_n h(x_n) ≥ 1` |
| Dual feasibility | 每个 `a_n ≥ 0` |
| Stationarity | 对 w、b 求导为 0，得到权重表达式与等式约束 |
| Complementary slackness | 每个 `a_n (1 - y_n h(x_n)) = 0` |

## 7. 相似度与 Mercer 条件

### 把 Kernel 看作相似度

预测可以理解成：比较新点与各支持向量的“相似程度”，再用 `a_n y_n` 加权，最后加上偏置。

不过，Kernel 严格来说是 **某个 Feature Space 中的内积** ，不一定满足日常相似度的所有直觉。例如 Linear Kernel：

$$
\phi(x)=x,\qquad k(x,z)=x^Tz
$$

线性内积可以为负，也受向量长度影响；它不等同于欧氏距离，更不必一定处于 0 到 1 之间。

能否直接设计相似度函数，而不先写出 φ？可以，但需要确保它是合法 Kernel。

### Gram Matrix

任取 M 个输入点，构造一个 M × M 矩阵 K：

$$
K_{ij}=k(x_i,x_j)
$$

K 称为 **Gram matrix** 。课件采用的 Mercer 条件要求：对 **任意有限点集** ，K 都对称且 Positive Semidefinite（半正定）。

对称性要求：

$$
k(x_i,x_j)=k(x_j,x_i)
$$

半正定性要求：

$$
c^TKc\geq0\qquad\text{for all }c\in\mathbb{R}^M
$$

这里 c 是任意实向量。若 Kernel 确实来自特征内积，则：

$$
c^TKc=\left\|\sum_{i=1}^{M}c_i\phi(x_i)\right\|^2\geq0
$$

**两点易错：** 半正定不要求每个矩阵元素都非负；只检查当前训练集上的一个 Gram matrix，也不足以证明函数对所有输入都是合法 Kernel。

## 8. Kernel 的组合规则

已知 `k_1` 和 `k_2` 是合法 Kernel，可以用下面的规则构造新 Kernel。

| 新 Kernel | 条件或说明 |
| --- | --- |
| `c k_1(x,z)` | 常数 `c ≥ 0` |
| `f(x) k_1(x,z) f(z)` | f 可以是任意实值函数 |
| `q(k_1(x,z))` | q 是各项系数都非负的多项式 |
| `exp(k_1(x,z))` | 指数展开的系数都非负 |
| `k_1(x,z) + k_2(x,z)` | 两个合法 Kernel 相加 |
| `k_1(x,z) + c` | 常数 `c ≥ 0` |
| `k_1(x,z) k_2(x,z)` | 两个合法 Kernel 逐点相乘 |

其中“逐点相乘”指对同一对输入求两个 Kernel 值再相乘，对应 Gram matrix 的逐元素乘积。

例如，Linear Kernel 合法；加上非负常数 1 仍合法；再取正整数次幂，就得到合法的 Polynomial Kernel。

**系数的非负性不能随意删去。** 例如将合法 Kernel 乘以负数，一般会破坏半正定性。

## 9. Gaussian 和 RBF Kernel

### 定义与直觉

Gaussian Kernel 也称 **Radial Basis Function（RBF）Kernel** ：

$$
k(x,z)=\exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right),\qquad\sigma>0
$$

- 当 `x = z`，Kernel 值为 1。
- 两点距离越远，Kernel 值越接近 0；对于有限距离，其值大于 0。
- σ 越小，相似度随距离下降得越快，影响范围更局部；σ 越大，影响范围更广。

因此，RBF 可以在原始输入空间中产生弯曲的决策边界。σ 会影响边界形状，但仅凭 σ 大小不能断言模型一定过拟合或欠拟合。

### 为什么对应无限维特征

先展开距离平方：

$$
\|x-z\|^2=\|x\|^2+\|z\|^2-2x^Tz
$$

因此：

$$
k(x,z)=
\exp\left(-\frac{\|x\|^2}{2\sigma^2}\right)
\exp\left(\frac{x^Tz}{\sigma^2}\right)
\exp\left(-\frac{\|z\|^2}{2\sigma^2}\right)
$$

中间的指数可以展开为：

$$
\exp\left(\frac{x^Tz}{\sigma^2}\right)
=\sum_{j=0}^{\infty}\frac{(x^Tz)^j}{j!\,\sigma^{2j}}
$$

它包含从 0 次到任意高次的项，因此 Gaussian Kernel 对应的特征表示通常是 **无限维** 的。实际计算仍然只需要原始输入之间的距离，无需构造无限长的向量。

### 用组合规则证明它合法

1. `x^Tz` 是合法的 Linear Kernel。
2. 对每个非负整数 j，`(x^Tz)^j` 合法；j = 0 对应常数 Kernel。
3. 乘以正系数 `1 / (j! σ^(2j))`，仍合法。
4. 有限部分和合法；上述级数逐点收敛，其对应 Gram matrix 的极限仍半正定，因此中间的指数项合法。
5. 两侧分别乘同一个实值函数 `f(x)` 和 `f(z)`，用组合规则即可得到 Gaussian Kernel。

这里取：

$$
f(x)=\exp\left(-\frac{\|x\|^2}{2\sigma^2}\right)
$$

以上保留一般的 σ。若令 `σ = 1`，就得到课件用于说明的简化展开式。

## 10. 例题与复习重点

### 课件中的图想说明什么

课件先展示“一类点在中心、另一类围成外圈”的数据，再比较不同 Kernel。

| Kernel | 图中现象与含义 |
| --- | --- |
| Linear | 原始二维空间的一条直线无法把中心与外圈完全分开 |
| 二次 Polynomial | 能形成封闭的曲线边界，把中心与外围分开 |
| Gaussian / RBF | 能形成灵活的曲线，也能处理后面更复杂的形状 |

图中实线是决策边界，虚线对应间隔边界；圈出的点是支持向量。 **支持向量在间隔边界上，不在 `h(x) = 0` 的决策边界上。**

边界在 Input Space 中弯曲，并不改变 SVM 在 Feature Space 中使用线性超平面这一点。

### 补充手算例题：把 Dual、w、b 和预测串起来

一维训练集有两个点：`x_1 = -1, y_1 = -1`；`x_2 = +1, y_2 = +1`。使用 Linear Kernel，即 `k(x,z) = xz`。

**第一步：利用等式约束。**

$$
-a_1+a_2=0\quad\Rightarrow\quad a_1=a_2=t\geq0
$$

**第二步：写出并最大化 Dual 目标。** 因为两个 `y_n x_n` 都等于 1，双重求和共有四个相同的 `t²` 项：

$$
G(t)=2t-\frac12(4t^2)=2t-2t^2
$$

导数是 `2 - 4t`，所以最优值对应 `t = 1/2`。两个乘子都大于 0，因此两个点都是支持向量。

**第三步：恢复 w。**

$$
w=\frac12(-1)(-1)+\frac12(+1)(+1)=1
$$

**第四步：求 b。** 用正类支持向量，`b = y_2 - w x_2 = 1 - 1 = 0`；用负类点也得到 0。

**第五步：预测。** 最终 `h(x) = x`；正数判为 +1，负数判为 -1，决策边界是 `x = 0`。两个训练点都满足 `y_n h(x_n) = 1`，间隔宽度是 2。

核对 Strong Duality：Primal 目标值为 `1/2 · 1² = 1/2`，Dual 目标值为 `2 · 1/2 - 2 · (1/2)² = 1/2`，两者相同。

### 复习时优先掌握

1. **完整推导链：** Primal 约束标准化 → Lagrangian → Minimax → 在 Strong Duality 下转为 Maxmin → 用 Stationarity 消去 w、b → 得到 Dual。
2. **两个关键结果：** w 是 `a_n y_n φ(x_n)` 的求和；乘子必须满足 `Σ a_n y_n = 0`。
3. **Dual 目标与约束：** 线性求和项减去二分之一的双重求和，另外有非负约束和一个等式约束。
4. **Kernel Trick：** 直接算特征内积；二次 Polynomial 的 √2 缩放是展开系数的来源。
5. **支持向量：** 非零乘子决定哪些点进入预测；用 Complementary Slackness 推出间隔等式，再求 b。
6. **Kernel 合法性：** 任意有限点集上的 Gram matrix 对称、半正定；能运用组合规则证明 RBF 合法。

最后检查易错点： **不要漏 Dual 的等式约束；不要把支持向量放到决策边界上；不要把任意相似度都当成 Kernel；不要把“Dual 可处理高维”理解成“Dual 总是更快”。**
