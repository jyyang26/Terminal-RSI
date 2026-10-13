将 Score Centering（SC）引入 PPO，**不需要修改 PPO 经典的截断（Clipping）机制**。在保持 PPO 截断规则的前提下，SC 通过在梯度更新外层执行**加性中心化（Additive Centering）**，消除截断破坏重要性采样无偏性后引入的系统性蒸馏漂移。

---

### 1. PPO 的原始目标与有效梯度形式

在 [PPO 论文](https://arxiv.org/pdf/1707.06347?utm_source=gemini) 中，核心的 Clipped Surrogate Objective 为式 (7)：

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中重要性采样比率定义为 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$。

在大语言模型（LLM RL）语境下：

* $\pi_{\theta_{\text{old}}}$ 对应采样推理引擎分布 $q(y_t \mid y_{<t})$。
* $\pi_\theta$ 对应训练引擎分布 $p_\theta(y_t \mid y_{<t})$。
* $r_t(\theta) = \frac{p_\theta(y_t \mid y_{<t})}{q(y_t \mid y_{<t})}$。
* 优势 $\hat{A}_t$ 对应奖励或优势标量 $R$。

根据自动微分机制，对 $L^{\text{CLIP}}(\theta)$ 求导得到的策略梯度等价于对每个 token 乘以一个**有效权重** $\bar{w}(y_t)$：

$$g_{\text{PPO}} = \bar{w}(y_t) R \nabla_\theta \log p_\theta(y_t \mid y_{<t})$$

在 PPO 中，$\bar{w}(y_t)$ 由截断区间决定：

* 当目标处于未截断区域（或主流 LLM 框架直接使用截断比率）时，$\bar{w}(y_t) = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$。
* 当利用原生 min-clip 算子且梯度在平坦区被遮蔽时，对应 $\bar{w}(y_t) = 0$。

---

### 2. 为什么 PPO 截断会重新引入“漂移（Drift）”？

在理想的重要性采样（IS）下，如果使用未截断的原始比率 $w(y_t) = \frac{p_\theta(y_t)}{q(y_t)}$，期望梯度是严格无偏的：

$$\mathbb{E}_{y_t \sim q} [w(y_t) \nabla_\theta \log p_\theta(y_t)] = \sum_{v \in \mathcal{V}} q(v) \frac{p_\theta(v)}{q(v)} \nabla_\theta \log p_\theta(v) = \sum_{v \in \mathcal{V}} \nabla_\theta p_\theta(v) = 0$$

然而，为了防止方差爆炸，**PPO 对比率进行了截断（或置零屏蔽）**。一旦比率被替换为截断后的 $\bar{w}$，无偏性即刻失效：

$$\mathbb{E}_{y_t \sim q} [\bar{w}(y_t) \nabla_\theta \log p_\theta(y_t)] \neq 0$$

在 [Score Centering 论文](https://arxiv.org/pdf/2609.20807?utm_source=gemini) 的协方差分解框架下：

$$\mathbb{E}_{y_t \sim q}[g_{\text{PPO}}] = \underbrace{\mathbb{E}_{y_t \sim q}[R] \cdot \mathbb{E}_{y_t \sim q}[\bar{w}(y_t) \nabla_\theta \log p_\theta(y_t)]}_{\text{Drift Term（PPO 截断漂移项）}} + \text{Cov}_{y_t \sim q}(R, \bar{w}(y_t) \nabla_\theta \log p_\theta(y_t))$$

当模型存在训练-推理不匹配（TIM，如量化误差、异步陈旧性）且经历多轮迭代时，这个非零漂移项会强行将训练模型拉向采样器分布，导致训练退化与崩溃。

---

### 3. PPO + Score Centering 的梯度设计

Score Centering 的解决方案是：**保留 PPO 的有效权重 $\bar{w}$ 不变，在梯度外层强制减去其期望值**。

定义 token $v$ 的 PPO 加权分值向量为：


$$s'_v \triangleq \bar{w}(v) \nabla_\theta \log p_\theta(v)$$

计算它在采样策略 $q$ 下的期望值（漂移向量）：


$$\bar{s}'_{\text{PPO}} \triangleq \mathbb{E}_{v \sim q}[s'_v] = \sum_{v \in \mathcal{V}} q(v) \bar{w}(v) \nabla_\theta \log p_\theta(v)$$

复合后的 PPO+SC 梯度更新为：

$$g_{\text{PPO+SC}} = R \cdot \left[ \bar{w}(y_t) \nabla_\theta \log p_\theta(y_t) - \bar{s}'_{\text{PPO}} \right]$$

#### 零漂移验证

在采样分布 $q$ 下对该梯度取条件期望：

$$\mathbb{E}_{y_t \sim q}[g_{\text{PPO+SC}}] = R \cdot \left( \sum_{y_t \in \mathcal{V}} q(y_t) \bar{w}(y_t) \nabla_\theta \log p_\theta(y_t) - \bar{s}'_{\text{PPO}} \right) = R \cdot (\bar{s}'_{\text{PPO}} - \bar{s}'_{\text{PPO}}) = 0$$

无论 PPO 的超参数 $\epsilon$ 设为多少、如何截断，加性中心化项都无条件消除了 PPO 截断带来的系统漂移。

---

### 4. Top-$k$ 工程近似化简

全词表 $\vert{}\mathcal{V}\vert{}$（通常 $\ge 150{,}000$）无法直接遍历求和。论文在 Appendix A.2 中给出了结合重要性权重时的 Top-$k$ 解析化简：

1. **头部 Token 集合 $\mathcal{K}$（如 $k=32$ 或 $128$）**：直接计算各自的截断权重 $\bar{w}(v)$。
2. **尾部 Token（$v \notin \mathcal{K}$）**：根据尾部质量重缩放假设 $q(v) \approx \rho \cdot p_\theta(v)$，其中尾部质量比率为：

$$\rho = \frac{1 - \sum_{v \in \mathcal{K}} q(v)}{1 - \sum_{v \in \mathcal{K}} p_\theta(v)}$$



因此所有尾部 token 的重要性比率均退化为标量常数：

$$r_{\text{tail}} = \frac{p_\theta(v)}{q(v)} \approx \frac{1}{\rho}$$



其对应的 PPO 权重也退化为常数标量：

$$\bar{w}_{\text{tail}} \triangleq \bar{w}\left(\frac{1}{\rho}\right) = \text{clip}\left(\frac{1}{\rho}, 1-\epsilon, 1+\epsilon\right)$$


3. **折叠尾部求和**：
利用在策恒等式 $\sum_{v \notin \mathcal{K}} p_\theta(v) \nabla_\theta \log p_\theta(v) = - \sum_{v \in \mathcal{K}} p_\theta(v) \nabla_\theta \log p_\theta(v)$，尾部的漂移项可直接折叠回头部：

$$\sum_{v \notin \mathcal{K}} q(v) \bar{w}(v) \nabla_\theta \log p_\theta(v) \approx - \rho \bar{w}_{\text{tail}} \sum_{v \in \mathcal{K}} p_\theta(v) \nabla_\theta \log p_\theta(v)$$



最终，PPO 的期望漂移向量简化为**仅在 Top-$k$ 个 token 上的加权和**：

$$\bar{s}'_{\text{PPO}} \approx \sum_{v \in \mathcal{K}} \Big( q(v)\bar{w}(v) - \rho \bar{w}_{\text{tail}} p_\theta(v) \Big) \nabla_\theta \log p_\theta(v)$$

---

### 5. 自动微分（Autograd）标量损失函数写法

在 PyTorch 或 JAX 训练框架中，为了让反向传播自动产出上述无漂移梯度，构造如下标量损失函数：

$$\mathcal{L}_{\text{PPO+SC}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) + R \cdot \sum_{v \in \mathcal{K}} \text{sg}\Big( q(v)\bar{w}(v) - \rho \bar{w}_{\text{tail}} p_\theta(v) \Big) \log p_\theta(v)$$

或者采用显式分离写法的单 token 损失（Appendix A.3）：

$$\mathcal{L}_{\text{PPO+SC}}(\theta) = - R \cdot \left[ \text{sg}(\bar{w}_{y_t}) \log p_\theta(y_t) - \sum_{v \in \mathcal{K}} \text{sg}\Big( q(v)\bar{w}(v) - \rho \bar{w}_{\text{tail}} p_\theta(v) \Big) \log p_\theta(v) \right]$$

其中：

* $\text{sg}(\cdot)$ 表示 `stop_gradient`（PyTorch 中的 `.detach()`），确保修正系数仅作为标量权重，梯度纯粹通过 $\log p_\theta$ 回传。
* 前半部分保持原版 PPO 的梯度贡献，后半部分通过对 Top-$k$ 词表 log-prob 的加权回传，以近乎零的显存和算力开销将截断带来的漂移精确消除。