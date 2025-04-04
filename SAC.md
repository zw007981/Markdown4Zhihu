# SAC(Soft Actor-Critic)

<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [SAC(Soft Actor-Critic)](#sacsoft-actor-critic)
  - [1. 熵的引入](#1-熵的引入)
  - [2. Critic更新](#2-critic更新)
  - [3. Actor更新](#3-actor更新)
    - [3.1. Lemma\_Soft Policy Improvement](#31-lemma_soft-policy-improvement)
  - [4. 工程优化](#4-工程优化)
    - [4.1. 自适应温度参数调整](#41-自适应温度参数调整)
    - [4.2. Critic网络优化](#42-critic网络优化)

<!-- /TOC -->

代码见：[SAC](https://github.com/zw007981/BasicRLAlgo/blob/master/src/algo/SAC.py)。

## 1. 熵的引入
在常规的策略梯度方法中，我们的目标是最大化这样的一个累积折扣奖励：

```math
J(\pi) = \mathbb{E}_{s \sim \mu, \tau \sim p_{\pi}(\cdot | s_0 = s)} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
```

其中 $\mu$ 是初始状态分布，$\tau$ 是在给定初始状态 $s_0$ 的情况下策略 $\pi$ 的轨迹分布。$\gamma$ 是折扣因子，$r(s_t, a_t)$ 是在状态 $s_t$ 下采取动作 $a_t$ 从环境中获得的奖励。

而在SAC中我们的目标函数中额外加入了一个熵项：

```math
\begin{align*}
J(\pi) =& \mathbb{E}_{s \sim \mu, \tau \sim p_{\pi}(\cdot | s_0 = s)} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]\\
=& \mathbb{E}_{s \sim \mu, \tau \sim p_{\pi}(\cdot | s_0 = s)} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) - \alpha \log \pi(a_t|s_t) \right) \right]
\end{align*}
```

其中 $\alpha$ 是一个大于0的温度参数，用于调整熵项对奖励的相对重要性。熵项的引入有一些好处：鼓励探索；增强策略的鲁棒性。在引入熵项后我们的价值函数变为了：

```math
V^{\pi}(s) = \mathbb{E}_{\tau \sim p_{\pi}(\cdot | s_0 = s)} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) - \alpha \log \pi(a_t|s_t) \right) \right]
```

接下来我们可以定义在这种场景下的Q值函数，注意在这里我们只考虑了执行动作$a$后直接从环境中获得的奖励和执行后的状态的期望价值，而没有考虑动作$a$本身的熵：

```math
Q^{\pi}(s, a) \stackrel{\triangle}{=} r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot | s, a)} \left[ V^{\pi}(s') \right]
```

接下来我们推导$V^{\pi}(s)$和$Q^{\pi}(s, a)$之间的关系，为之后的证明做准备：

```math
\begin{align*}
V^{\pi}(s)
=& \mathbb{E}_{\tau \sim p_{\pi}(\cdot | s_0 = s)} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) - \alpha \log \pi(a_t|s_t) \right) \right] \text{, (把$t=0$时的奖励和熵与其他项分开)}\\
=& \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ r(s, a) - \alpha \log \pi(a|s) \right]\\
&+ \gamma \cdot \mathbb{E}_{a \sim \pi(\cdot|s), s' \sim p(\cdot | s, a), \tau \sim p_{\pi}(\cdot | s_0 = s')} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) - \alpha \log \pi(a_t|s_t) \right) \right]\\
=& \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ r(s, a) - \alpha \log \pi(a|s) \right] + \gamma \cdot \mathbb{E}_{a \sim \pi(\cdot|s), s' \sim p(\cdot | s, a)} \left[ V^{\pi}(s') \right]\\
=& \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \underbrace{r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot | s, a)} \left[ V^{\pi}(s') \right]}_{Q^{\pi}(s, a)} - \alpha \log \pi(a|s) \right]\\
=& \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ Q^{\pi}(s, a) - \alpha \log \pi(a|s) \right]
\end{align*}
```

## 2. Critic更新

这个问题中我们用Q值神经网络充当Critic，并利用时序差分的方法来更新网络的参数。我们首先定义这个问题中针对状态-动作对$(s,a)$的TD目标：

```math
\begin{align*}
Q^{\text{target}}(s, a) \stackrel{\triangle}{=}& r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot | s, a)} \left[ V^{\pi}(s') \right]\\
=& r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot | s, a), a' \sim \pi(\cdot|s')} \left[ Q^{\pi}(s', a') - \alpha \log \pi(a'|s') \right]
\end{align*}
```

接下来基于均方误差来定义损失函数：

```math
critic\_loss = MSELoss(Q^{\text{target}}(s, a), Q^{\pi}(s, a))
```

在代码中实现中如果我们通过与环境交互获取到了一组数据$(s, a, r, s\_prime)$，可以在PyTorch框架下利用下面的伪代码来更新Critic的参数：

```python
a_prime, a_prime_prob = policy_net(s_prime)
td_target = r + GAMMA * (q_net(s_prime, a_prime) - (alpha * log(a_prime_prob)))
critic_loss = torch.nn.MSELoss(td_target, q_net(s, a))
optimizer.zero_grad()
critic_loss.backward()
optimizer.step()
```

可以看到这里用于生成样本数据的行为策略和用于更新网络参数的目标策略可以是不同的，因此SAC算法是一个off-policy算法，这也意味着它可以更充分地利用历史数据。

## 3. Actor更新

在SAC算法中我们用来更新策略的底层原理如下：

```math
\begin{align*}
\pi_{new} =& \arg \min_{\pi \in \Pi} D_{KL} \left[ \pi(\cdot | s) || \frac{\exp(\frac{1}{\alpha} Q^{\pi_{old}}(s, \cdot))}{Z^{\pi_{old}}(s)} \right]\\
=& \arg \min_{\pi \in \Pi} \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \log \pi(a|s) - \frac{1}{\alpha} Q^{\pi_{old}}(s, a) + \log Z^{\pi_{old}}(s) \right]
\end{align*}
```

其中$D_{KL}$用于计算两个分布之间的KL散度：$D_{KL}(p||q) = \sum_{x} p(x) [\log p(x) - \log q(x)]$，通过降低KL散度，我们可以鼓励两个分布更加接近。另外，$Z^{\pi_{old}}(s) = \sum_{a} \exp(\frac{1}{\alpha} Q^{\pi_{old}}(s, a))$是归一化因子，用于将Q函数的输出转换为概率分布。接下来我们分别从定性和定量两个角度来解释这个更新方法的合理性。

定性地说，我们先根据旧的策略构造了一个分布，在这个分布中Q值越大的动作被选择的概率越大，接下来我们通过最小化KL散度的方法鼓励新策略去接近这个分布。虽然这种方法依然会鼓励新策略去选择Q值大的动作，但是一方面我们这里用的是概率分布，另一方面引入了熵正则，这使得新策略在选择动作时不会过于极端地去选择Q值最大的动作，而是会在一定程度上进行探索。

接下来我们需要定量地证明这种策略更新方法的合理性，即：

### 3.1. Lemma_Soft Policy Improvement
```math
Q^{\pi_{new}}(s, a) \geq Q^{\pi_{old}}(s, a), \forall (s, a) \in \mathcal{S} \times \mathcal{A}
```

Proof:

注意到在上述更新方法中$Z^{\pi_{old}}(s)$是一个常数项，因此我们可以将其忽略，对于给定状态$s$，记需要最小化的目标函数是$L(\pi)$：

```math
L(\pi) \stackrel{\triangle}{=} \mathbb{E}_{a \sim \pi(\cdot|s)} \left[ \log \pi(a|s) - \frac{1}{\alpha} Q^{\pi_{old}}(s, a) \right]
```

由于我们的目标是最小化这个函数，再考虑到在最差的情况下我们可以令$\pi_{new} = \pi_{old}$，因此我们有$L(\pi_{new}) \leq L(\pi_{old})$，进而有：

```math
\begin{align*}
&\mathbb{E}_{a \sim \pi_{new}(\cdot|s)} \left[ \log \pi_{new}(a|s) - \frac{1}{\alpha} Q^{\pi_{old}}(s, a) \right]
\leq \mathbb{E}_{a \sim \pi_{old}(\cdot|s)} \left[ \log \pi_{old}(a|s) - \frac{1}{\alpha} Q^{\pi_{old}}(s, a) \right]\\
\Rightarrow &\mathbb{E}_{a \sim \pi_{new}(\cdot|s)} \left[ Q^{\pi_{old}}(s, a) - \alpha \log \pi_{new}(a|s) \right]
\geq
\underbrace{\mathbb{E}_{a \sim \pi_{old}(\cdot|s)} \left[ Q^{\pi_{old}}(s, a) - \alpha \log \pi_{old}(a|s) \right]}_{V^{\pi_{old}}(s)}\\
\Rightarrow &\mathbb{E}_{a \sim \pi_{new}(\cdot|s)} \left[ Q^{\pi_{old}}(s, a) - \alpha \log \pi_{new}(a|s) \right] \geq V^{\pi_{old}}(s)\\
\end{align*}
```

接下来基于Q函数的定义展开$Q^{\pi_{old}}(s, a)$：

```math
Q^{\pi_{old}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p(\cdot | s, a)} \left[ V^{\pi_{old}}(s') \right]
```

将上面关于$V^{\pi_{old}}(s)$的不等式带入（这里为了节省书写空间对概率分布的表示做了简化）：

```math
\begin{align*}
Q^{\pi_{old}}(s, a)
\leq& r(s, a) + \gamma \mathbb{E}_{s', a' \sim \pi_{new}} \left[ Q^{\pi_{old}}(s',a') - \alpha \log \pi_{new}(a'|s') \right]\\
=& r(s, a) + \gamma \alpha \mathbb{E}_{s'} \left[ \mathcal{H}(\pi_{new}(\cdot|s')) \right] + \gamma \mathbb{E}_{s', a' \sim \pi_{new}} \left[ Q^{\pi_{old}}(s', a') \right]\\
\end{align*}
```

注意到可以用同样的方法展开右边的最后一项$Q^{\pi_{old}}(s', a')$：

```math
\begin{align*}
Q^{\pi_{old}}(s, a)
\leq& r(s, a) + \gamma \alpha \mathbb{E}_{s'} \left[ \mathcal{H}(\pi_{new}(\cdot|s')) \right] + \gamma \mathbb{E}_{s', a' \sim \pi_{new}} \left[ Q^{\pi_{old}}(s', a') \right]\\
\leq& r(s, a) + \gamma \alpha \mathbb{E}_{s'} \left[ \mathcal{H}(\pi_{new}(\cdot|s')) \right]\\
&+ \gamma \mathbb{E}_{s', a' \sim \pi_{new}} \left[r(s', a')\right] + \gamma^2 \alpha \mathbb{E}_{s''} \left[ \mathcal{H}(\pi_{new}(\cdot|s'')) \right] + \gamma^2 \mathbb{E}_{s'', a'' \sim \pi_{new}} \left[ Q^{\pi_{old}}(s'', a'') \right]\\
=& r(s, a) + \gamma \mathbb{E}_{s', a' \sim \pi_{new}} \left[r(s', a')\right]\\
&+ \gamma \alpha \mathbb{E}_{s'} \left[ \mathcal{H}(\pi_{new}(\cdot|s')) \right] + \gamma^2 \alpha \mathbb{E}_{s''} \left[ \mathcal{H}(\pi_{new}(\cdot|s'')) \right]\\
&+ \gamma^2 \mathbb{E}_{s'', a'' \sim \pi_{new}} \left[ Q^{\pi_{old}}(s'', a'') \right]\\
\end{align*}
```

重复上述过程，我们可以得到：

```math
\begin{align*}
Q^{\pi_{old}}(s, a) &\leq
\mathbb{E}_{\tau \sim p_{\pi_{new}}(\cdot | s_0 = s, a_0 = a)} \left[ \lim_{T\rightarrow \infty} \{
\sum_{t=0}^T \gamma^t r(s_t, a_t) + \alpha \sum_{t=1}^T \gamma^t \mathcal{H}(\pi_{new}(\cdot|s_t))
+ \gamma^T Q^{\pi_{old}}(s_T, a_T)\} \right]\\
&\approx \mathbb{E}_{\tau \sim p_{\pi_{new}}(\cdot | s_0 = s, a_0 = a)} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) + \alpha \sum_{t=1}^{\infty} \gamma^t \mathcal{H}(\pi_{new}(\cdot|s_t)) \right]\\
&= Q^{\pi_{new}}(s, a)
\end{align*}
```

在此基础上，我们用$Q(s,a|\theta)$来表示Q神经网络，用$\pi(a|s;\phi)$来表示策略神经网络，则我们需要最小化的目标函数如下，在代码中直接利用梯度下降法来更新策略神经网络的参数即可：

```math
L_{\pi}(\phi) = \mathbb{E}_{s \sim \rho_{\pi}(\cdot), a \sim \pi(\cdot|s)} \left[ \log \pi(a|s;\phi) - \frac{1}{\alpha} Q(s, a|\theta) \right]
```

当然，为了避免做除法，我们也可以选择最小化下面这个等价的目标函数：

```math
L_{\pi}(\phi) = \mathbb{E}_{s \sim \rho_{\pi}(\cdot), a \sim \pi(\cdot|s)} \left[ \alpha \log \pi(a|s;\phi) - Q(s, a|\theta) \right]
```

## 4. 工程优化

上面是理论推导部分，在实际工程中我们还会做以下的一些优化。

### 4.1. 自适应温度参数调整

这里的$\alpha$是一个重要的超参数，我们可以通过自适应的方法来更好地调整这个参数：在一个陌生的环境中我们可以通过增大熵的权重来增强探索，而在一个熟悉的环境中，可以通过减小熵的权重来更好地利用已有经验。我们先考虑下面这样一个带约束的优化问题：

```math
\max_{\pi_0, \pi_1, \ldots, \pi_T} \mathbb{E}\left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right], \text{ s.t. } \forall t, \mathcal{H}(\pi_t(\cdot|s_t)) \geq \mathcal{H}_{0}
```

其中$\mathcal{H}_{0}$是一个常数，表示我们的期望的熵的最小值。从字面意义上来说每一步我们都需要在保证熵不低于$\mathcal{H}_{0}$的情况下通过调整策略来获得最大化累计折扣奖励。为简化问题，我们设$\gamma = 1.0$并针对每一个时间步$t$定义如下的函数：

```math
\begin{align*}
&f(\pi_t) \stackrel{\triangle}{=} \mathbb{E}_{s \sim \rho_{\pi_{t-1}}(\cdot), a \sim \pi_t(\cdot|s)} \left[ r(s, a) \right]\\
&h(\pi_t) \stackrel{\triangle}{=} \mathbb{E}_{s \sim \rho_{\pi_{t-1}}(\cdot), a \sim \pi_t(\cdot|s)} \left[ \mathcal{H}(\pi_t(\cdot|s)) \right] - \mathcal{H}_{0}\\
\end{align*}
```

则原问题可以转化为，在每一个时间步$t$：

```math
\max f(\pi_t), \text{ s.t. } h(\pi_t) \geq 0
```

考虑拉格朗日乘数法，我们可以通过引入不小于0的拉格朗日乘子$\alpha_t$来将这个带约束的优化问题转化为无约束的优化问题：

```math
L(\pi_t, \alpha_t) \stackrel{\triangle}{=} f(\pi_t) + \alpha_t h(\pi_t)
```

在此基础上，我们假设原问题具有强对偶性（Strong Duality），现在考虑原问题的对偶问题。固定$t$时刻的策略$\pi_t$，把$\alpha_t h(\pi_t)$看作在违反约束条件时的惩罚项：
- 1. 若约束条件被满足，则无需加入惩罚，惩罚项可以直接取0；
- 2. 若约束条件未被满足，考虑到这个时候策略的熵小于预设值$h(\pi_t) \leq 0$，我们需要增大惩罚来让这个时候的$L$尽可能的差从而避免陷入这种情况，这个时候可以让$\alpha_t \rightarrow \infty$，从而使得惩罚项$\alpha_t h(\pi_t)$趋向于负无穷大。

综上所述在固定$\pi_t$的情况下，我们可以得到：

```math
f(\pi_t) = \min_{\alpha_t \geq 0} L(\pi_t, \alpha_t)
```

这样我们可以得到原问题的对偶问题，在满足强对偶性的情况下，下面这个问题和原问题是等价的：

```math
\min_{\alpha_t \geq 0} \max_{\pi_t} L(\pi_t, \alpha_t)
```

接着我们根据原函数的定义展开上式：

```math
\begin{align*}
\min_{\alpha_t \geq 0} \max_{\pi_t} L(\pi_t, \alpha_t)
=& \min_{\alpha_t \geq 0} \max_{\pi_t} \left[ f(\pi_t) + \alpha_t h(\pi_t) \right]\\
=& \min_{\alpha_t \geq 0} \max_{\pi_t} \mathbb{E}_{s \sim \rho_{\pi_{t-1}}(\cdot), a \sim \pi_t(\cdot|s)} \left[ r(s, a) + \alpha_t \mathcal{H}(\pi_t(\cdot|s)) - \alpha_t \mathcal{H}_{0} \right]\\
\end{align*}
```

理论上来说我们可以采用迭代求解的思路来解决这个问题：先固定外层的$\alpha_t$，优化策略以达到最大化内层函数的目的；然后再固定策略$\pi_t$，优化$\alpha_t$以最小化内层函数。因为我们这里主要讨论的是$\alpha$的自适应调整问题，我们聚焦于在固定策略$\pi_t$的情况下，如何调整$\alpha_t$。可以注意到内层函数中的$r(s,a)$是不受$\alpha_t$影响的，因此我们最小化如下的目标函数即可：

```math
J(\alpha)
= \mathbb{E}_{s \sim \rho_{\pi}(\cdot), a \sim \pi(\cdot|s)} \left[ \alpha \mathcal{H}(\pi_t(\cdot|s)) - \alpha \mathcal{H}_{0} \right]
= \mathbb{E}_{s \sim \rho_{\pi}(\cdot), a \sim \pi(\cdot|s)} \left[ - \alpha \log \pi(a|s) - \alpha \mathcal{H}_{0} \right]
```

如果我们基于PyTorch框架进行开发，那么在计算出上述的目标函数后，直接让PyTorch的优化器来最小化这个目标函数即可。现在让我们回过头来定性地分析一下这个调整$\alpha$的过程：如果当前策略的熵小于预设值，这说明策略分布过于集中，通过最小化$J(\alpha)$，我们可以增大$\alpha$来增大熵的权重，从而增强探索；反之，如果当前策略的熵大于预设值，那么通过最小化$J(\alpha)$，我们可以减小$\alpha$来减小熵的权重，从而更好地利用已有经验。

### 4.2. Critic网络优化

针对因为倾向选择Q值最大的动作带来的对Q值的过度估计问题，我们可以引入双Q网络并取两网络输出的最小值作为对Q值的估计来减小这种估计偏差；针对时序差分方法中常见的因为自举（bootstrapping）导致的误差不断传播的问题，我们可以引入目标网络来截断这种传播（这里推荐张志华老师的《深度强化学习》一书，书中针对这两种优化方法的原理有清晰的解释）。