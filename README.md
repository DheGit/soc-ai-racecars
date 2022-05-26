# soc-ai-racecars
The following contains the notes related to Reinforcement Learning from the YouTube playlist by David Silver. The playlist contains all the relevant info about RL that is required in this SoC project.

# Markov Process
A process which has the property: *The future is independent of the past given the present*
Mathematically, 
$$
\mathbb{P}[S_{t+1}\space|\space S_t]=\mathbb{P}[S_{t+1}\space|\space S_1,\cdots,S_t]
$$

That is, for the purpose of decision making, the past can be simply thrown away.
Example: Chess

# State Transition Matrix
The state transition matrix $\mathcal{P}$ defines transition probabilities for every pair of current and next state.
Let the probability of transition $s$ to $s'$ be denoted as $P_{ss'}$
Then,
$$
\begin{bmatrix}
P_{11} & \cdots & P_{1n} \\
\vdots & \ddots &  \vdots \\
P_{n1} & \cdots & P_{nn} \\
\end{bmatrix}
$$


Thus, a given markov process can be completely defined by the tuple $\braket{\mathcal{S},\mathcal{P}}$ where $\mathcal{S}$ denotes the finite set of all possible states.

# Markov Reward Process
In addition to simple markov process, the markov **reward** process involves reward $\mathcal{R}$ and discount factor $\gamma$ as well.
Thus, a markov reward process is characterised by $\braket{\mathcal{S},\mathcal{P},\mathcal{R},\gamma}$

**Note**: It will be helpful to think of both $\mathcal{S}$ and $\mathcal{R}$ as column vectors containing the state and corresponding reward in a column vector.

**Q. But why use the discount factor?**
A. Because, our model may not be correct, in which case the future is probably not what we expect it to be. So, discount factor sort of denotes our trust in the model too. It may also be something that models instant gratification nature of humans (\* shrugs \*)

## Return ($\bf G_t$)
Definition: Return is the total discounted reward that an agent recieves from time $t=t+1$ to $t=\infty$
Mathematically, return $G_t$ is defined as
$$
G_t=\sum_{k=0}^\infty \gamma^kR_{t+k+1}
$$

## Value function
Represents the return that can be expected to be recieved once a given state $\mathcal{s}$ is reached.
$$
V(s)=\mathbb{E}[G_t\space |\space S_t=s]
$$

Denote $\mathcal{v}$ as a column vector containing the value function of each corresponding state in $S$

## Bellman Equation for the MRP
Analysing the value function, we see
$$
\begin{flalign}
V(s)&=\mathbb{E}[G_t\space |\space S_t=s] \\\\
&=\mathbb{E}[R_{t+1}+\gamma R_{t+2}+\cdots\space |\space S_t=s] \\\\
&=\mathbb{E}[R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots)\space |\space S_t=s] \\ \\
&=\mathbb{E}[R_{t+1}+\gamma(G_{t+1})\space |\space S_t=s] \\ \\
&=\mathbb{E}[R_{t+1}+\gamma V(S_{t+1}) \space |\space S_t=s] \\ \\
\end{flalign}
$$

### In the matrix form
$\mathcal{v=R+\gamma Pv}$

#homework: Verify this

So, well, we can just find the value function, by solving the equation, right? Right, but it can take a lot of time. So, we'll find faster and more realistic ways to compute the value function.

# Markov Decision Process
Has everything that the reward process does, but also includes the notion of **action**
That is, the behaviour of the agent in a given state is also required to define this process.

$\braket{\mathcal{S,A,R,P},\gamma}$ thus defines this process.

**Modifications**: Now, transition probability depends on action taken too. Also, in general, reward function depends on the action taken too.


## Policy ($\pi$)
This is what fully defines the behaviour of an agent.
A policy $\pi$ is a distribution of actions over states:
$$
\pi(a|s)=\mathbb{P}[A_t=a\space|\space S_t=s]
$$

## Amending the functions using the policy
When the policy is introduced, the value function is also dependent on it. That is, $v_\pi(s)$ denotes the expected return starting from state $s$ and then following the policy $\pi$

Similarly, the value of an state-action pair $q_\pi(s,a)$ is defined as the expected return if an agent starts from a state $s$, takes the action $a$ and then continues following the policy $\pi$

Notice that $v_\pi(s)$ can be calculated from $q_\pi(s,a)$ and $\pi$ as
$$
v_\pi(s)=\sum_{a\in A}\pi(a|s)q_\pi(s,a)
$$
Also, if we know the environment's transition probability distribution, $q_\pi(s,a)$ can be calculated as
$$
q_\pi(s,a)=\mathcal{R}_s^a+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_\pi(s')
$$

But wait, that means we can find $v_\pi(s)$ using $v_\pi(s)$ as:
$$
v_\pi(s)=\sum_{a\in A}\pi(a|s)\left(\mathcal{R}_s^a+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_\pi(s')\right)
$$
This is the Bellman equation for the MDP

# Policy Optimisation
To find an optimal policy, let us first define a partial ordering over policies:
$$
\pi \geq \pi' \space\text{ if } v_\pi(s)\geq v_{\pi'},\text{ } \forall s
$$

For every Markov Decision Process:
1) There exists an optimal policy $\pi_\star$ that is greater than any other policy (inequality defined above)
2) All optimal policies achieve the optimal value function, $v_{\pi_\star}(s)=v_\star(s)$
3)  All optimal policies achieve the optimal action-value function, $q_{\pi_\star}(s,a)=q_\star(s,a)$

Example of an optimal policy:
$$
\pi(a|s)=
\begin{cases}
1 & \text{if }a=\underset{a\in A}{\text{argmax}}\space q_\star(s,a) \\
0 & \text{otherwise}
\end{cases}
$$

# Policy Evaluation
The process of taking a policy, making an agent act according to that policy and iteratively finding the functions $v_\pi$ and $q_\pi$ is called policy evaluation.

This is essentially evaluating how good a policy is, because it represents the net reward that the agent can get on following the said policy in the form of the value of each state and of each state-action pair.

Once we know how to evaluate a policy, we can now start thinking about how to improve the policy

# Policy Iteration
- Given a policy $\pi$
	- **Evaluate** the policy $\pi$
	- **Improve** the policy by generating a new policy that simply acts greedily with respect to the evaluated $v_\pi$

Usually, this process needs to be repeated multiple times in order to get to an optimal policy $\pi^\star$

However, the process of policy iteration always converges to $\pi^\star$
**But why?**
The answer to that, lies in the fact that the new policies are constructed *greedily*. This means that at every *improvement* of a policy, the actions are adjusted in a way that the value of the corresponding state or the state-action pair strictly improves, because that is what greedy means.

#doubt: Is it possible for such a policy iteration to get stuck at a local maximum?

But when do we know when to stop? (practically, when the changes in value function become very small)
When we reach optimal policy. But what is the optimal policy?

## Principle of Optimality
A policy $\pi$ is said to have achieved the optimal value from state $s$ ($v_\pi(s)=v_*(s)$), **iff** for any state $s'$ reachable from $s$, 
$$
v_\pi(s')=v_*(s')
$$

## Comparison with value iteration
Value iteration is simply the process of finding a value function greedily with respect to a previously found value function. This can be thought of as a policy iteration with 1 round of value update for each round of policy update.

However, the value function's values can eventually decrease as the iteration proceeds.
**But wasn't it supposed to improve?**
Nope, the evaluated value function of policies generated is what is supposed to improve. Not all generated value functions correspond to a policy. This is a clear difference between value iteration and policy iteration.

# Asynchronous Dynamic Programming
In the discussion above, we usually consider every single action from every single state and every single state that the action can lead us to, and so on. This is called *synchronous dynamic programming*.

However, that probably involves too much unnecessary computation, the effects of which are apparent when the action space or the state space is very large.

Enter *asynchronous dynamic programming*, where only some subset of all of that computation is done. The value function is still guaranteed to go closer and closer to the optimal value function, so the algorithm is still guaranteed to converge.

We now discuss various methods of using asynchronous dynamic programming

## In-place dynamic programming
In synchronous value iteration, two functions (old and new) need to be stored.

However, using in-place value iteration, only one value function needs to be saved. The value function is updated in place, instead of in a new location. In this case, in one iteration, a greater progress is made, because new values generated in the iteration are used in the later stages of the same iteration.

## Prioritised Value Iteration
At the end of every value iteration, one can store the Bellman Error in the value of each state.

$$
\epsilon(s)=\left|\underset{a\in A}{\text{max}}\left(\mathcal{R}^a_s+\gamma\sum_{s'\in \mathcal{S}}\mathcal{P}^a_{ss'}v(s')\right)-v(s)\right|
$$


**Note**: This is computationally less expensive than the value-updation of that state.

If this error is large for some state, it indicates that there is more *information*(or a larger update) that can be extracted from processing that specific state (and even its neighbouring states). Thus, such a state should be given a higher priority in the next value iteration.
