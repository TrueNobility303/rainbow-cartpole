# rainbow-cartpole

## Abstract

使用rainbow进行gym上的cartpole游戏

rainbow是深度强化学习的集大成者，由7大部分组成，故称rainbow
* DQN 深度Q网络，利用神经网络估计Q值为agent采取策略
* Double DQN 将critic DQN和actor DQN分开，分离策略和评估环节，解决DQN中取max操作造成的估计偏差，类似演员-评论家算法
* PER(Prioritized exprience replay) 优先记忆回放，使用TD Error作为记忆的优先级，类似OHEM（Online Hard Exmaple Mining）
* Duel DQN 将DQN预测的Q值分为state value和action value（advantage）两部分
* Noisy DQN 在DQN中引入噪声层，使评估结果自带噪声，取代DQN中原本的$\epsilon$ greedy policy
* Distribution Perspectve 认为Q值作为一种期望，其背后为一种分布，因此让DQN直接估计该分布，推广原先基于期望的Bellman方程
* NStep 将Bellman方程由1 step推广至n step, 也即使得DQN的眼光放到n步之内

Rianbow 集成了上述7大技术，并的确在鲁棒性和性能上超过远远超越其他模型

## Results

| Method | Score |
| --- | --- |
|DQN|112.17|
|Double DQN|122.68|
|PER|138.53|
|Duel DQN|123.82|
|Noisy DQN|111.95|
|Distribution|113.3|
|N Step|145.87|
|**Rainbow**|**162.36**|
