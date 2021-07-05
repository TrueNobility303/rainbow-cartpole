import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net.replay import PrioritizedReplayBuffer 
from net.utils import initial_seed

#REF:https://github.com/Curt-Park/rainbow-is-all-you-need/

#优先级记忆回放，采用TD Error进行优先级采样，再使用重要性采样控制优先级采样的bias
initial_seed(42)

#三层MLP神经网络
class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DQNAgent:
    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        #优先级回放记忆，alpha=0.2
        self.beta = 0.6
        self.prior_eps = 1e-6
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size,alpha=0.2)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        self.device = torch.device('cuda:0')
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.transition = list()
        self.is_test = False

    #使用DQN，根据输入state，依据epsilon greegy策略选择action
    def select_action(self, state: np.ndarray) -> np.ndarray:
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        #训练阶段，将state,action对放入内存
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    #根据action返回env的response
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        #训练阶段将reward,next_state,done记录进内存并且放入回放记忆中
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        life_cnt = 0
        for frame_idx in range(1, num_frames + 1):
            #使用DQN进行action和step并且获取reward等response信息
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                life_cnt += 1
                print('life',life_cnt,'score',score)
                state = self.env.reset()

                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                #更新DQN
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

        plt.figure()
        plt.plot(scores)       
        mean_score = np.mean(scores)  
        mean_score = np.around(mean_score,2)
        plt.title('score: ' + str(mean_score))  
        plt.savefig('results/3_per.png')
        self.env.close()
                
    def test(self) -> List[np.ndarray]:
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        #print("score: ", score)
        self.env.close()
        
        return frames

    #根据记忆中的采样记录计算DQN损失
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()

        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        #计算当前Q value与target Q value之间的L1 los,逐点返回
        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return loss

    #每隔一段时间，将DQN的状态复制给target DQN
    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()

env_id = "CartPole-v0"
env = gym.make(env_id)

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
#frames = agent.test()
