import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from advanced_env import BlenderEnv
import os
import bpy
import matplotlib.pyplot as plt


# ---- Hyperparameter ----
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-5
LR_CRITIC = 5e-5
BUFFER_SIZE = 100000
BATCH_SIZE = 64

class Actor(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, act_dim)
        bound = 0.06
        nn.init.uniform_(self.fc3.weight, -bound, bound)
        nn.init.uniform_(self.fc3.bias, -bound, bound)
        nn.init.uniform_(self.fc1.weight, -bound, bound)
        nn.init.uniform_(self.fc1.bias, -bound, bound)
        nn.init.uniform_(self.fc2.weight, -bound, bound)
        nn.init.uniform_(self.fc2.bias, -bound, bound)
        nn.init.uniform_(self.fc4.weight, -bound, bound)
        nn.init.uniform_(self.fc4.bias, -bound, bound)
        

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.clamp(self.fc4(x)/5.0, -20.0, 20.0)


class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(act_dim + state_dim, 256) # state dimension + action dimension
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(terminateds).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DDPG_Agent:
    def __init__(self):
        self.actor = Actor(6, 2).to(device)
        self.actor_target = Actor(6, 2).to(device)
        self.critic = Critic(6, 2).to(device)
        self.critic_target = Critic(6, 2).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=1e-3)

        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, noise_scale=1.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(2)
        return action

    def train(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, terminateds = self.buffer.sample(BATCH_SIZE)

        states = states.detach().clone().to(device)
        actions = actions.detach().clone().to(device)
        rewards = rewards.detach().clone().to(device)
        next_states = next_states.detach().clone().to(device)
        terminateds = terminateds.detach().clone().to(device)

        # Critic loss
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_value = rewards + GAMMA * (1 - terminateds) * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target update
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
    def save_agent(self, path):
        file_name = f"{path}adv_ddpg_agent.pt"
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, file_name)

    def load_agent(self, path):
        file_name = f"{path}adv_ddpg_agent.pt"
        checkpoint = torch.load(file_name)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


def train_agent(episodes):
    noise_scale = 1.0
    for ep in range(episodes):
        print(f"training progress: {ep+1}/{episodes}")
        terminated = False
        state = env.reset()
        while (not terminated):
            action = agent.select_action(state)
            next_state, reward, terminated = env.step(action)

            agent.buffer.push(state, action, reward, next_state, terminated)
            agent.train()

            state = next_state

        noise_scale *= 0.9998

def eval_critic():
    agent.load_agent(agent_path)
    x_res, y_res = 50, 50
    state = torch.FloatTensor([0.5, 0.5, 100, 100, 200, 150.0]).unsqueeze(0)
    q_s = torch.zeros((x_res, y_res))
    for i in range(x_res):
        for j in range(y_res):
            q_s[i,j] = agent.critic.forward(state, torch.FloatTensor([i*0.2-50, j*0.2-50]).unsqueeze(0)).detach()

    plt.imshow(np.array(q_s), cmap="viridis", origin="lower")
    plt.colorbar()   # Farbskala hinzufügen
    plt.show()

    state = torch.FloatTensor([0.5, 0.5, 0, 0, 4.0, 150.0]).unsqueeze(0)
    q_s = torch.zeros((x_res, y_res))
    for i in range(x_res):
        for j in range(y_res):
            q_s[i,j] = agent.critic.forward(state, torch.FloatTensor([i*0.2-50, j*0.2-50]).unsqueeze(0)).detach()

    plt.imshow(np.array(q_s), cmap="viridis", origin="lower")
    plt.colorbar()   # Farbskala hinzufügen
    plt.show()

    state = torch.FloatTensor([0.5, 0.5, 199, 199, 4.0, 150.0]).unsqueeze(0)
    q_s = torch.zeros((x_res, y_res))
    for i in range(x_res):
        for j in range(y_res):
            q_s[i,j] = agent.critic.forward(state, torch.FloatTensor([i*0.2-50, j*0.2-50]).unsqueeze(0)).detach()

    plt.imshow(np.array(q_s), cmap="viridis", origin="lower")
    plt.colorbar()   # Farbskala hinzufügen
    plt.show()
    
# initialize parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_new_variations = True
load_agent = False
eval = False
training_episodes = 25000

log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Results/adv_ddpg_log.txt")

agent = DDPG_Agent()

agent_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Results/Agents/")

if eval: eval_critic()
else:   
    # create environment
    env = BlenderEnv(training_episodes, log_path, "adv_ddpg")
    if generate_new_variations:
        env.generate_env_variations()
        env.write_env_variations()
    else: env.read_env_variations()
    env.number_resets = 0
    if load_agent: agent.load_agent(agent_path)

    #train_agent(training_episodes)

    #agent.save_agent(agent_path)

    #env.close_log_file()
    #bpy.ops.wm.save_as_mainfile(filepath=f"{env.env_path}/a.blend")
