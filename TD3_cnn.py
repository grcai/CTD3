import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

nl1 = 256
nl2 = 256
ncc1 = 128
ncc2 = 128
pd = 0.2
nc = 512
nk = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=(8, state_dim), stride=2, padding=0),
            torch.nn.ReLU()
        )

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.l01 = nn.Linear(nc, ncc1)
        self.l02 = nn.Linear(state_dim, ncc2)
        self.l1 = nn.Linear(ncc1+ncc2, nl1)
        self.l2 = nn.Linear(nl1, action_dim)

        self.dropout = nn.Dropout(p=pd)
        self.max_action = max_action

    def forward(self, con_state, state):
        a = self.conv1(con_state)
        a = self.maxpool(a)
        a = a.view(a.size(0), -1)
        a = F.relu(self.l01(a))
        b = F.relu(self.l02(state))
        a = torch.cat([a, b], 1)
        a = F.leaky_relu(self.l1(a))
        a = self.dropout(a)
        return self.max_action * torch.tanh(self.l2(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, action, state):
        q1 = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(q1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.cat([state, action], 1)
        q2 = F.relu(self.l4(q2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, action, state):

        q1 = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(q1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class CTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, con_state, state):
        con_state = torch.FloatTensor(con_state).to(device)
        state = torch.FloatTensor(state).to(device)
        return self.actor(con_state, state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, con_state, con_next_state = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(con_next_state, next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value

            target_Q1, target_Q2 = self.critic_target(next_action, next_state)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(action, state)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(self.actor(con_state, state), state).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
