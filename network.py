import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Categorical
def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def random(adagrahnum,numnode_v=25,numnode_w=25):
    return np.random.randn(adagrahnum,3, numnode_v, numnode_w) * .02 + .04

class Memory:
    def __init__(self):
        self.s_actions = []
        self.t_actions = []

        self.states = []

        self.s_logprobs = []
        self.t_logprobs = []

        self.rewards = []
        self.is_terminals = []
        self.s_hidden = []
        self.t_hidden = []

    def clear_memory(self):
        del self.s_actions[:]
        del self.t_actions[:]
        del self.states[:]
        del self.s_logprobs[:]
        del self.t_logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.s_hidden[:]
        del self.t_hidden[:]


class ActorCritic(nn.Module):

    def __init__(self, feature_dim, state_dim, hidden_state_dim=256, policy_conv=True, action_std=0.1, graph=None):
        super(ActorCritic, self).__init__()
        self.common_state_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        part_count = len(graph.part_node[0])
        # self.s_prime_state_encoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(48 * 11, hidden_state_dim),
        #     nn.ReLU()
        # )
        self.max_part_node =graph.part_A[0][0].shape[1]
        prime_A  = torch.tensor(random(8, 9, 8), dtype=torch.float32, requires_grad=False)
        self.s_prime_A = nn.Parameter(prime_A.clone().permute(1, 0, 2, 3).contiguous())
        self.s_prime_conv = nn.Conv2d(48, 3 * (hidden_state_dim//8), 1)


        focus_A  = torch.tensor(random(8, self.max_part_node, 8), dtype=torch.float32, requires_grad=False)
        self.s_focus_A = nn.Parameter(focus_A.clone().permute(1, 0, 2, 3).contiguous())
        self.s_focus_conv = nn.Conv2d(48, 3 * (hidden_state_dim//8), 1)

        self.relu = nn.ReLU()



        self.t_prime_state_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 25, hidden_state_dim),
            nn.ReLU()
        )
        self.t_focus_state_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8, hidden_state_dim),
            nn.ReLU()
        )




        self.t_gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)
        self.s_gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)

        self.t_actor = nn.Sequential(
            nn.Linear(hidden_state_dim, 1),
            nn.Sigmoid()
        )

        self.s_actor = nn.Sequential(
            nn.Linear(hidden_state_dim, part_count),
            nn.Softmax(dim=-1))

        self.s_critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))
        self.t_critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.action_var = torch.full((1,), action_std).cuda()
        self.flatten = nn.Flatten()
        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=False):
        if restart_batch:
            del memory.s_hidden[:]
            del memory.t_hidden[:]
            memory.s_hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())
            memory.t_hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        N, M, C, T, V = state_ini.shape
        state = state_ini.reshape(N * M, C, T, V)
        state = self.common_state_encoder(state)

        s_state = state.mean(dim=2,keepdim=True)
        t_state = state.mean(dim=3)

        if restart_batch:

            prime_A = self.s_prime_A.repeat(
                1, (self.hidden_state_dim // 8 )//8, 1, 1)
            s_state = self.s_prime_conv(s_state)
            s_state = s_state.view(N * M, 3, self.hidden_state_dim//8, 1, 9)
            s_state = torch.einsum('nkctv,kcvw->nctw', (s_state, prime_A)).contiguous()
            s_state = torch.squeeze(s_state)
            s_state = self.flatten(self.relu(s_state))

            t_state = self.t_prime_state_encoder(t_state)

        else:
            focus_A = self.s_focus_A.repeat(
                1, (self.hidden_state_dim // 8 )//8, 1, 1)
            s_state = self.s_focus_conv(s_state)
            s_state = s_state.view(N * M, 3, self.hidden_state_dim//8, 1, self.max_part_node)
            s_state = torch.einsum('nkctv,kcvw->nctw', (s_state, focus_A)).contiguous()
            s_state = torch.squeeze(s_state)
            s_state = self.flatten(self.relu(s_state))

            t_state = self.t_focus_state_encoder(t_state)

        s_state = s_state.reshape(N, M, -1)
        s_state = s_state.mean(dim=1)
        t_state = t_state.reshape(N, M, -1)
        t_state = t_state.mean(dim=1)

        s_state, s_hidden_output = self.s_gru(s_state.view(1, s_state.size(0), s_state.size(1)), memory.s_hidden[-1])
        t_state, t_hidden_output = self.t_gru(t_state.view(1, t_state.size(0), t_state.size(1)), memory.t_hidden[-1])

        memory.s_hidden.append(s_hidden_output)
        memory.t_hidden.append(t_hidden_output)
        # state = state[0]
        # action_mean = self.actor(state)

        s_state = s_state[0]
        s_action_mean = self.s_actor(s_state)
        t_state = t_state[0]
        t_action_mean = self.t_actor(t_state)

        cov_mat = torch.diag(self.action_var).cuda()
        t_dist = torch.distributions.multivariate_normal.MultivariateNormal(t_action_mean, scale_tril=cov_mat)

        s_dist = Categorical(s_action_mean)

        if training:
            t_action = t_dist.sample().cuda()
            t_action = F.relu(t_action)
            t_action = 1 - F.relu(
                1 - t_action)  # 第一行代码的作用是将action中小于0的部分变为0，第二行代码的作用是将action中大于1的部分变为1，其余部分不变。这样可以确保动作向量中的每个元素都在0到1之间。
            t_action_logprob = t_dist.log_prob(t_action).cuda()

            s_action = s_dist.sample().cuda()
            s_action_logprob = s_dist.log_prob(s_action)

            memory.states.append(state_ini)

            memory.s_actions.append(s_action)
            memory.t_actions.append(t_action)
            memory.s_logprobs.append(s_action_logprob)
            memory.t_logprobs.append(t_action_logprob)
        else:
            s_action = s_action_mean.max(1)[1]
            t_action = t_action_mean

        return s_action, t_action.detach()

    def evaluate(self, state, s_action, t_action):

        seq_l = len(state)

        s_states = []
        t_states = []
        for idx, s in enumerate(state):
            N, M, C, T, V = s.shape
            s = s.reshape(N * M, C, T, V)
            s = self.common_state_encoder(s)
            s_state = s.mean(dim=2,keepdim=True)
            t_state = s.mean(dim=3)
            if idx==0:

                prime_A = self.s_prime_A.repeat(
                    1, (self.hidden_state_dim // 8) // 8, 1, 1)
                # s_state = self.s_prime_state_encoder(s_state)
                s_state = self.s_prime_conv(s_state)
                s_state = s_state.view(N * M, 3, self.hidden_state_dim//8, 1, 9)
                s_state = torch.einsum('nkctv,kcvw->nctw', (s_state, prime_A)).contiguous()
                s_state = torch.squeeze(s_state)
                s_state = self.flatten(self.relu(s_state))

                t_state = self.t_prime_state_encoder(t_state)

            else:
                focus_A = self.s_focus_A.repeat(
                    1, (self.hidden_state_dim // 8) // 8, 1, 1)
                # s_state = self.s_prime_state_encoder(s_state)
                s_state = self.s_focus_conv(s_state)
                s_state = s_state.view(N * M, 3, self.hidden_state_dim//8, 1, self.max_part_node)
                s_state = torch.einsum('nkctv,kcvw->nctw', (s_state, focus_A)).contiguous()
                s_state = torch.squeeze(s_state)
                s_state = self.flatten(self.relu(s_state))

                t_state = self.t_focus_state_encoder(t_state)

            s_state = s_state.reshape(N, M, -1)
            s_state = s_state.mean(dim=1)
            t_state = t_state.reshape(N, M, -1)
            t_state = t_state.mean(dim=1)
            s_states.append(s_state)
            t_states.append(t_state)

        s_state = torch.cat(s_states, 0)
        t_state = torch.cat(t_states, 0)

        # state = state.reshape(seq_l, N, -1)
        s_state = s_state.reshape(seq_l, N, -1)
        t_state = t_state.reshape(seq_l, N, -1)

        # state = state.mean(dim=2)
        # state, hidden = self.gru(state, torch.zeros(1, N, state.size(2)).cuda())
        s_state, s_hidden_output = self.s_gru(s_state, torch.zeros(1, N, s_state.size(2)).cuda())
        t_state, t_hidden_output = self.t_gru(t_state, torch.zeros(1, N, t_state.size(2)).cuda())

        s_state = s_state.view(seq_l * N, -1)
        t_state = t_state.view(seq_l * N, -1)

        s_action_probs = self.s_actor(s_state)
        t_action_probs = self.t_actor(t_state)

        cov_mat = torch.diag(self.action_var).cuda()
        t_dist = torch.distributions.multivariate_normal.MultivariateNormal(t_action_probs, scale_tril=cov_mat)
        s_dist = Categorical(s_action_probs)

        s_action_logprobs = s_dist.log_prob(torch.squeeze(s_action.view(seq_l * N, -1))).cuda()
        t_action_logprobs = t_dist.log_prob(t_action .view(seq_l * N, -1)).cuda()

        t_dist_entropy = t_dist.entropy().cuda()
        s_dist_entropy = s_dist.entropy().cuda()

        s_state_value = self.s_critic(s_state)
        t_state_value = self.t_critic(t_state)

        return s_action_logprobs.view(seq_l, N), t_action_logprobs.view(seq_l, N), \
               s_state_value.view(seq_l, N), t_state_value.view(seq_l, N), \
               s_dist_entropy.view(seq_l, N), t_dist_entropy.view(seq_l, N)


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim,graph=None, policy_conv=True,
                 action_std=0.1, lr=0.0001, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std, graph).cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std, graph).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.graph = graph
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # old_states = [state.cuda().detach() for state in memory.states]
        # old_actions = [action.cuda().detach() for action in memory.actions]

        old_states = [state.cuda().detach() for state in memory.states]

        s_old_logprobs = torch.stack(memory.s_logprobs, 0).cuda().detach()
        t_old_logprobs = torch.stack(memory.t_logprobs, 0).cuda().detach()

        s_old_actions = torch.stack(memory.s_actions, 0).cuda().detach()
        t_old_actions = torch.stack(memory.t_actions, 0).cuda().detach()

        for _ in range(self.K_epochs):
            s_logprobs, t_logprobs, s_state_values, t_state_values, s_dist_entropy, t_dist_entropy = self.policy.evaluate(
                old_states, s_old_actions, t_old_actions)

            s_ratios = torch.exp(s_logprobs - s_old_logprobs.detach())
            t_ratios = torch.exp(t_logprobs - t_old_logprobs.detach())

            s_advantages = rewards - s_state_values.detach()
            t_advantages = rewards - t_state_values.detach()

            s_surr1 = s_ratios * s_advantages
            s_surr2 = torch.clamp(s_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * s_advantages

            t_surr1 = t_ratios * t_advantages
            t_surr2 = torch.clamp(t_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * t_advantages

            s_loss = -torch.min(s_surr1, s_surr2) + 0.5 * self.MseLoss(s_state_values, rewards) - 0.01 * s_dist_entropy
            t_loss = -torch.min(t_surr1, t_surr2) + 0.5 * self.MseLoss(t_state_values, rewards) - 0.01 * t_dist_entropy
            loss = s_loss + t_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class First_step_state_encoder(nn.Module):
    def __init__(self, feature_dim, hidden_state_dim=256):
        super(First_step_state_encoder, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_state_dim, 1, bias=False),
            nn.Tanh(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.Tanh(),
        )

    def forward(self, state_ini):
        N, M, C, T, V = state_ini.shape
        state = state_ini.reshape(N * M, C, T, V)
        state = self.state_encoder(state)
        state = state.reshape(N, M, -1)
        state = state.mean(dim=1)
        return state


class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, hidden_state_dim=1024, fc_rnn=True, class_num=1000):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num

        self.hidden_state_dim = hidden_state_dim
        self.hidden = None
        self.fc_rnn = fc_rnn
        if fc_rnn:
            self.rnn = nn.GRU(feature_num, self.hidden_state_dim)
            self.fc = nn.Linear(self.hidden_state_dim, class_num)
        else:
            self.fc_2 = nn.Linear(self.feature_num * 2, class_num)
            self.fc_3 = nn.Linear(self.feature_num * 3, class_num)
            self.fc_4 = nn.Linear(self.feature_num * 4, class_num)
            self.fc_5 = nn.Linear(self.feature_num * 5, class_num)

    def forward(self, x, restart=False):

        if self.fc_rnn:
            if restart:
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)),
                                       torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
                self.hidden = h_n
            else:
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), self.hidden)
                self.hidden = h_n

            return self.fc(output[0])
        else:
            if restart:
                self.hidden = x
            else:
                self.hidden = torch.cat([self.hidden, x], 1)

            if self.hidden.size(1) == self.feature_num:
                return None
            elif self.hidden.size(1) == self.feature_num * 2:
                return self.fc_2(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 3:
                return self.fc_3(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 4:
                return self.fc_4(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 5:
                return self.fc_5(self.hidden)
            else:
                print(self.hidden.size())
                exit()