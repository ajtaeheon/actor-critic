import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

learning_rate = 2e-4
gamma = 0.98

# if torch.cuda.is_available():
#     from torch.cuda import FloatTensor
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     from torch import FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc_shared = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc_shared(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc_shared(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        states, actions, rewards, states_prime, done_list = [], [], [], [], []
        for trans in self.data:
            s,a,r,s_prime,done = trans
            states.append(s)
            actions.append([a])
            rewards.append([r/100.0])
            states_prime.append(s_prime)
            d_mask = 0.0 if done else 1.0
            done_list.append([d_mask])


        s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
            torch.tensor(states, dtype=torch.float).to(device),\
            torch.tensor(actions).to(device),\
            torch.tensor(rewards,dtype=torch.float).to(device),\
            torch.tensor(states_prime,dtype=torch.float).to(device),\
            torch.tensor(done_list, dtype=torch.float).to(device)
        
        # s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
        #     FloatTensor(states),\
        #     torch.tensor(actions),\
        #     FloatTensor(rewards),\
        #     FloatTensor(states_prime),\
        #     torch.tensor(done_list, dtype=torch.float)

        self.data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s,a,r,s_prime,done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
