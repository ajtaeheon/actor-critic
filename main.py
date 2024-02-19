import gym
import torch
from torch.distributions import Categorical
from model import ActorCritic
import matplotlib.pyplot as plt

num_rollout = 10


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    term_print = 20
    score = 0.0

    history_epi = []
    history_score = []

    for num_epi in range(10000):
        done = False
        s, _ = env.reset()

        while not done:
            for t in range(num_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)
                transition = (s,a,r,s_prime,done)
                model.put_data(transition)

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        if num_epi % term_print == 0 and num_epi != 0:
            avg_score = score/term_print
            print("Episode {} - Average Score: {:.1f}".format(num_epi, avg_score))
            history_epi.append(num_epi)
            history_score.append(avg_score)
            score = 0.0
    env.close()

    plt.plot(history_epi, history_score)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()


