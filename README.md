# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
We are assigned with the task of creating an RL agent to solve the "Bandit Slippery Walk" problem. The environment consists of Seven states representing discrete positions the agent can occupy. The agent must learn to navigate this environment while dealing with the challenge of slippery terrain. Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## POLICY EVALUATION FUNCTION
Policy evaluation refers to the objective and systematic examination of the effects of ongoing policies and public programs on their intended goals. It involves assessing whether policies are achieving their stated objectives and identifying any impediments to their attainment.
![image](https://github.com/user-attachments/assets/a895dad5-6784-4ec0-8da0-b3f254ef59d6)
```py
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V
```

## PROGRAM:
```py
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk

import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)

def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)

P

init_state

state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))

pi_2 = lambda s: {
    0: DOWN,
    1: RIGHT,
    2: RIGHT,
    3: DOWN,
    4: LEFT,
    5: RIGHT,
    6: DOWN,
    7: LEFT,
    8: DOWN,
    9: RIGHT,
    10: LEFT,
    11: DOWN,
    12: RIGHT,
    13: DOWN,
    14: RIGHT,
    15: LEFT  # Stop at goal
}[s]

print("Name: Archana k")
print("Register Number: 212222240011")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))
success_pi2 = probability_success(env, pi_2, goal_state=goal_state) * 100
mean_return_pi2 = mean_return(env, pi_2)

print("\nYour Policy Results:")
print(f"Reaches goal: {success_pi2:.2f}%")
print(f"Average undiscounted return: {mean_return_pi2:.4f}")
success_pi1 = probability_success(env, pi_frozenlake, goal_state=goal_state) * 100
mean_return_pi1 = mean_return(env, pi_frozenlake)

print("\nComparison of Policies:")
print(f"First Policy - Success Rate: {success_pi1:.2f}%, Mean Return: {mean_return_pi1:.4f}")
print(f"Your Policy  - Success Rate: {success_pi2:.2f}%, Mean Return: {mean_return_pi2:.4f}")

if success_pi1 > success_pi2:
    print("\nThe first policy is better based on success rate.")
elif success_pi2 > success_pi1:
    print("\nYour policy is better based on success rate.")
else:
    print("\nBoth policies have the same success rate.")

if mean_return_pi1 > mean_return_pi2:
    print("The first policy is better based on mean return.")
elif mean_return_pi2 > mean_return_pi1:
    print("Your policy is better based on mean return.")
else:
    print("Both policies have the same mean return.")

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V


V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P, gamma=0.99)

print("\nState-value function for Your Policy:")
print_state_value_function(V2, P, n_cols=4, prec=5)

if np.sum(V1 >= V2) == len(V1):
    print("\nThe first policy is the better policy.")
elif np.sum(V2 >= V1) == len(V2):
    print("\nYour policy is the better policy.")
else:
    print("\nBoth policies have their merits.")
V1>=V2

if(np.sum(V1>=V2)==11):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")

```

## OUTPUT:
Mention the first and second policies along with its state value function and compare them
<img width="365" height="210" alt="image" src="https://github.com/user-attachments/assets/4169e01a-f758-4b3d-bd33-d42f91396f8f" />
<img width="623" height="139" alt="image" src="https://github.com/user-attachments/assets/12310036-ab06-4f93-ab39-142598985830" />
<img width="433" height="86" alt="image" src="https://github.com/user-attachments/assets/6c3162f8-569c-4a47-bcb4-d71464e67f9a" />
<img width="501" height="36" alt="image" src="https://github.com/user-attachments/assets/cfd14fd8-3b27-48b2-bffc-115eff439eeb" />
<img width="407" height="106" alt="image" src="https://github.com/user-attachments/assets/a92111d8-6995-4029-9ddc-fb3429c5d9f0" />
<img width="500" height="24" alt="image" src="https://github.com/user-attachments/assets/e09fc4be-8d3c-436e-9127-2f365d0f0244" />
<img width="305" height="67" alt="image" src="https://github.com/user-attachments/assets/36808206-7312-40d4-81f9-f686d841b076" />
<img width="434" height="70" alt="image" src="https://github.com/user-attachments/assets/a0093cae-0159-42b0-9b7c-1341f68973fb" />
<img width="385" height="59" alt="image" src="https://github.com/user-attachments/assets/ac786d16-43a5-4030-b6bf-303b786ca08d" />
<img width="434" height="201" alt="image" src="https://github.com/user-attachments/assets/fb77f0eb-dc25-4a2b-a1ea-18a0f82238f6" />
<img width="311" height="59" alt="image" src="https://github.com/user-attachments/assets/8783e97c-0dd9-4beb-82a8-ef14d5726691" />



## RESULT:

Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and execcuted successfully.
