# Q Learning Algorithm


## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.



## Q LEARNING FUNCTION

### NAME :  PRIYADARSHINI R 

### REFERENCE NO :  212220230038

```PYTHON3

def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target = reward + gamma * np.max(Q[next_state]) * (not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

```

## OUTPUT:

### Mention the optimal policy, optimal value function , success rate for the optimal policy.

![image](https://github.com/priya672003/q-learning/assets/81132849/79a4a469-1e2c-460d-b458-0c5d93e0621a)


### Include plot comparing the state value functions of Monte Carlo method and Qlearning.

![image](https://github.com/priya672003/q-learning/assets/81132849/3246a327-c2ac-4a74-bc68-7fdf365701d1)



![image](https://github.com/priya672003/q-learning/assets/81132849/862c7cf6-f6c4-401d-9475-82fa6ad8aef8)


## RESULT:

Write your result here
