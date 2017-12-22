import numpy as np
import networkx as nx

class MVC_env:
    def __init__(self, number_nodes, p = 0.15, replay_penalty=0):
        self.number_nodes = number_nodes
        self.p = p
        self.state_shape = [self.number_nodes]
        self.num_actions = self.number_nodes
        self.env_name = 'MVC'
        self.replay_penalty = replay_penalty

    def reset(self):
        self.acc_reward = 0
        self.graph = nx.erdos_renyi_graph(n = self.number_nodes, p = self.p)
        self.nodes = list(self.graph.nodes)
        self.edges = list(self.graph.edges)
        self.state = np.zeros(self.number_nodes)
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)
        self.weight_matrix = self.adjacency_matrix

        if len(self.edges) == 0:
            self.reset()
        return self.state

    def is_done(self, state):
        done = True
        for e in self.edges:
            if state[e[0]] == 0 and state[e[1]] == 0:
                done = False

        return done

    def step(self, action):
        if self.state[action] != 1:
            self.state[action] = 1
            rew = -1
        else:
            rew = -self.replay_penalty

        self.acc_reward += rew

        return self.state, rew, self.is_done(self.state)

    def accumulated_reward(self):
        return self.acc_reward

    def at_random_solution(self):
        temp_state = np.zeros(self.number_nodes)
        while not self.is_done(temp_state):
            temp_state[np.random.randint(self.number_nodes)] = 1

        return -np.sum(temp_state), temp_state

    def optimal_solution(self):
        # TODO
        return 0, None