import numpy as np
import matplotlib.pyplot as plt
import random

class ECommerceEnvironment:
    def __init__(self, price_sensitivity, max_price):
        self.price_sensitivity = price_sensitivity
        self.max_price = max_price

    def get_demand(self, price):
        """
        Simulate the demand based on the price.
        Demand decreases with price and is affected by price sensitivity.
        """
        base_demand = 100
        demand = base_demand - self.price_sensitivity * (price - self.max_price / 2)
        return max(0, demand)  # Demand can't be negative

    def step(self, price):
        """
        Calculate revenue and next state (demand) based on the price.
        """
        demand = self.get_demand(price)
        revenue = price * demand
        return revenue

class DynamicPricingAgent:
    def __init__(self, price_space, learning_rate=0.1, discount_factor=0.9):
        self.q_table = {price: 0 for price in price_space}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.price_space = price_space

    def choose_action(self, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.price_space)  # Explore
        else:
            return max(self.q_table, key=self.q_table.get)  # Exploit

    def update_q_value(self, price, reward):
        """
        Update Q-value using the Q-learning formula.
        """
        self.q_table[price] = self.q_table[price] + self.learning_rate * (
            reward - self.q_table[price]
        )

# Simulation Parameters
price_sensitivity = 1.5
max_price = 100
price_space = list(range(1, max_price + 1))
num_episodes = 5000
epsilon = 0.2  # Exploration rate

environment = ECommerceEnvironment(price_sensitivity, max_price)
agent = DynamicPricingAgent(price_space)

# Training Loop
revenue_history = []
for episode in range(num_episodes):
    price = agent.choose_action(epsilon)
    revenue = environment.step(price)
    agent.update_q_value(price, revenue)
    revenue_history.append(revenue)

# Results
plt.plot(revenue_history)
plt.xlabel("Episode")
plt.ylabel("Revenue")
plt.title("Revenue over Time")
plt.show()

# Optimal Pricing
optimal_price = max(agent.q_table, key=agent.q_table.get)
print(f"Optimal price: {optimal_price}")


