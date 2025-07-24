# Bayesian Bandits for Airbnb Experimentation
# A Proof of Concept with A/B Test Comparison

# ### The Airbnb Problem: Optimizing for Bookings
#
# At its core, Airbnb wants to create the best user experience to facilitate bookings. 
# A key area for optimization is the **listing page**, where guests decide whether a 
# property is right for them. A small change, like how amenities are displayed, can 
# have a significant impact on conversion rates when scaled across millions of users.
#
# The challenge is to test new design ideas (variations) quickly and efficiently, 
# without losing potential bookings by showing users an inferior design for too long. 
# This is the classic **explore-exploit trade-off**:
#
# * **Explore:** We need to gather enough data on all variations to know which one 
#   is truly the best.
# * **Exploit:** Once we have evidence that a variation is winning, we should start 
#   showing it to more users to maximize bookings.

# ## Step 1: Imports and Class Definition
# First, import the necessary libraries for numerical operations (`numpy`), 
# visualization (`matplotlib`, `seaborn`), statistical distributions (`scipy`)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns
import copy

# Set a professional plot style
sns.set(style="whitegrid")

# Set a random seed for reproducibility
np.random.seed(42)

# Define Bayesian Bandit function for sampling
class BayesianBandit:
    """
    Represents a single 'arm' or variation in our experiment.
    """
    def __init__(self, name):
        self.name = name
        self.alpha = 1
        self.beta = 1

    def sample(self):
        """Draws a random sample from the current Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, result):
        """Updates the Beta distribution based on the result of a trial."""
        if result == 1:
            self.alpha += 1
        else:
            self.beta += 1
            
    @property
    def estimated_conversion_rate(self):
        """Calculates the expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

# ## Step 2: Setup the Experiment
# Now, define our experimental variations and 
# their *true* (but unknown to the algorithm) conversion rates.

# Define the true conversion rates for each variation
true_conversion_rates = {
    "Control": 0.030,       # 3.0% booking rate
    "Icons": 0.032,         # 3.2% booking rate (a small improvement)
    "Smart Summary": 0.045, # 4.5% booking rate (the clear winner)
    "Visual Grid": 0.028    # 2.8% booking rate (worse than control)
}

num_trials = 20000 # Simulate 20,000 users visiting the listing page
checkpoints = [100, 500, 2000, 5000, num_trials]

# ## Step 3: Run Simulations and Collect Data
# Run both simulations trial-by-trial to collect data for our plots.

# --- Bayesian Bandit Simulation ---
print("--- Running Bayesian Bandit Simulation ---")
bandits = [BayesianBandit(name) for name in true_conversion_rates.keys()]
bandit_selections = {name: 0 for name in true_conversion_rates.keys()}
bandit_total_reward = 0
bandit_conversion_history = []
bandit_states_at_checkpoints = []

for i in range(num_trials):
    samples = [b.sample() for b in bandits]
    chosen_bandit_index = np.argmax(samples)
    chosen_bandit = bandits[chosen_bandit_index]
    
    true_rate = true_conversion_rates[chosen_bandit.name]
    reward = 1 if np.random.random() < true_rate else 0
    
    chosen_bandit.update(reward)
    
    bandit_selections[chosen_bandit.name] += 1
    bandit_total_reward += reward
    bandit_conversion_history.append(bandit_total_reward / (i + 1))

    if i + 1 in checkpoints:
        # Save the state of the bandits at this checkpoint for later plotting
        bandit_states_at_checkpoints.append(copy.deepcopy(bandits))

# --- Traditional A/B Test Simulation ---
print("\n--- Running Traditional A/B Test Simulation ---")
ab_test_total_reward = 0
ab_conversion_history = []
variation_names = list(true_conversion_rates.keys())
num_variations = len(variation_names)

for i in range(num_trials):
    # Assign user to a variation in a round-robin fashion for even split
    variation_name = variation_names[i % num_variations]
    true_rate = true_conversion_rates[variation_name]
    
    reward = 1 if np.random.random() < true_rate else 0
    ab_test_total_reward += reward
    ab_conversion_history.append(ab_test_total_reward / (i + 1))

# ## Step 4: Create Plots
# Now use the collected data to generate our visualizations.

# Plot 1: Bandit Posterior Distributions Panel
print("\n--- Generating plot: Distribution Panel ---")
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten() # Flatten the 2x3 grid into a 1D array

for i, (bandits_state, trial_num) in enumerate(zip(bandit_states_at_checkpoints, checkpoints)):
    ax = axes[i]
    x = np.linspace(0, 0.1, 200)
    for bandit in bandits_state:
        y = beta.pdf(x, bandit.alpha, bandit.beta)
        ax.plot(x, y, label=f'{bandit.name} (ECR: {bandit.estimated_conversion_rate:.3%})')
    ax.set_title(f'Distributions after {trial_num} Trials')
    ax.legend()
    ax.set_xlabel('Conversion Rate')
    ax.set_ylabel('Density')

# Hide the last unused subplot
axes[-1].axis('off')

fig.suptitle('Evolution of Bandit Posterior Beliefs Over Time', fontsize=20)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('distribution_panel.png')
plt.close()

# Plot 2: Cumulative Conversion Rate History
print("--- Generating plot: Conversion Rate History ---")
plt.figure(figsize=(14, 7))
plt.plot(bandit_conversion_history, label='Bayesian Bandit')
plt.plot(ab_conversion_history, label='Traditional A/B Test')
plt.title('Cumulative Conversion Rate Over Time', fontsize=16)
plt.xlabel('Number of Trials')
plt.ylabel('Cumulative Conversion Rate')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('conversion_rate_history.png')
plt.close()

# Plot 3: Final Traffic Allocation for Bandit
print("--- Generating plot: Final Bandit Traffic Allocation ---")
labels = list(true_conversion_rates.keys())
bandit_counts = list(bandit_selections.values())

plt.figure(figsize=(12, 6))
plt.bar(labels, bandit_counts, color=sns.color_palette("viridis", len(labels)))
plt.title('Bayesian Bandit: Final Traffic Allocation', fontsize=16)
plt.ylabel('Number of Trials (Users)', fontsize=12)
plt.savefig('final_allocation_bandit.png')
plt.close()

# ## Step 5: Analyze and Compare Final Results
# With both simulations complete, print a summary and compare their performance directly

print("\n--- Simulation Complete: Final Comparison ---")

print(f"\nBayesian Bandit Total Bookings: {bandit_total_reward}")
print(f"Traditional A/B Test Total Bookings: {ab_test_total_reward}")

print("\n--- Performance Comparison ---")
lift = bandit_total_reward - ab_test_total_reward
lift_percentage = (lift / ab_test_total_reward) * 100
print(f"The Bayesian Bandit approach resulted in {lift} more bookings.")
print(f"This is a {lift_percentage:.2f}% lift in conversions during the experiment.")

print("\nAll plots have been saved to the working directory.")

