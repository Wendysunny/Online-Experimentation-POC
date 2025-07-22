# Bayesian Bandits for Airbnb Experimentation
# A Proof of Concept

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

# ### The Bayesian Bandit Solution
#
# This proof-of-concept uses a **Bayesian multi-armed bandit** approach with 
# **Thompson Sampling** to solve this problem.
#
# 1. **The "Arms"**: Each variation of the amenity display (`Control`, `Icons`, 
#    `Smart Summary`, `Visual Grid`) is an "arm" of a multi-armed bandit.
#
# 2. **Modeling Beliefs**: We model our belief about the true booking rate of each 
#    arm using a **Beta distribution**. Initially, this distribution is flat (a uniform 
#    distribution from 0 to 1), signifying that any conversion rate is equally likely.
#
# 3. **Thompson Sampling in Action**: For each user that visits the page, the algorithm 
#    does the following:
#    * It draws one random sample from each arm's current Beta distribution. You can 
#      think of this sample as a "plausible" conversion rate for that arm, given the 
#      data seen so far.
#    * It shows the user the variation that had the **highest sample**.
#    * It observes the user's action (booking or no booking).
#    * It updates the Beta distribution of the variation that was shown. A booking 
#      makes the distribution's mean shift higher; a non-booking makes it shift lower.

# ## Step 1: Imports and Class Definition
#
# First, we'll import the necessary libraries for numerical operations (`numpy`), 
# plotting (`matplotlib`, `seaborn`), and statistical distributions (`scipy`).
#
# We also define the `BayesianBandit` class. This class represents a single 'arm' or 
# variation in our experiment. It handles the logic for storing successes and failures 
# (as `alpha` and `beta` parameters of a Beta distribution), sampling from the 
# distribution, and updating it with new results.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns

# Set a professional plot style
sns.set(style="whitegrid")

class BayesianBandit:
    """
    Represents a single 'arm' or variation in our experiment.
    
    This class models our belief about the conversion rate of a single variation 
    using a Beta distribution. The Beta distribution is a great choice for modeling
    the probability of a binary outcome (like a booking).

    Attributes:
        name (str): The name of the variation (e.g., "Control", "Icons").
        alpha (int): The number of successes (e.g., bookings). Initialized to 1.
        beta (int): The number of failures (e.g., no booking). Initialized to 1.
    """
    def __init__(self, name):
        self.name = name
        # We start with an uninformative prior (alpha=1, beta=1), which is a uniform distribution.
        # This represents our initial lack of knowledge about the arm's performance.
        self.alpha = 1
        self.beta = 1

    def sample(self):
        """
        Draws a random sample from the current Beta distribution.
        This is the core of Thompson Sampling. The sampled value represents a plausible
        conversion rate for this arm, given the data we've seen so far.
        """
        return np.random.beta(self.alpha, self.beta)

    def update(self, result):
        """
        Updates the Beta distribution based on the result of a trial.
        
        Args:
            result (int): 1 for a success (booking), 0 for a failure.
        """
        if result == 1:
            self.alpha += 1
        else:
            self.beta += 1
            
    @property
    def estimated_conversion_rate(self):
        """Calculates the expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

# ## Step 2: Setup the Experiment
#
# Now, we'll configure the simulation. We'll define our experimental variations and 
# their *true* (but unknown to the algorithm) conversion rates. This allows us to 
# simulate how users would react to each design.
#
# **Use Case**: Optimizing the Amenity Display on a Listing Page.
#
# The "Arms" (Variations):
# * **Control**: The current list-based display.
# * **Icons**: Highlights top amenities with icons.
# * **Smart Summary**: AI-powered summary of amenities relevant to the guest's search.
# * **Visual Grid**: A more visual, grid-based layout.
#
# We also initialize a `BayesianBandit` object for each variation and set up some 
# data trackers.

# Define the true, unknown conversion rates for each variation.
# In a real experiment, we wouldn't know these.
true_conversion_rates = {
    "Control": 0.030,       # 3.0% booking rate
    "Icons": 0.032,         # 3.2% booking rate (a small improvement)
    "Smart Summary": 0.045, # 4.5% booking rate (the clear winner)
    "Visual Grid": 0.028    # 2.8% booking rate (worse than control)
}

bandits = [BayesianBandit(name) for name in true_conversion_rates.keys()]

num_trials = 20000 # Simulate 20,000 users visiting the listing page

# Data trackers for analysis
selections = {name: 0 for name in true_conversion_rates.keys()}
total_reward = 0
rewards_per_bandit = {name: 0 for name in true_conversion_rates.keys()}


# ## Step 3: Define a Helper Function for Plotting
#
# This function will help us visualize the posterior Beta distribution for each bandit 
# at different points in the simulation. This is key to understanding how the 
# algorithm is learning and updating its beliefs over time.

def plot_distributions(bandits, trial_number):
    """Helper function to visualize the posterior distributions of the bandits."""
    plt.figure(figsize=(12, 6))
    x = np.linspace(0, 0.1, 200) # Range of possible conversion rates
    for bandit in bandits:
        y = beta.pdf(x, bandit.alpha, bandit.beta)
        plt.plot(x, y, label=f'{bandit.name} (ECR: {bandit.estimated_conversion_rate:.3%})')
    plt.title(f'Posterior Distributions after {trial_number} Trials', fontsize=16)
    plt.xlabel('Conversion Rate', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.savefig(f'images/distribution_trial_{trial_number}.png')
    plt.close() # Prevents the plot from displaying in the console


# ## Step 4: Run the Simulation
#
# This is the core of the experiment. We loop for the total number of trials (users). 
# In each loop, we perform Thompson Sampling:
# 1.  **Sample**: Draw a random value from each bandit's posterior distribution.
# 2.  **Select**: Choose the bandit (variation) with the highest sample.
# 3.  **Simulate**: Simulate a user interacting with that variation and determine the 
#     outcome (a booking or not) based on its true conversion rate.
# 4.  **Update**: Update the chosen bandit's distribution with the outcome.
#
# We'll also periodically call our plotting function to see the learning process in action.

for i in range(num_trials):
    # 1. Sample from each bandit's posterior distribution.
    samples = [b.sample() for b in bandits]
    
    # 2. Choose the bandit with the highest sample.
    chosen_bandit_index = np.argmax(samples)
    chosen_bandit = bandits[chosen_bandit_index]
    
    # 3. Simulate the user's response to the chosen variation.
    true_rate = true_conversion_rates[chosen_bandit.name]
    reward = 1 if np.random.random() < true_rate else 0
    
    # 4. Update the chosen bandit's distribution with the result.
    chosen_bandit.update(reward)
    
    # Record results
    selections[chosen_bandit.name] += 1
    total_reward += reward
    rewards_per_bandit[chosen_bandit.name] += reward

    # Periodically plot the distributions to see the learning process
    if i + 1 in [100, 500, 2000, 5000, num_trials]:
        print(f"--- Plotting at trial {i+1} ---")
        plot_distributions(bandits, i + 1)

# ## Step 5: Analyze the Final Results
#
# With the simulation complete, we'll print a summary of the results. This includes 
# the total number of bookings and a breakdown for each variation, showing how many 
# times it was shown and what its observed conversion rate was compared to its true rate.

print("\n--- Simulation Complete ---")
print(f"Total Trials: {num_trials}")
print(f"Total Bookings (Reward): {total_reward}\n")

print("--- Results per Variation ---")
for bandit in bandits:
    pulls = selections[bandit.name]
    conv_rate = (rewards_per_bandit[bandit.name] / pulls) if pulls > 0 else 0
    print(
        f"  - {bandit.name}:\n"
        f"    - Pulled {pulls} times ({pulls/num_trials:.2%})\n"
        f"    - True Rate: {true_conversion_rates[bandit.name]:.3%}\n"
        f"    - Observed Rate: {conv_rate:.3%}"
    )


# ## Step 6: Plot Final Traffic Allocation
#
# Finally, we'll create a bar chart to visualize how the algorithm allocated traffic 
# over the course of the experiment. This plot clearly shows the explore-exploit 
# dynamic: after an initial period of exploration, the algorithm heavily exploited 
# the winning variation (`Smart Summary`) while allocating very little traffic to the 
# underperforming ones.

plt.figure(figsize=(12, 6))
plt.bar(selections.keys(), selections.values(), color=sns.color_palette("viridis", len(bandits)))
plt.title('Total Times Each Variation Was Shown', fontsize=16)
plt.ylabel('Number of Trials', fontsize=12)

plt.savefig('images/final_allocation.png')
plt.close()

