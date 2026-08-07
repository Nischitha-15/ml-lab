import numpy as np
from hmmlearn.hmm import CategoricalHMM

# -------------------------------------------------
# Hidden States:
# 0 - Walking
# 1 - Running
# 2 - Resting
# -------------------------------------------------

states = ["Walking", "Running", "Resting"]

# Observation Symbols:
# 0 - Low
# 1 - Medium
# 2 - High

# -------------------------------------------------
# Given HMM Parameters
# -------------------------------------------------

# Initial State Probability Vector (π)
startprob = np.array([0.5, 0.3, 0.2])

# State Transition Probability Matrix (A)
transmat = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
])

# Emission Probability Matrix (B)
emissionprob = np.array([
    [0.6, 0.3, 0.1],   # Walking
    [0.1, 0.3, 0.6],   # Running
    [0.7, 0.2, 0.1]    # Resting
])

# -------------------------------------------------
# Create Categorical HMM Model
# -------------------------------------------------

model = CategoricalHMM(
    n_components=3,
    init_params=""
)

# Assign given parameters
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob


# -------------------------------------------------
# Observation Sequences
# Low = 0
# Medium = 1
# High = 2
# -------------------------------------------------

sequences = {
    "O (High, Medium, Low, High)": np.array([[2], [1], [0], [2]]),

    "O1 (Low, Medium, High, Medium)": np.array([[0], [1], [2], [1]]),

    "O2 (High, High, High, High)": np.array([[2], [2], [2], [2]]),

    "O3 (Low, Low, Low, Low)": np.array([[0], [0], [0], [0]]),

    "O4 (Medium, Medium, Medium, Medium)": np.array([[1], [1], [1], [1]])
}


print("Hidden Markov Model Observation Likelihoods\n")

for name, obs in sequences.items():

    # hmmlearn returns log likelihood
    log_likelihood = model.score(obs)

    # Convert log likelihood to normal probability
    likelihood = np.exp(log_likelihood)

    print(name)
    print("Log Likelihood :", log_likelihood)
    print("Likelihood     :", likelihood)
    print("-" * 50)