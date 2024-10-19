import matplotlib.pyplot as plt
import numpy as np
import SamplingLIB as sp
import matplotlib.animation as animation


# DONUT
donut_dim = 2  # dimension of the parametr
donut_mu = np.array([0.5, 1.0])  # prior mean

donut_sigma = 1.0  # prior standard deviation
def donut_G(u):  # forward model
    return (u[0]**2+u[1]**2)

donut_y = 2.0  # observation
donut_gamma = 0.5  # noise standard deviation

def donut_posterior(u): # Unnormalized posterior
    return np.exp(-((donut_y-donut_G(u))**2)/(2*donut_gamma**2)-np.dot(u-donut_mu, u-donut_mu)/(2*donut_sigma**2))

# Sampling instace creation
donut = sp.Sampling(posterior=donut_posterior, parametr_dimension=donut_dim)


# BANANA
banana_dim = 2  # dimension of the parametr
banana_mu = np.array([1.0, 1.0])  # prior mean

banana_sigma = 1.0  # prior standard deviation
def banana_G(u):  # forward model
    a = 1.0
    b = 10.0
    return (a-u[0])**2 + b*(u[1]-(u[0])**2)**2

banana_y = 3.0  # observation
banana_gamma = 1.0  # noise standard deviation

def banana_posterior(u): # Unnormalized posterior
    return np.exp(-((banana_y-banana_G(u))**2)/(2*banana_gamma**2)-np.dot(u-banana_mu, u-banana_mu)/(2*banana_sigma**2))

# Sampling instace creation
banana = sp.Sampling(posterior=banana_posterior, parametr_dimension=banana_dim)


# TESTING ANIMATION
MH_N = 10000
donut_initial = np.random.rand(2)
def MH_proposal_distribution(mu): return np.random.normal(mu, 1.0)
MH_buring = 0.0

donut_samples_MH = donut.MH(N = MH_N, initial = donut_initial, proposal_distribution = MH_proposal_distribution, burnin = MH_buring)

donut.animate_chain_movement(samples=donut_samples_MH[:1000, :])






# TESTING ANIMATION LINES

# Fixing random state for reproducibility
np.random.seed(19680801)


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 30
walks = [random_walk(num_steps) for index in range(40)]

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
lines = [ax.plot([], [], [])[0] for _ in walks]

# Setting the Axes properties
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=100)

plt.show()