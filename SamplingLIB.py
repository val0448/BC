import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable
from matplotlib.animation import FuncAnimation

class Sampling:
    """
    A class for applying sampling methods on a given posterior function.

    Attributes:
    - posterior (Callable): The posterior function for a given problem.
    - dimension (int): The dimension of the parameter.

    Methods:
    - __init__(self, posterior: Callable, parametr_dimension: int): Constructor for Sampling class.
    - visualize(self, samples: np.ndarray): Visualizes the samples based on their dimensionality.
    - MH(self, N: int = 1000, initial: np.ndarray = None, proposal_distribution: Callable = None, burnin: float = 0): Random walk Metropolis-Hastings algorithm.

    Example usage:
    ```
    sampler = Sampling(posterior_function, 2)
    samples = sampler.MH(N=1000, initial=np.zeros(2), proposal_distribution=np.random.normal, burnin=0.2)
    sampler.visualize(samples)
    ```

    """

    def __init__(self, posterior: Callable, parametr_dimension: int):
        """
        Constructor for Sampling class.
        Creates an instance for a given posterior on which you can then apply sampling methods.

        Parameters:
        - posterior (Callable): posterior function for a given problem.
        - parametr_dimension (int): dimension of the parameter.

        Returns:
        - None

        Raises:
        - None

        Example usage:
        ```
        sampler = Sampling(posterior_function, 2)
        ```

        """
    
        self.posterior = posterior
        self.dimension = parametr_dimension

    def visualize(self, samples: np.ndarray):
        """
        Visualizes the samples based on their dimensionality.
        
        Parameters:
        - samples (np.ndarray): An array of samples to visualize.
        - If multiple chains (from DREAM), samples should have shape (N_gen, N_chains, dimension).
        - If single chain, samples should have shape (N_gen, dimension).
        
        Returns:
        - None

        Raises:
        - None

        Example usage:
        ``` 
        sampler = SamplingLIB()
        samples = np.random.randn(100, 2)
        sampler.visualize(samples)
        ```

        If the dimension of the samples is 1, a histogram of the samples is plotted.
        If the dimension of the samples is 2, a 2D scatter plot of the samples is plotted.
        If the dimension of the samples is greater than 2, scatter plots of all possible pairs of dimensions are plotted.
        For multiple chains (DREAM), samples are plotted with different colors for each chain.
        """
        
        # Check if samples come from multiple chains
        if samples.ndim == 3:
            N_gen, N_chains, dimension = samples.shape
        elif samples.ndim == 2:
            N_gen, dimension = samples.shape
            N_chains = 1  # Single chain
        else:
            raise ValueError("Samples array must be 2D or 3D")

        # If dimensionality is 1
        if dimension == 1:
            plt.figure()
            if N_chains > 1:
                for chain in range(N_chains):
                    plt.hist(samples[:, chain, 0], bins=200, alpha=0.5, label=f'Chain {chain+1}')
            else:
                plt.hist(samples[:, 0], bins=200)  # Single chain
            plt.xlabel('u')
            plt.ylabel('Density')
            if N_chains > 1:
                plt.legend(loc='upper right')  # Move the legend to the top right corner
            plt.show()

        # If dimensionality is 2
        elif dimension == 2:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            if N_chains > 1:
                colors = plt.cm.rainbow(np.linspace(0, 1, N_chains))  # Color for each chain
                for chain, color in zip(range(N_chains), colors):
                    ax[0].hist2d(samples[:, chain, 0], samples[:, chain, 1], bins=20, cmap="Blues", alpha=0.5)
                    ax[1].plot(samples[:, chain, 0], samples[:, chain, 1], '.', color=color, label=f'Chain {chain+1}')
            else:
                ax[0].hist2d(samples[:, 0], samples[:, 1], bins=20)  # 2D histogram for single chain
                ax[1].plot(samples[:, 0], samples[:, 1], '.')  # Scatter plot for single chain

            ax[0].set_xlabel('$u_1$')
            ax[0].set_ylabel('$u_2$')
            ax[1].set_xlabel('$u_1$')
            ax[1].set_ylabel('$u_2$')
            if N_chains > 1:
                ax[1].legend(loc='upper right')  # Move the legend to the top right corner
            plt.tight_layout()
            plt.show()

        # If dimensionality is greater than 2
        else:
            num_pairs = dimension * (dimension - 1) // 2
            fig, ax = plt.subplots(num_pairs, 1, figsize=(10, 5 * num_pairs))

            pair_index = 0
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    if N_chains > 1:
                        for chain in range(N_chains):
                            ax[pair_index].plot(samples[:, chain, i], samples[:, chain, j], '.', alpha=0.5, label=f'Chain {chain+1}')
                    else:
                        ax[pair_index].plot(samples[:, i], samples[:, j], '.')  # Single chain

                    ax[pair_index].set_xlabel(f'$u_{i+1}$')
                    ax[pair_index].set_ylabel(f'$u_{j+1}$')
                    ax[pair_index].set_aspect('equal')
                    if N_chains > 1:
                        ax[pair_index].legend(loc='upper right')  # Move the legend to the top right corner
                    pair_index += 1

            plt.tight_layout()
            plt.show()

    def animate_chain_movement(self, samples: np.ndarray, chain: int = 0, subsample_rate: int = 100, interval: int = 20):
        """
        Creates an animation of a selected chain's movement through the sample space.

        Parameters:
        - samples (np.ndarray): Array of shape (N_gen, N_chains, dimension) from the DREAM algorithm.
        - chain (int): Index of the chain to animate (default is 0).
        - subsample_rate (int): Subsample the chain for faster animation (default is 100).
        - interval (int): Time between frames in milliseconds (default is 20 ms for faster animation).

        Returns:
        - None

        Raises:
        - ValueError if the samples array is not 3D or dimension is not 2.
        """
        
        # Ensure the correct shape and dimensionality
        if samples.ndim != 3:
            raise ValueError("Expected samples array of shape (N_gen, N_chains, dimension)")
        
        N_gen, N_chains, dimension = samples.shape

        if dimension != 2:
            raise ValueError("Currently only 2D sample space is supported for animation.")
        
        # Extract the selected chain's samples and subsample it
        chain_samples = samples[::subsample_rate, chain, :]  # Subsample the chain
        N_subsampled_gen = chain_samples.shape[0]  # New number of generations after subsampling

        # Set up the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(np.min(chain_samples[:, 0]) - 0.1, np.max(chain_samples[:, 0]) + 0.1)
        ax.set_ylim(np.min(chain_samples[:, 1]) - 0.1, np.max(chain_samples[:, 1]) + 0.1)
        ax.set_xlabel('$u_1$')
        ax.set_ylabel('$u_2$')
        ax.set_title(f'Movement of Chain {chain+1} Through Sample Space')

        # Initialize the scatter plot and line plot
        scatter, = ax.plot([], [], 'o', color='blue', markersize=5)
        line, = ax.plot([], [], '-', color='gray', alpha=0.7)  # To track the movement

        # Function to initialize the plot
        def init():
            scatter.set_data([], [])
            line.set_data([], [])
            return scatter, line

        # Function to update the plot for each frame
        def update(frame):
            x_data = chain_samples[:frame+1, 0]
            y_data = chain_samples[:frame+1, 1]
            scatter.set_data(x_data[-1], y_data[-1])  # Update the current position
            line.set_data(x_data, y_data)  # Update the trajectory
            return scatter, line

        # Create the animation dynamically based on the subsampled chain
        ani = FuncAnimation(fig, update, frames=N_subsampled_gen, init_func=init, blit=True, interval=interval, repeat=False)

        plt.show()

    def MH(self, 
           N: int = 10000, 
           initial: np.ndarray = None, 
           proposal_distribution: Callable = None, 
           burnin: float = 0, 
           acc_rate :bool =False):
        """
        Random walk Metropolis-Hastings algorithm.

        Parameters:
        - N (int): number of samples
        - initial (np.ndarray): initial sample
        - proposal_distribution (Callable): function to draw samples from
        - burnin (float): length of the burnin period on a scale 0 to 1 
        - acc_rate (bool): if True, returns the acceptance rate 

        Returns:
        - samples (np.ndarray): N samples
        - acceptance rate (float): acceptance rate

        Example usage:
        ```
        sampler = SamplingLIB()
        samples = sampler.MH(N=1000, initial=np.zeros(2), proposal_distribution=np.random.normal, burnin=0.2)
        ```

        """
        if initial is None:
            initial = np.zeros(self.dimension)

        if proposal_distribution is None:
            def proposal_distribution(mu): return np.random.normal(mu, 1)

        if burnin < 0 or burnin > 1:
            raise ValueError('Burnin period should be between 0 and 1.')
        
        if acc_rate:
            acc = 0

        burnin_index = int(burnin * N)
        samples = np.zeros((N, self.dimension))
        current = initial
        current_likelihood = self.posterior(current)

        for i in range(N + burnin_index):
            proposal = proposal_distribution(current)
            proposal_likelihood = self.posterior(proposal)
            acceptance_probability = min(1, proposal_likelihood / current_likelihood)
            if np.random.rand() < acceptance_probability:
                if acc_rate:
                    acc += 1
                current = proposal
                current_likelihood = proposal_likelihood
            if i >= burnin_index:
                samples[i-burnin_index, :] = current

        if acc_rate:
            return samples, acc / N
        else:
            return samples
        
    def AM(self, 
           N: int = 10000, 
           initial: np.ndarray = None, 
           proposal_cov: np.ndarray = None,
           update_step: int = 1,
           burnin: float = 0, 
           acc_rate: bool = False,
           cov_matrix: bool = False
           ):
        """
        Adaptive Metropolis algorithm.

        Parameters:
        - N (int): number of samples
        - initial (np.ndarray): initial sample
        - proposal_cov (np.ndarray): initial covariance matrix for the proposal distribution
        - update_step (int): number of samples between covariance matrix updates
        - burnin (float): length of the burnin period on a scale 0 to 1  
        - acc_rate (bool): if True, returns the acceptance rate
        - cov_matrix (bool): if True, returns the covariance matrix

        Returns:
        - samples (np.ndarray): N samples
        - acceptance rate (float): acceptance rate

        Example usage:
        ```
        sampler = SamplingLIB()
        samples = sampler.AM(N=1000, initial=np.zeros(2), proposal_distribution=np.random.normal, burnin=0.2)
        ```

        """
        if initial is None:
            initial = np.zeros(self.dimension)

        if burnin < 0 or burnin > 1:
            raise ValueError('Burnin period should be between 0 and 1.')
        
        if acc_rate:
            acc = 0

        burnin_index = int(burnin * N)
        samples = np.zeros((N, self.dimension))
        current = initial
        current_likelihood = self.posterior(current)
        sd = (2.4**2) / self.dimension
        epsilon = 0.01

        if proposal_cov is None:
            proposal_cov = sd * np.eye(self.dimension)

        if cov_matrix:
            cov_matrix_list = [proposal_cov]

        for i in range(N + burnin_index):
            proposal = np.random.multivariate_normal(current, proposal_cov)
            proposal_likelihood = self.posterior(proposal)
            acceptance_probability = min(1, proposal_likelihood / current_likelihood)
            if np.random.rand() < acceptance_probability:
                if acc_rate:
                    acc += 1
                current = proposal
                current_likelihood = proposal_likelihood
            if i >= burnin_index:
                samples[i-burnin_index, :] = current
                if i>= burnin_index + 1:
                    proposal_cov = (sd * np.cov(samples[:i-burnin_index+1].T)) + (sd * epsilon * np.eye(self.dimension))
                    if cov_matrix:
                        cov_matrix_list.append(proposal_cov)
                    """
                    average_X_curr = np.mean(samples, axis=0)
                    proposal_cov = (i-1/i) * proposal_cov + (sd/i) * (i * average_X_prev * average_X_prev.T - (i + 1) * average_X_curr * average_X_curr.T + current * current.T + epsilon * np.eye(self.dimension))
                    if cov_matrix:
                        cov_matrix_list.append(proposal_cov)
                    average_X_prev = average_X_curr
                else:
                    average_X_prev = current
                    """

        if acc_rate and cov_matrix:
            return samples, acc / N, cov_matrix_list
        elif acc_rate:
            return samples, acc / N
        elif cov_matrix:
            return samples, cov_matrix_list
        else:
            return samples
        
    def DRAM(self, 
             N: int = 10000, 
             initial: np.ndarray = None, 
             proposal_cov: np.ndarray = None,
             update_step: int = 1,
             burnin: float = 0, 
             acc_rate: bool = False,
             cov_matrix: bool = False,
             beta: float = 0.5):
        """
        Delayed Rejection Adaptive Metropolis (DRAM) algorithm.

        Parameters:
        - N (int): number of samples
        - initial (np.ndarray): initial sample
        - proposal_cov (np.ndarray): initial covariance matrix for the proposal distribution
        - update_step (int): number of samples between covariance matrix updates
        - burnin (float): length of the burn-in period on a scale from 0 to 1  
        - acc_rate (bool): if True, returns the acceptance rate
        - cov_matrix (bool): if True, returns the covariance matrix
        - beta (float): scaling factor for second-stage proposal (0 < beta < 1)

        Returns:
        - samples (np.ndarray): N samples
        - acceptance rate (float): acceptance rate

        Example usage:
        ```
        sampler = SamplingLIB(posterior_func=my_posterior, dimension=2)
        samples = sampler.DRAM(N=1000, initial=np.zeros(2), burnin=0.2)
        ```
        """
        if initial is None:
            initial = np.zeros(self.dimension)

        if burnin < 0 or burnin > 1:
            raise ValueError('Burn-in period should be between 0 and 1.')

        if acc_rate:
            acc = 0

        burnin_index = int(burnin * N)
        samples = np.zeros((N, self.dimension))
        current = initial
        current_likelihood = self.posterior(current)
        sd = (2.4**2) / self.dimension
        epsilon = 1e-5

        if proposal_cov is None:
            proposal_cov = sd * np.eye(self.dimension)

        if cov_matrix:
            cov_matrix_list = [proposal_cov]

        for i in range(N + burnin_index):
            # First-stage proposal
            proposal = np.random.multivariate_normal(current, proposal_cov)
            proposal_likelihood = self.posterior(proposal)
            acceptance_probability = min(1, proposal_likelihood / current_likelihood)

            if np.random.rand() < acceptance_probability:
                if acc_rate:
                    acc += 1
                current = proposal
                current_likelihood = proposal_likelihood
            else:
                # Second-stage proposal (delayed rejection)
                scaled_cov = beta**2 * proposal_cov
                proposal2 = np.random.multivariate_normal(current, scaled_cov)
                proposal2_likelihood = self.posterior(proposal2)

                acceptance_probability2 = min(1, 
                    (proposal2_likelihood / current_likelihood) *
                    (1 - min(1, proposal_likelihood / proposal2_likelihood)) /
                    (1 - acceptance_probability)
                )

                if np.random.rand() < acceptance_probability2:
                    current = proposal2
                    current_likelihood = proposal2_likelihood

            # Burn-in period handling and storing samples
            if i >= burnin_index:
                samples[i - burnin_index, :] = current

                # Update the covariance matrix adaptively
                if i >= burnin_index + 1 and (i - burnin_index) % update_step == 0:
                    sample_cov = np.cov(samples[:i-burnin_index+1].T)
                    proposal_cov = sd * sample_cov + sd * epsilon * np.eye(self.dimension)
                    if cov_matrix:
                        cov_matrix_list.append(proposal_cov)

        if acc_rate and cov_matrix:
            return samples, acc / N, cov_matrix_list
        elif acc_rate:
            return samples, acc / N
        elif cov_matrix:
            return samples, cov_matrix_list
        else:
            return samples
        
    def DREAM(self, 
              N: int = 10000, 
              initial: np.ndarray = None, 
              burnin: float = 0.1, 
              delta: int = 3, 
              CR: float = 0.9,
              outlier_detection: bool = True,
              adapt_CR: bool = True,
              chains: int = 10,
              history_length: int = None):
        """
        Differential Evolution Adaptive Metropolis (DREAM) algorithm.

        Parameters:
        - N (int): number of generations (iterations) for the algorithm
        - initial (np.ndarray): initial samples for the chains
        - burnin (float): burn-in period as a fraction of N_gen
        - delta (int): number of pairs used to generate proposals
        - CR (float): initial crossover probability
        - outlier_detection (bool): if True, detect and correct outlier chains
        - adapt_CR (bool): if True, adapt the crossover probability during burn-in
        - chains (int): number of chains

        Returns:
        - samples (np.ndarray): generated samples after burn-in
        - acceptance rates (list): acceptance rate for each chain
        """
        if initial is None:
            initial = np.random.rand(chains, self.dimension)

        if burnin < 0 or burnin > 1:
            raise ValueError('Burn-in period should be between 0 and 1.')
        
        if history_length is None:
            history_length = min(100, max(10, N//100))

        burnin_index = int(burnin * N)
        samples = np.zeros((N, chains, self.dimension))
        current = initial.copy()
        prob = np.array([self.posterior(x) for x in current])
        prob_history = np.zeros((N, chains))
        prob_history[0] = prob
        acceptance_counts = np.zeros(chains)

        gamma = 2.38 / np.sqrt(2 * delta * self.dimension)
        eps = 1e-6

        def differential_evolution_proposal(i, CR, current):
            """
            Generate a proposal using Differential Evolution (DE).
            
            Parameters:
            - i (int): Index of the chain for which to generate the proposal.
            - CR (float): Crossover probability.
            - current (np.ndarray): Current states of all chains.
            
            Returns:
            - z (np.ndarray): Proposed state.
            """

            chains = current.shape[0]
            dimension = current.shape[1]

            # Select two distinct chains randomly, ensuring they are not the current chain
            indices = np.random.choice([j for j in range(chains) if j != i], size=2, replace=False)
            x1, x2 = current[indices[0]], current[indices[1]]

            # Compute the difference vector
            diff = x1 - x2

            # Generate the proposed point
            z = current[i] + gamma * diff + np.random.normal(0, eps, size=dimension)

            # Apply subspace sampling using the crossover probability CR
            crossover_mask = np.random.rand(dimension) < CR
            z = np.where(crossover_mask, z, current[i])

            # Ensure that the proposal does not diverge too far
            # Clip values or reflect within a reasonable range if necessary
            max_bound = 1e10
            min_bound = -1e10
            z = np.clip(z, min_bound, max_bound)

            return z

        def outlier_detection_step(prob_history, prob, current, chains):
            """
            Detect and correct outlier chains based on the log posterior values.

            Parameters:
            - log_posterior (np.ndarray): Log posterior values of all chains.
            - current (np.ndarray): Current states of all chains.

            Returns:
            - current (np.ndarray): Updated states of all chains.
            """

            prob_mean = np.mean(prob_history)
            prob_std = np.std(prob_history)
            outliers = np.abs(prob - prob_mean) > 2 * prob_std
            for i in np.where(outliers)[0]:
                chain = [np.argmax(prob)]  # Currently best chain
                if np.random.rand() > prob[chain]:
                    chain = np.random.choice([x for x in range(chains) if x != i])  # Random chain
                current[i] = current[np.argmax(prob_mean)]
                prob[i] = prob[np.argmax(prob_mean)]
            return current, prob
        
        def adapt_crossover_probability(gen, acceptance_rates):
            # linearly decrease CR
            return max(0.1, 1 - gen / 10000)

        for gen in range(N):
            for i in range(chains):
                z = differential_evolution_proposal(i, CR, current)
                prob_z = self.posterior(z)

                alpha = min(1, prob_z / prob[i])

                if np.random.rand() < alpha:
                    current[i] = z
                    prob[i] = prob_z
                    acceptance_counts[i] += 1

            if outlier_detection and gen > history_length:
                m = max(0, gen - history_length)
                n = min(N, gen)
                current, prob = outlier_detection_step(prob_history[m:n], prob, current, chains)

            if adapt_CR and gen < burnin_index:
                CR = adapt_crossover_probability(gen, acceptance_counts / (gen + 1))

            prob_history[gen] = prob
            samples[gen] = current

        samples = samples[burnin_index:]  # Remove burn-in samples
        acceptance_rates = acceptance_counts / N
        return samples, acceptance_rates

    