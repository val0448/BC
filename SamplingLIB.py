import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import matplotlib.animation as animation
from scipy.stats import gaussian_kde, entropy
from IPython.display import HTML

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
            def proposal_distribution(mu): return np.random.normal(mu, 1.0)

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

    def AM2(self, initial_cov=None, epsilon=1e-5, scale_factor=None, burnin=0.0, update_step=1):
        """
        Creates an instance of the Adaptive Metropolis algorithm for a given Sampling instance.

        Parameters:
        - initial_cov (np.ndarray): initial covariance matrix for the proposal distribution
        - epsilon (float): small value to ensure positive definiteness of the covariance matrix
        - scale_factor (float): scaling factor for the covariance matrix
        - burnin (float): length of the burn-in period on a scale from 0 to 1
        - update_step (int): number of samples between covariance matrix updates

        Returns:
        - AdaptiveMetropolis: an instance of the Adaptive Metropolis algorithm
        """

        return AM(self, initial_cov, epsilon, scale_factor, burnin, update_step)     

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
        - cov_matrix (bool): if True, returns the covariance matrices

        Returns:
        - samples (np.ndarray): N samples
        - acceptance rate (float): acceptance rate
        - cov_matrix_list (list): list of covariance matrices

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
                if (i>= burnin_index + 1) and (i - burnin_index) % update_step == 0:
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
                    (1 - acceptance_probability))

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
        - history_length (int): length of history for outlier detection

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
            - prob_history (np.ndarray): Log posterior values for all chains.
            - prob (np.ndarray): Log posterior values for the current generation.
            - current (np.ndarray): Current states of all chains.
            - chains (int): Number of chains.

            Returns:
            - current (np.ndarray): Corrected states of all chains.
            - prob (np.ndarray): Corrected log posterior values for the current generation
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
            """
            Adapt the crossover probability based on the acceptance rates.

            Parameters:
            - gen (int): Current generation.
            - acceptance_rates (np.ndarray): Acceptance rates for all chains.

            Returns:
            - CR (float): Adapted crossover probability.
            """
            
            # linearly decrease CR
            return max(0.3, 1 - gen / 10000)

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

    def visualize(self, visuals: list = [], grid: tuple = None, ranges: list = None, max_points: int = 100):
        """
        Visualizes the samples based on their dimensionality, and compares to posterior.

        Parameters:
        - visuals (list): A list containing the visuals to display (samples or posterior function).
        - grid (tuple): Grid for posterior and histogram comparisons (default is None).
        - ranges (list): List of min and max bounds for each dimension (default is None).
        - max_points (int): Maximum number of grid points for the longest range (default is 100).

        Returns:
        - None

        Example usage:
        ```
        visualize(samples)
        visualize(posterior, ranges=[(-5, 5), (-5, 5)], max_points=1000)
        ```
        """

        if visuals == []:
            visuals = [self.posterior]

        # Determine how many axes are required:
        num_axes = sum(2 if (isinstance(visual, np.ndarray) and self.dimension < 3) else 1 for visual in visuals)

        # Create subplots based on the required number of axes
        fig, axes = plt.subplots(1, num_axes, figsize=(min(18, 6*num_axes), 6))

        # If only one axis is required, make axes iterable
        if num_axes == 1:
            axes = [axes]

        axis_iter = iter(axes)  # Create an iterator over axes

        for visual in visuals:
            if isinstance(visual, np.ndarray):
                if grid is None:
                    # Create grid for posterior and histogram comparisons
                    grid = self.create_grid(visual=visual, ranges=ranges, max_points=max_points)
                # Visualize samples in histograms and scatter plots (2 separate plots)
                self.visualize_samples_hist(visual, grid, ax=next(axis_iter), show = False)
                if self.dimension < 3:
                    self.visualize_samples_scatter(visual, ax=next(axis_iter), show = False)
            elif isinstance(visual, dict) and visual.get("is_kde"):
                # If it's a KDE approximation
                self.visualize_kde(visual['data'], grid, ax=next(axis_iter))
            elif callable(visual):
                if grid is None:
                    print(type(ranges))
                    # Create grid for posterior and histogram comparisons
                    grid = self.create_grid(visual=visual, ranges=ranges, max_points=max_points)
                # Visualize the posterior (1 plot)
                self.visualize_posterior(visual, grid, ax=next(axis_iter), show = False)

        plt.tight_layout()
        plt.show()

    def visualize_samples_hist(self, samples: np.ndarray, grid: tuple = None, ax = None, show = True):
        """
        Visualizes histograms for the samples based on their dimensionality.
        
        Parameters:
        - samples (np.ndarray): An array of samples to visualize.
        - grid (np.ndarray): Grid to base the histograms on (default is None).
        - ax (matplotlib.axes.Axes): Axis to plot the histograms on (default is None).
        - show (bool): If True, displays the plot (default is True).
        
        Returns:
        - None

        Example usage:
        ```
        visualize_samples_hist(samples, grid, ax)
        visualize_samples_hist(samples, show=True)
        ```
        """

        dimension = samples.shape[-1]

        # Check if samples come from multiple chains
        if samples.ndim == 3:
            samples = samples.transpose(1, 0, 2).reshape(-1, dimension)
        elif samples.ndim > 3:
            raise ValueError("Samples array must be 2D or 3D")

        # Create a grid if not provided
        if grid is None:
            grid = self.create_grid(visual=samples)

        # Create an axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6) if dimension == 1 else (6, 6))

        if dimension == 1:
            # 1D histogram
            ax.hist(samples[:, 0], bins=grid[2][0], alpha=0.5)
            ax.set_xlabel('u')
            ax.set_ylabel('Density')

        elif dimension == 2:
            # 2D histogram
            ax.hist2d(samples[:, 0], samples[:, 1], bins=grid[2][:2], range=[grid[1][0], grid[1][1]])
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('$u_2$')

        else:
            # N-dimensional Case: Use pairplots
            for i in range(dimension):
                for j in range(dimension):
                    if i == j:
                        # Diagonal: 1D histogram for individual dimensions
                        ax[i, j].hist(samples[:, i], bins=grid[2][i], alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                    else:
                        # Off-diagonal: 2D scatter or density plot
                        ax[i, j].scatter(samples[:, i], samples[:, j], s=5, alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                        ax[i, j].set_ylabel(f'$u_{j+1}$')

            plt.tight_layout()

        if show:
            plt.show()

    def visualize_samples_scatter(self, samples: np.ndarray, ax = None, show = True):
        """
        Visualizes 1D or 2D scatter plots for samples.
        
        Parameters:
        - samples (np.ndarray): An array of samples to visualize.
        - ax (matplotlib.axes.Axes): Axis to plot the samples on (default is None).
        - show (bool): If True, displays the plot (default is True).
        
        Returns:
        - None

        Example usage:
        ```
        visualize_samples_scatter(samples, ax)
        visualize_samples_scatter(samples, show=True)
        ```
        """

        # Check if samples come from multiple chains
        if samples.ndim == 3:
            N_gen, N_chains, dimension = samples.shape
        elif samples.ndim == 2:
            N_gen, dimension = samples.shape
            N_chains = 1  # Single chain
        else:
            raise ValueError("Samples array must be 2D or 3D")

        # Create an axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6) if dimension == 1 else (6, 6))

        if dimension == 1:
            # 1D scatter plot: Plot samples against their index
            if N_chains > 1:
                colors = plt.cm.rainbow(np.linspace(0, 1, N_chains))
                for chain, color in zip(range(N_chains), colors):
                    ax.plot(range(N_gen), samples[:, chain, 0], '.', color=color, label=f'Chain {chain+1}')
            else:
                ax.plot(range(N_gen), samples[:, 0], '.', alpha=0.5)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('$u_1$')
            ax.set_title('1D Scatter Plot')
            if N_chains > 1:
                ax.legend(loc='upper right')

        elif dimension == 2:
            # 2D scatter plot
            if N_chains > 1:
                colors = plt.cm.rainbow(np.linspace(0, 1, N_chains))
                for chain, color in zip(range(N_chains), colors):
                    ax.plot(samples[:, chain, 0], samples[:, chain, 1], '.', color=color, label=f'Chain {chain+1}')
            else:
                ax.plot(samples[:, 0], samples[:, 1], '.', alpha=0.5)
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('$u_2$')
            ax.set_title('2D Scatter Plot')
            if N_chains > 1:
                ax.legend(loc='upper right')

        else:
            raise ValueError("This function only handles 1D and 2D cases.")

        if show:
            plt.show()

    def visualize_posterior(self, posterior: np.ndarray, grid: tuple = None, ax = None, ranges: list = None, max_points: int = 100 , show = True):
        """
        Visualizes the posterior distribution on a given grid.
        
        Parameters:
        - posterior (np.ndarray): Evaluated posterior distribution.
        - grid (np.ndarray): Grid used for visualizing the posterior.
        - ax (matplotlib.axes.Axes): Axis to plot the posterior on (default is None).
        - ranges (list): List of min and max bounds for each dimension (default is None).
        - max_points (int): Maximum number of grid points for the longest range (default is 100).
        - show (bool): If True, displays the plot (default is True).
        
        Returns:
        - None

        Example usage:
        ```
        visualize_posterior(posterior, grid, ax)
        visualize_posterior(posterior, grid, ranges=[(-5, 5), (-5, 5)], max_points=1000, show=True)
        ```
        """

        # Create a grid if not provided
        if grid is None:
            grid = self.create_grid(visual=posterior, ranges=ranges, max_points=max_points)

        # Create an axis if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6) if dimension == 1 else (6, 6))

        # Evaluate posterior on the grid
        posterior_values = np.array([posterior(point) for point in grid[0]])

        dimension = grid[0].shape[1]
        if dimension == 1:
            ax.plot(grid[0][:, 0], posterior_values, label='Posterior')
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('Density')
            ax.set_title('Posterior')

        elif dimension == 2:
            posterior_reshaped = posterior_values.reshape((grid[2][0], grid[2][1]))
            ax.imshow(posterior_reshaped.T, extent=[grid[0][:, 0].min(), grid[0][:, 0].max(),
                                                    grid[0][:, 1].min(), grid[0][:, 1].max()],
                    origin='lower', aspect='auto')
            ax.set_title('Posterior')

        else:
            # N-dimensional Case: Use pairplots
            fig, ax = plt.subplots(dimension, dimension, figsize=(15, 15))
            for i in range(dimension):
                for j in range(dimension):
                    if i == j:
                        marginals = np.array([self.posterior([grid[0][k, i] if k == i else 0 for k in range(dimension)]) for k in range(len(grid[0]))])
                        ax[i, j].plot(marginals)
                    else:
                        ax[i, j].scatter(grid[0][:, i], grid[0][:, j], s=5, alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                        ax[i, j].set_ylabel(f'$u_{j+1}$')

            plt.tight_layout()

        if show:
            plt.show()

    def visualize_kde(self, kde_approximation, grid: tuple, ax):
        """
        Visualizes the KDE approximation on the grid for N-dimensional cases.
        Only used from sampling_quality method.

        Parameters:
        - kde_approximation (np.ndarray): KDE approximation of the samples.
        - grid (tuple): The grid used to evaluate the KDE.
        - ax (matplotlib.axes.Axes): Axis to plot the KDE on.

        Returns:
        - None

        Example usage:
        ```
        visualize_kde(kde_approximation, grid, ax)
        ```
        """
        if self.dimension == 1:
            # 1D Case: Plot KDE approximation as a line plot
            ax.plot(grid[0][:, 0], kde_approximation, label='KDE Approximation')
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('Density')
            ax.set_title('KDE Approximation')

        elif self.dimension == 2:
            # 2D Case: Use imshow for KDE approximation heatmap
            kde_approximation_reshaped = kde_approximation.reshape((grid[2][0], grid[2][1]))
            ax.imshow(kde_approximation_reshaped.T, extent=[grid[0][:, 0].min(), grid[0][:, 0].max(),
                                                            grid[0][:, 1].min(), grid[0][:, 1].max()],
                    origin='lower', aspect='auto')
            ax.set_title('KDE Approximation')

        else:
            # N-dimensional Case: Use pair plots
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        # Diagonal: 1D KDE approximation
                        ax[i, j].plot(grid[0][:, i], kde_approximation, label='KDE', alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                    else:
                        # Off-diagonal: 2D scatter plot
                        ax[i, j].scatter(grid[0][:, i], grid[0][:, j], s=5, alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                        ax[i, j].set_ylabel(f'$u_{j+1}$')

        plt.tight_layout()

    def create_grid(self, visual = None, max_points: int = 100, min_points: int = 10, margin: float = 0.1, ranges: list = None):
        """
        Creates a grid for comparing samples or posterior, with adaptive num_points for each dimension
        based on relative ranges and scaling for total number of dimensions.
        
        Parameters:
        - visual (np.ndarray or Callable): Samples or posterior function to visualize.
        - max_points (int): Maximum number of grid points for the longest range (default is 100).
        - min_points (int): Minimum number of grid points for the shortest range (default is 10).
        - margin (float): Margin to extend the grid beyond the sample bounds (default is 10%).
        - ranges (list): List of min and max bounds for each dimension (default is

        Returns:
        - grid (np.ndarray): Multi-dimensional grid of shape (num_points^dimension, dimension).
        - ranges (list of tuples): List of min and max bounds for each dimension.
        - dim_num_points (list): Number of points per dimension for the grid.

        Raises:
        - ValueError if ranges are not provided for each dimension.

        Example usage:
        ```
        grid, ranges, dim_num_points = create_grid(samples)
        grid, ranges, dim_num_points = create_grid(posterior_function, ranges=[(-5, 5), (-5, 5)])
        ```
        """

        dim_ranges = []
        if ranges is None:
            ranges = []

        # Case 1: If samples are provided, use them to calculate the grid bounds
        if isinstance(visual, np.ndarray):
            for dim in range(self.dimension):
                dim_min = np.min(visual[:, dim])
                dim_max = np.max(visual[:, dim])
                dim_range = dim_max - dim_min
                dim_ranges.append(dim_range)

                # Add margin to extend the bounds
                range_min = dim_min - margin * dim_range
                range_max = dim_max + margin * dim_range
                ranges.append((range_min, range_max))
        
        # Case 2: If a posterior function is provided, use default heuristic ranges
        elif (visual is None or callable(visual)):
            if ranges == []:
                for dim in range(self.dimension):
                    ranges.append((-5, 5))  # Use default range for posterior
                    dim_ranges.append(10)  # Example range for posterior
            else:
                if len(ranges) != self.dimension:
                    raise ValueError("Ranges must be provided for each dimension.")
                else:
                    dim_ranges = [r[1] - r[0] for r in ranges]

        # Find the longest range and normalize other ranges relative to it
        max_range = max(dim_ranges)

        # Calculate num_points for each dimension, relative to the max range and the number of dimensions
        dim_num_points = []
        for dim_range in dim_ranges:
            relative_scale = dim_range / max_range
            num_points = int(relative_scale * (max_points - min_points) + min_points)
            num_points = int(num_points * (1 / self.dimension))  # Scale down based on number of dimensions
            num_points = max(min_points, min(max_points, num_points))  # Ensure within bounds
            dim_num_points.append(num_points)

        # Create a grid using the determined ranges and num_points per dimension
        grid_ranges = [np.linspace(r[0], r[1], n) for r, n in zip(ranges, dim_num_points)]
        mesh = np.meshgrid(*grid_ranges, indexing='ij')

        # Stack the meshgrid into a list of points in shape (num_points^dimension, dimension)
        grid = np.vstack([m.ravel() for m in mesh]).T

        return grid, ranges, dim_num_points

    def animate_chain_movement(self, samples: np.ndarray, subsample_rate: int = 1, time: int = 60):
        """
        Creates an animation of a selected chain's movement through the sample space.

        Parameters:
        - samples (np.ndarray): Array of shape (N_gen, dimension) from the sampling algorithm.
        - subsample_rate (int): Subsample the chain (default is 1).
        - time (int): Time in seconds for the animation (default is 60).

        Returns:
        - None

        Raises:
        - ValueError if the samples array is not 2D or dimension is not 2.

        Example usage:
        ```
        sampler = SamplingLIB(posterior_func=my_posterior, dimension=2)
        samples = sampler.MH(N=1000, initial=np.zeros(2))
        sampler.animate_chain_movement(samples)
        ```
        """

        # Ensure the correct shape and dimensionality
        if samples.ndim != 2:
            raise ValueError("Expected samples array of shape (N_gen, dimension)")

        N_gen, dimension = samples.shape

        if dimension != 2:
            raise ValueError("Currently only 2D sample space is supported for animation.")

        # Subsample the data
        #samples = samples[::subsample_rate]

        # Calculate the number of frames and the interval (in ms) between frames
        num_frames = len(samples)
        interval = (time / num_frames) * 1000  # Convert seconds to milliseconds per frame

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(np.min(samples[:, 0]) - 0.1, np.max(samples[:, 0]) + 0.1)
        ax.set_ylim(np.min(samples[:, 1]) - 0.1, np.max(samples[:, 1]) + 0.1)

        # Plot setup: initialize an empty scatter plot
        scatter = ax.scatter([], [], c='blue')

        # Initialize function for animation
        def init():
            scatter.set_offsets(np.empty((0, 2)))  # Initialize empty 2D array
            return scatter,

        # Update function for animation
        def update(frame):
            # Set the data for the current frame (i.e., up to the current sample)
            scatter.set_offsets(samples[:frame])
            return scatter,

        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=interval, blit=True)

        # Display the animation
        return HTML(anim.to_jshtml())  # Return the HTML representation for display

    def sampling_quality(self, samples: np.ndarray, visualise: bool = False):
        """
        Compares KDE approximation from samples with the actual posterior using KL divergence.
        
        Parameters:
        - samples (np.ndarray): Generated samples from the sampling algorithm.
        - visualise (bool): If True, visualizes the KDE approximation and the posterior (default is False).
        
        Returns:
        - kl_divergence (float): KL Divergence value between the KDE approximation and the posterior.

        Example usage:
        ```
        sampler = SamplingLIB(posterior_func=my_posterior, dimension=2)
        samples = sampler.MH(N=1000, initial=np.zeros(2))
        kl_divergence = sampler.sampling_quality(samples)
        ```
        """

        dimension = samples.shape[-1]

        # Check if samples come from multiple chains
        if samples.ndim == 3:
            samples = samples.transpose(1, 0, 2).reshape(-1, dimension)
        elif samples.ndim > 3:
            raise ValueError("Samples array must be 2D or 3D")

        # Create a grid using the samples (adjustable for N dimensions)
        grid = self.create_grid(visual=samples)

        # Apply KDE to the samples
        kde = gaussian_kde(samples.T)  # Transpose to match expected shape

        # Evaluate the KDE on the grid
        kde_approximation = kde(grid[0].T)  # Evaluate KDE on the grid points
        kde_approximation_flat = kde_approximation.flatten()

        # Evaluate the target posterior on the grid
        target_pdf = np.array([self.posterior(u) for u in grid[0]])
        target_pdf_flat = target_pdf.flatten()

        # Normalize both distributions to ensure they sum to 1
        kde_approximation_flat /= np.sum(kde_approximation_flat)
        target_pdf_flat /= np.sum(target_pdf_flat)

        # To avoid division by zero and log issues, add a small epsilon to the values
        epsilon = 1e-10
        kde_approximation_flat += epsilon
        target_pdf_flat += epsilon

        # Normalize both distributions again after adding epsilon
        kde_approximation_flat /= np.sum(kde_approximation_flat)
        target_pdf_flat /= np.sum(target_pdf_flat)

        # Calculate Kullback-Leibler divergence
        kl_divergence = entropy(target_pdf_flat, kde_approximation_flat)

        if visualise:
            # Visualize the KDE approximation and the target posterior
            self.visualize(visuals=[{"data": kde_approximation, "is_kde": True}, self.posterior], grid=grid)

        # Return KL Divergence value
        return kl_divergence
    
    def optimalize(self, method: str = None, parametrs: list = [], parametr: str = None, range: tuple = None, metric: str = 'time'):
        # this function will find the most optimal value for a given parametr, according to the chosen metric
        parametr_values = np.linsapce(range[0], range[1], 100)
        results = []

        if metric == 'time':
            pass

        for par in parametr_values:
            # time start
            samples = self.method(parametrs, parametr = par)
            # time end
            results.append()

        # plot parametr values against results

        return np.min(results)
    
    def benchmark(self, method, parametrs, metric: str = None):
        # this function will benchmark the chosen method
        
        # we will be looking at autocorrelation length, ESS (effective sample size), time complexity
        # we will be looking at the chosen method for the chosen parametrs
        # result data will be stored in table and plot showcasing the benchmark

        return # table, plot

    def compare(self, methods_dic: dict = {}, metric: str = None):
        # this function will just be used to call benchmark on all methods and then merge the result data intonice and comprehensive form
        tables = []
        plots = []

        for method, parametrs in methods_dic:
            tmp = self.benchmark(self, method, parametrs, metric)
            tables.append(tmp[0])
            plots.append(tmp[1])

        # merge tables

        # merge and show plots

        return tables

class MH(Sampling):
    pass

class AM(Sampling):
    """
    Adaptive Metropolis algorithm for sampling from a target distribution.

    Attributes:
    - epsilon (float): Small value to ensure positive definiteness of the covariance matrix.
    - scale_factor (float): Scaling factor for the covariance matrix.
    - mean (np.ndarray): Mean of the samples.
    - burnin (float): Fraction of the total number of samples to discard as burn-in.
    - update_step (int): Number of samples between covariance matrix updates.
    - C (list): List of covariance matrices.
    - acc_rate (float): Acceptance rate of the samples.
    - samples (np.ndarray): Generated samples.

    Example usage:
    ```
    distribution = Sampling(posterior=my_posterior, dimension=2)
    sampler = AM(distribution, initial_cov=np.eye(2), epsilon=1e-5, scale_factor=None, burnin=0.2, update_step=1)
    samples = sampler.sample(N=10000)
    ```
    """

    def __init__(self, distribution, initial_cov, epsilon, scale_factor, burnin, update_step):
        """
        Initialize the Adaptive Metropolis algorithm.

        Parameters:
        - distribution (Sampling): Sampling object with the target distribution.
        - initial_cov (np.ndarray): Initial covariance matrix for the proposal distribution.
        - epsilon (float): Small value to ensure positive definiteness of the covariance matrix.
        - scale_factor (float): Scaling factor for the covariance matrix.
        - burnin (float): Fraction of the total number of samples to discard as burn-in.
        - update_step (int): Number of samples between covariance matrix updates.
        """

        # Inherit attributes from Sampling class
        super().__init__(distribution.posterior, distribution.dimension)

        # Additional attributes for AM
        self.epsilon = epsilon
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.mean = np.zeros(self.dimension)
        self.burnin = burnin
        self.update_step = update_step
        self.C = [np.eye(self.dimension)] if initial_cov is None else [initial_cov]
        self.acc_rate = 0.0
        self.samples = None

    def sample(self, x0 = None, N = 10000):
        """
        Run the Adaptive Metropolis algorithm for n_samples iterations.

        Parameters:
        - x0 (np.ndarray): Initial point for the chain.
        - N (int): Number of samples to generate.

        Returns:
        - None
        """

        current = np.zeros(self.dimension) if x0 is None else x0
        acc = 0
        current_likelihood = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))
        
        for t in range(N + burnin_index):
            # Propose a new point
            proposal = np.random.multivariate_normal(current, self.scale_factor * self.C[-1])
            proposal_likelihood = self.posterior(proposal)
            acceptance_probability = min(1, proposal_likelihood / current_likelihood)
            if np.random.rand() < acceptance_probability: # Accept the proposal with probability "acceptance_probability"
                current = proposal  
                current_likelihood = proposal_likelihood
                acc += 1
            if t >= burnin_index:
                self.samples[t-burnin_index, :] = current
                if (t >= burnin_index + 1) and (t - burnin_index) % self.update_step == 0:
                    # Update covariance matrix with the recursive formula
                    old_mean = self.mean
                    self.mean = old_mean + (current - old_mean) / t
                    diff = np.outer(current - self.mean, current - self.mean)
                    self.C.append((t - 1) / t * self.C[-1] + (self.scale_factor / t) * (diff + self.epsilon * np.eye(self.dimension)))
        
        self.acc_rate = acc / N

class DRAM(Sampling):
    """
    Delayed Rejection Adaptive Metropolis (DRAM) algorithm for sampling from a target distribution.
    
    Attributes:
    - scale_factor (float): Scaling factor for the proposal covariance matrix.
    - epsilon (float): Small constant for numerical stability.
    - burnin (float): Fraction of the total number of samples to discard as burn-in.
    - update_step (int): Number of samples between covariance matrix updates.
    - num_stages (int): Number of proposal stages for delayed rejection.
    - gammas (list): List of scaling factors for each proposal stage.
    - C (list): List of covariance matrices.
    - samples (np.ndarray): Generated samples.
    - mean (np.ndarray): Mean of the samples.
    - acc_rate (float): Acceptance rate of the samples.

    Example usage:
    ```
    distribution = Sampling(posterior=my_posterior, dimension=2)
    sampler = DRAM(distribution, scale_factor=0.1, gammas=[1.0, 0.5], epsilon=1e-8, burnin=0.2, update_step=1, num_stages=2)
    samples = sampler.sample(N=10000)
    ```
    """

    def __init__(self, distribution, scale_factor=None, gammas=None, epsilon=1e-8, burnin = 0.0, update_step = 1, num_stages=2):
        """
        DRAM initialization.
        
        Parameters:
        - posterior: Posterior distribution function.
        - dimension: Number of dimensions for the parameter space.
        - scaling_factor: Scaling factor for the proposal covariance matrix.
        - gammas: List of scaling factors for each proposal stage. Default: [1.0, 0.5] (for 2 stages).
        - epsilon: Regularization term to avoid singular matrices.
        - n0: Number of initial iterations before adaptation starts.
        - num_stages: Number of proposal stages for delayed rejection.
        """

        super().__init__(distribution.posterior, distribution.dimension)
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.epsilon = epsilon  # Small constant for numerical stability
        self.burnin = burnin  # Non-adaptive period
        self.num_stages = num_stages  # Number of delayed rejection stages
        self.update_step = update_step  # Update covariance matrix every n samples
        self.gammas = [1.0] + [0.5**i for i in range(1, num_stages)] if gammas is None else gammas
        self.C = np.eye(self.dimension)  # Initial covariance matrix
        self.samples = None  # To store past samples
        self.mean = np.zeros(self.dimension)  # Mean of the samples

    def acceptance_probability(self, lh):
        """
        Calculate the acceptance probability for a given likelihood history.
        
        Parameters:
        - lh: Likelihood history for the current stage.

        Returns:
        - alpha: Acceptance probability for the likelihood history.
        """

        alpha = lh[-1] / lh[0]

        numerator = 1.0
        denominator = 1.0

        for i in range(len(lh) - 2, 0, -1):
            numerator *= (1 - self.acceptance_probability(lh[i:][::-1]))
            denominator *= (1 - self.acceptance_probability(lh[:len(lh) - i]))
            
        alpha *= numerator / denominator
        
        return min(1, alpha)

    def sample(self, x0, N=10000):
        """
        Run the DRAM algorithm with multiple proposal stages.

        Parameters:
        - x0: Initial point for the chain.
        - N: Number of samples to generate.

        Returns:
        - None
        """
    
        current = np.zeros(self.dimension) if x0 is None else x0
        acc = 0
        current_likelihood = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))

        for n in range(N + burnin_index):
            likelihood_history = [current_likelihood]
            for stage in range(1, self.num_stages + 1):
                # Propose a new point for the current stage
                proposal = np.random.multivariate_normal(current, self.scale_factor * self.gammas[stage - 1] * self.C)
                proposal_likelihood = self.posterior(proposal)
                likelihood_history.append(proposal_likelihood)

                # Calculate acceptance probability for the current stage
                alpha = self.acceptance_probability(lh=likelihood_history)

                if np.random.rand() < alpha:
                    current = proposal  # Accept the proposal
                    acc += 1
                    break 

            if n >= burnin_index:
                self.samples[n-burnin_index, :] = current
                if (n >= burnin_index + 1) and (n - burnin_index) % self.update_step == 0:
                    # Update covariance matrix with the recursive formula
                    old_mean = self.mean
                    self.mean = old_mean + (current - old_mean) / n
                    diff = np.outer(current - self.mean, current - self.mean)
                    self.C.append((n - 1) / n * self.C[-1] + (self.scale_factor / n) * (diff + self.epsilon * np.eye(self.dimension)))
        
        self.acc_rate = acc / N

class DREAM(Sampling):
    pass