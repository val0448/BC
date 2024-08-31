import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable

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
        If the dimension of the samples is 2, a 2D histogram and a scatter plot of the samples are plotted side by side.
        If the dimension of the samples is greater than 2, scatter plots of all possible pairs of dimensions are plotted.

        Note: The number of plots in the figure depends on the dimensionality of the samples.

        """
        if self.dimension == 1:
            plt.figure()
            plt.hist(samples, bins=200)  # 1D histogram
            plt.xlabel('u')
            plt.ylabel('Density')
            plt.show()
        elif self.dimension == 2:
            fig, ax = plt.subplots(1, 2)
            ax[0].hist2d(samples[:, 0], samples[:, 1], bins=20)  # 2D histogram
            ax[1].plot(samples[:, 0], samples[:, 1], '.')  # scatter plot
            for i in range(2):
                ax[i].set_xlabel('$u_1$')
                ax[i].set_ylabel('$u_2$')
                ax[i].set_aspect('equal')
            plt.show()
        else:
            num_pairs = self.dimension * (self.dimension - 1) // 2
            fig, ax = plt.subplots(num_pairs, 1, figsize=(10, 10 * num_pairs))
            pair_index = 0
            for i in range(self.dimension):
                for j in range(i + 1, self.dimension):
                    ax[pair_index].plot(samples[:, i], samples[:, j], '.')  # scatter plot
                    ax[pair_index].set_xlabel(f'$u_{i+1}$')
                    ax[pair_index].set_ylabel(f'$u_{j+1}$')
                    ax[pair_index].set_aspect('equal')
                    pair_index += 1
            plt.tight_layout()
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
              chains: int = 10):
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

        burnin_index = int(burnin * N)
        samples = np.zeros((N, chains, self.dimension))
        current = initial
        log_posterior = np.array([self.posterior(x) for x in current])
        acceptance_counts = np.zeros(chains)

        gamma = 2.38 / np.sqrt(2 * delta * self.dimension)
        eps = 1e-6

        def differential_evolution_proposal(i, CR, current):
            pairs = np.random.choice(chains, size=(delta, 2), replace=False)
            diff = np.sum([current[p[0]] - current[p[1]] for p in pairs], axis=0)
            z = current[i] + gamma * diff + np.random.normal(0, eps, size=self.dimension)
            z = np.where(np.random.rand(self.dimension) < CR, z, current[i])
            return z

        def outlier_detection_step(log_posterior, current):
            if outlier_detection:
                log_posterior_mean = np.mean(log_posterior)
                log_posterior_std = np.std(log_posterior)
                outliers = np.abs(log_posterior - log_posterior_mean) > 2 * log_posterior_std
                for i in np.where(outliers)[0]:
                    current[i] = current[np.argmin(log_posterior)]  # Reset to the best current chain
            return current

        for gen in range(N):
            for i in range(chains):
                z = differential_evolution_proposal(i, CR, current)
                log_posterior_z = self.posterior(z)

                # Check for numerical issues
                if np.isnan(log_posterior_z) or np.isinf(log_posterior_z):
                    print(f"Warning: log_posterior_z is {log_posterior_z} at generation {gen}, chain {i}")
                    continue

                alpha = min(1, np.exp(log_posterior_z - log_posterior[i]))

                if np.isnan(alpha) or np.isinf(alpha):
                    print(f"Warning: alpha is {alpha} at generation {gen}, chain {i}")
                    alpha = 0

                if np.random.rand() < alpha:
                    current[i] = z
                    log_posterior[i] = log_posterior_z
                    acceptance_counts[i] += 1

            current = outlier_detection_step(log_posterior, current)

            if adapt_CR and gen < burnin_index:
                CR = self.adapt_crossover_probability(gen, acceptance_counts / (gen + 1))

            samples[gen] = current

        samples = samples[burnin_index:]  # Remove burn-in samples
        acceptance_rates = acceptance_counts / N
        return samples, acceptance_rates

    def adapt_crossover_probability(self, gen, acceptance_rates):
        # linearly decrease CR
        return max(0.1, 1 - gen / 10000)