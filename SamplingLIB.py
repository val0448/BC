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
           N: int = 1000, 
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

        Returns:
        - samples (np.ndarray): N samples

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
           N: int = 1000, 
           initial: np.ndarray = None, 
           proposal_distribution: Callable = None,
           proposal_cov: np.ndarray = None, 
           burnin: float = 0, 
           acc_rate: bool = False):
        """
        Adaptive Metropolis algorithm.

        Parameters:
        - N (int): number of samples
        - initial (np.ndarray): initial sample
        - proposal_distribution (Callable): function to draw samples from
        - burnin (float): length of the burnin period on a scale 0 to 1  

        Returns:
        - samples (np.ndarray): N samples

        Example usage:
        ```
        sampler = SamplingLIB()
        samples = sampler.AM(N=1000, initial=np.zeros(2), proposal_distribution=np.random.normal, burnin=0.2)
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
        sd = (2.4**2) / self.dimension

        if proposal_cov is None:
            proposal_cov = sd * np.eye(self.dimension)

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
                proposal_cov = (sd * np.cov(samples[:i-burnin_index+1].T)) + (sd * 0.01 * np.eye(self.dimension))

        if acc_rate:
            return samples, acc / N
        else:
            return samples