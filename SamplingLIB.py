import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable

class Sampling:
    def __init__(self, posterior: Callable, parametr_dimension: int):
        """
        Constructor for Sampling class.
        Creates an instance for a given posterior on which you can then apply sampling methods.

        Args:
            self
            posterior (function): posterior function for a given problem,
            parametr_dimension (int): dimension of the parametr,
        """
    
        self.posterior = posterior
        self.dimension = parametr_dimension

    def visualize(self, samples: np.ndarray):
        if self.dimension == 1:
            plt.figure()
            plt.hist(samples, bins=200)  # 2d histogram
            plt.xlabel('u')
            plt.ylabel('Density')
            plt.show()
        elif self.dimension == 2:
            fig, ax = plt.subplots(1, 2)
            ax[0].hist2d(samples[:, 0], samples[:, 1], bins=20)  # 2d histogram
            ax[1].plot(samples[:, 0], samples[:, 1], '.')  # dots
            for i in range(2):
                ax[i].set_xlabel('$u_1$')
                ax[i].set_ylabel('$u_2$')
                ax[i].set_aspect('equal')
            plt.show()

    def MH(self, N: int = 1000, initial: np.ndarray = None, proposal_distribution: Callable = None, burnin: float = 0):
        """
        Random walk Metropolis-Hastings algorithm.

        Args:
            self
            N (int): number of samples
            initial (np.ndarray): initial sample
            proposal_distribution (Callable): function to draw samples from
            proposal_sd (float): proposal standard deviation
            burnin (float): length of the burnin period on a scale 0 to 1  

        Returns:
            samples (np.ndarray): N samples
        """

        if initial is None:
            initial = np.zeros(self.dimension)

        if proposal_distribution is None:
            def proposal_distribution(mu): return np.random.normal(mu, 1)

        burnin_index = int(burnin * N)
        samples = np.zeros((N - burnin_index, self.dimension))
        current = initial
        current_likelihood = self.posterior(current)

        for i in range(N):
            proposal = proposal_distribution(current)
            proposal_likelihood = self.posterior(proposal)
            acceptance_probability = min(1, proposal_likelihood / current_likelihood)
            if np.random.rand() < acceptance_probability:
                current = proposal
                current_likelihood = proposal_likelihood
            if i >= burnin_index:
                samples[i-burnin_index, :] = current

        return samples