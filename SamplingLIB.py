import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, chi2
from scipy.special import logsumexp
from typing import Callable
import matplotlib.animation as animation
from scipy.stats import gaussian_kde, entropy
from IPython.display import HTML
import arviz as az
import pandas as pd
import seaborn as sns
import emcee
import itertools


class Sampling:
    """
    A class for applying sampling methods on a given posterior function.

    Attributes:
    - dimension (int): The dimension of the parameter.
    - posterior (Callable): The posterior function for a given problem.
    - likelihood (Callable): The likelihood function for a given problem.
    - forward_model (Callable): The forward model function for a given problem.
    - observed_data (np.ndarray): The observed data for a given problem.
    - noise_cov (float): The standard deviation of the noise in the data.
    - prior (Callable): The prior function for a given problem.
    - prior_means (np.ndarray): The means of the prior distribution.
    - prior_covs (np.ndarray): The standard deviations of the prior distribution.
    - log (bool): If True, the function returns the log-probability density.

    Methods:
    - __init__(self, dimension=2, posterior=None, likelihood=None, forward_model=None, observed_data=None, noise_cov=None, 
               prior=None, prior_means=None, prior_covs=None, log=False): Constructor for Sampling class.
    """

    def __init__(self, dimension=2, posterior=None, likelihood=None, forward_model=None, observed_data=None, noise_cov=None,
                 prior=None, prior_means=None, prior_covs=None, weights=None, true_mean=None, kde=None, samples=None, log=False):
        """
        Constructor for the Sampling class.

        Parameters:
        - dimension (int): The dimension of the parameter.
        - posterior (Callable): The posterior function for a given problem.
        - prior (Callable): The prior function for a given problem.
        - prior_means (np.ndarray): The means of the prior distribution.
        - prior_covs (np.ndarray): The standard deviations of the prior distribution.
        - likelihood (Callable): The likelihood function for a given problem.
        - forward_model (Callable): The forward model function for a given problem.
        - observed_data (np.ndarray): The observed data for a given problem.
        - noise_cov (float): The standard deviation of the noise in the data.
        - log (bool): If True, the function returns the log-probability density.

        Returns:
        - None

        Raises:
        - ValueError: If required parameters for likelihood or prior are not provided.
        """

        self.dimension = dimension
        self.log = log
        self.true_mean = true_mean

        self.likelihood = likelihood
        self.forward_model = forward_model
        self.observed_data = observed_data
        self.noise_cov = noise_cov
        self.prior = prior
        self.prior_means = prior_means
        self.prior_covs = prior_covs
        self.weights = weights

        if posterior is None:
            # Likelihood
            if likelihood is None:
                if forward_model is None or observed_data is None or noise_cov is None:
                    raise ValueError("`forward_model`, `observed_data`, and `noise_cov` are required to set up the likelihood.")
                else:
                    self.forward_model = forward_model
                    self.observed_data = np.asarray(observed_data) if isinstance(observed_data, list) else observed_data
                    self.noise_cov = noise_cov
                    self.likelihood = self._likelihood_func
            else:
                self.likelihood = likelihood

            # Prior
            if prior is None:
                if prior_means is None or prior_covs is None:
                    raise ValueError("`prior_means` and `prior_covs` are required to set up the prior.")
                else:
                    self.prior_means = np.asarray(prior_means)
                    self.prior_covs = np.asarray([np.diag(cov ** 2) if cov.ndim == 1 else cov for cov in np.asarray(prior_covs)])
                    wghts = np.ones(len(self.prior_means)) / len(self.prior_means) if weights is None else weights / np.sum(weights)
                    self.weights = np.log(wghts) if self.log else wghts
                    self.prior = self._gaussian_mixture_pdf
            else:
                self.prior = prior

            # Posterior
            self.posterior = lambda x: self.likelihood(x) + self.prior(x) if self.log else self.likelihood(x) * self.prior(x)
        else:
            self.posterior = posterior

        self.kde = kde  # Placeholder for KDE approximation
        self.samples = samples  # Placeholder for generated samples

    def _likelihood_func(self, point):
        """
        Computes the likelihood of the forward model given the data.

        Parameters:
        - point (np.ndarray): The point at which to evaluate the likelihood.

        Returns:
        - float: The likelihood of the forward model given the data.
        """

        y_pred = self.forward_model(point)
        likelihood = multivariate_normal.logpdf(self.observed_data - y_pred, cov=self.noise_cov)

        return likelihood if self.log else np.exp(likelihood)

    def _gaussian_mixture_pdf(self, point):
        """
        Computes the combined Gaussian probability density function for multiple means and standard deviations.

        Parameters:
        - point (np.ndarray): The point at which to evaluate the PDF.

        Returns:
        - float: The combined Gaussian PDF at the given point.
        """
        
        # Compute log-probabilities for each Gaussian component
        log_probs = np.array([multivariate_normal.logpdf(point, mean=mean, cov=cov) for mean, cov in zip(self.prior_means, self.prior_covs)])

        return logsumexp(log_probs + self.weights) if self.log else np.sum(np.dot(self.weights, np.exp(log_probs)))

    def visualize(self, visuals: list = [], grid: tuple = None, ranges: list = [], max_points: int = 50, samples=None, vertical=False, cov=None):
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
        samples = self.samples if samples is None else samples
        visuals_map = {"prior": [self.prior, self.visualize_posterior],
                       "posterior": [self.posterior, self.visualize_posterior],
                       "likelihood": [self.likelihood, self.visualize_posterior],
                       "forward_model": [self.forward_model, self.visualize_posterior],
                       "kde": [self.kde, self.visualize_kde],
                       "samples_hist": [samples, self.visualize_samples_hist],
                       "samples_scatter": [samples, self.visualize_samples_scatter],
                       "samples_trace": [samples, self.visualize_samples_trace],
                       "samples_acf": [samples, self.visualize_samples_acf],
                       "covariance": [cov, self.visualize_covariance_2d]}

        visuals = ["posterior"] if visuals == [] else visuals
        fig, axes = plt.subplots(len(visuals), 1, figsize=(15, 5*len(visuals))) if vertical else plt.subplots(1, len(visuals), figsize=(5*len(visuals), 5))
        axes = [axes] if len(visuals) == 1 else axes
        axis_iter = iter(axes)  # Create an iterator over axes

        for visual in visuals:
            if visual in visuals_map and visuals_map[visual][0] is not None:
                obj, plot_func = visuals_map[visual]
                if visual not in ["samples_scatter", "samples_trace", "samples_acf"]:
                    if (grid is None and self.dimension <= 2):
                        grid = self.create_grid(visual=obj, ranges=ranges, max_points=max_points)                    
                    plot_func(visual=obj, grid=grid, title=visual, ax=next(axis_iter), show=False)
                else:
                    plot_func(visual=obj, title=visual, ax=next(axis_iter), show=False)
            else:
                raise ValueError(f"Unsupported visual: {visual}") if visual not in visuals_map else ValueError(f"Visual {visual} is None.")
        plt.tight_layout()
        plt.show()

    def visualize_samples_hist(self, visual: np.ndarray, grid: tuple = None, title="Sample histogram", ax = None, show = True):
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

        # Check if samples come from multiple chains
        if visual.ndim == 3:
            visual = visual.reshape(-1, self.dimension)
        elif visual.ndim > 3:
            raise ValueError("Samples array must be 2D or 3D")

        # Create a grid if not provided
        grid = self.create_grid(visual=visual) if (grid is None and self.dimension <= 2) else grid

        # Create an axis if not provided
        fig, ax = (plt.subplots(figsize=(6, 6)) if ax is None else (None, ax))
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        if self.dimension == 1:
            # 1D histogram
            ax.hist(visual[:, 0], bins=grid[2][0], alpha=0.5)
            ax.set_xlabel('u')
            ax.set_ylabel('Density')

        elif self.dimension == 2:
            # 2D histogram
            ax.hist2d(visual[:, 0], visual[:, 1], bins=grid[2][:2], range=[grid[1][0], grid[1][1]], cmap="viridis")
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('$u_2$')

        elif self.dimension > 2 and ax == None and show:  # N-dimensional histogram
            # N-dimensional histogram: Use pair plots
            df = pd.DataFrame(visual, columns=[f'$u_{i+1}$' for i in range(self.dimension)])
            g = sns.pairplot(df, diag_kind='hist', kind='hist', corner=False, plot_kws={'bins': 50, 'cmap': "viridis"}, diag_kws={'bins': 25})
            for ax in g.axes.flatten():
                if ax is not None:
                    ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)  # Enable grid for each subplot
        if show:
            plt.show()

    def visualize_samples_scatter(self, visual: np.ndarray, title="Sample scatter", ax = None, show = True):
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
        if visual.ndim == 3:
            N_chains, N_gen, dimension = visual.shape
        elif visual.ndim == 2:
            N_gen, dimension = visual.shape
            N_chains = 1  # Single chain
        else:
            raise ValueError("Samples array must be 2D or 3D")

        # Create an axis if not provided
        fig, ax = plt.subplots(figsize=(6, 6) if dimension == 1 else (6, 6)) if ax is None else (None, ax)
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)
        colors = plt.cm.rainbow(np.linspace(0, 1, N_chains))

        if dimension == 1: # 1D scatter plot
            if N_chains > 1:
                for chain, color in zip(range(N_chains), colors):
                    ax.plot(range(N_gen), visual[chain, :, 0], '.', color=color, label=f'Chain {chain+1}', alpha=0.5)
                    ax.legend(loc='upper right')
            else:
                ax.plot(range(N_gen), visual[:, 0], '.', alpha=0.5)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('$u_1$')    

        elif dimension == 2: # 2D scatter plot
            if N_chains > 1:
                for chain, color in zip(range(N_chains), colors):
                    ax.plot(visual[chain, :, 0], visual[chain, :, 1], '.', color=color, label=f'Chain {chain+1}', alpha=0.5)
                    ax.legend(loc='upper right')
            else:
                ax.plot(visual[:, 0], visual[:, 1], '.', alpha=0.5)
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('$u_2$')

        elif dimension > 2 and ax == None and show:  # N-dimensional scatter plot 
            if visual.ndim == 3:
                visual = visual.reshape(-1, self.dimension)
                chain_labels = np.repeat(np.arange(visual.shape[0] // N_gen), N_gen)
            else:
                chain_labels = np.zeros(visual.shape[0])

            df = pd.DataFrame(visual, columns=[f'$u_{i+1}$' for i in range(self.dimension)])
            df['Chain'] = chain_labels
            g = sns.pairplot(df, hue='Chain', palette='rainbow', diag_kind='hist', corner=True)
            for ax in g.axes.flatten():
                if ax is not None:
                    ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)  # Enable grid for each subplot
        if show:
            plt.show()

    def visualize_posterior(self, visual, grid=None, title="Posterior", ax=None, ranges=[], max_points=100 , show=True):
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
        if self.dimension > 2:
            raise ValueError("Posterior visualization is only supported for 1D or 2D cases.")

        grid = self.create_grid(visual=visual, ranges=ranges, max_points=max_points) if (grid is None and self.dimension <= 2) else grid # Create a grid if not provided
        fig, ax = plt.subplots(figsize=(6, 6) if dimension == 1 else (6, 6)) if ax is None else (None, ax) # Create an axis if not provided
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        # Evaluate posterior on the grid
        posterior_values = np.exp(np.array([visual(point) for point in grid[0]])) if self.log else np.array([visual(point) for point in grid[0]])

        dimension = grid[0].shape[1]
        if dimension == 1:
            ax.plot(grid[0][:, 0], posterior_values, label='Posterior')
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('Density')

        elif dimension == 2:
            posterior_reshaped = posterior_values.reshape((grid[2][0], grid[2][1]))
            ax.imshow(posterior_reshaped.T, extent=[grid[0][:, 0].min(), grid[0][:, 0].max(),
                                                    grid[0][:, 1].min(), grid[0][:, 1].max()], origin='lower', aspect='auto', cmap="viridis")
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('$u_2$')
            
        if show:
            plt.show()

    def visualize_samples_trace(self, visual, title="Sample trace", ax=None, chainwise=-1, show=True):
        """
        Visualize sample traces.
        
        Parameters:
        visual (np.ndarray): Samples array of shape (num_samples, dim) or (num_samples, num_chains, dim).
        chainwise (bool): If True, merge chains; otherwise plot each chain separately.
        ax (matplotlib.axes.Axes): Axis to plot on (if None, a new figure is created).
        title (str): Title for the plot.
        show (bool): Whether to call plt.show() at the end.
        
        Returns:
        None
        """
        # Convert samples to shape (num_chains, num_samples, dim)
        if visual.ndim == 2:
            visual = visual[np.newaxis, :, :]
        if visual.ndim != 3:
            raise ValueError("Samples must be a 2D or 3D array.")
        
        num_chains, num_samples, dim = visual.shape
        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)
        visual = visual.reshape(-1, dim) if (chainwise == -1) else visual[chainwise]
        ax.set_title(title) if (chainwise == -1) else ax.set_title(f"Chain {chainwise+1} - {title}")
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        for j in range(dim):
            ax.plot(visual[:, j], label=f"Dimension {j+1}")
        
        # Add vertical lines at every point = chain * num_samples
        if chainwise == -1:
            for chain in range(0, num_chains+1):
                ax.axvline(chain * num_samples, color='black', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        
        if show:
            plt.show()

    def visualize_samples_acf(self, visual, title="Autocorrelation", ax=None, IAT_avg=None, IAT_max=None, acfs=None, show=True):
        """
        Visualize the autocorrelation functions (ACF) for each dimension.
        
        Parameters:
        visual (np.ndarray): Samples array of shape (num_samples, dim) or (num_samples, num_chains, dim).
        chainwise (bool): If True, compute ACF for merged chains; otherwise, for first chain.
        max_lag (int): Maximum lag for ACF computation.
        ax (matplotlib.axes.Axes): Axis to plot on (if None, a new figure is created).
        title (str): Title for the plot.
        show (bool): Whether to display the plot.
        
        Returns:
        None
        """

        fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        if IAT_avg is None or acfs is None or IAT_max is None:
            IAT_avg, IAT_max, acfs = compute_autocorrelation(samples=visual)
    
        # Plot ACF as a line plot with dots at each point
        ax.plot(np.arange(1, len(acfs) + 1), acfs, color='blue', linewidth=1, label='ACF')
        ax.scatter(np.arange(1, len(acfs) + 1), acfs, color='blue', s=10)  # Add dots at each point
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(IAT_avg, color='black', linestyle=':', linewidth=2, label=f"IAT_avg = {IAT_avg:.2f}")
        ax.axvline(IAT_max, color='black', linestyle=':', linewidth=2, label=f"IAT_max = {IAT_max:.2f}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(loc="upper right")
        
        if show:
            plt.show()

    def visualize_kde(self, visual, grid: tuple, title="KDE approximation", ax=None, show = True):
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
        # Create an axis if not provided
        fig, ax = plt.subplots(figsize=(6, 6) if self.dimension == 1 else (6, 6)) if ax is None else (None, ax)
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        if self.dimension == 1:
            # 1D Case: Plot KDE approximation as a line plot
            ax.plot(grid[0][:, 0], visual, label='KDE Approximation')
            ax.set_xlabel('$u_1$')
            ax.set_ylabel('Density')

        elif self.dimension == 2:
            # 2D Case: Use imshow for KDE approximation heatmap
            kde_approximation_reshaped = visual.reshape((grid[2][0], grid[2][1]))
            ax.imshow(kde_approximation_reshaped.T, extent=[grid[0][:, 0].min(), grid[0][:, 0].max(),
                                                            grid[0][:, 1].min(), grid[0][:, 1].max()], origin='lower', aspect='auto', cmap="viridis")

        else:
            # N-dimensional Case: Use pair plots
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i == j:
                        # Diagonal: 1D KDE approximation
                        ax[i, j].plot(grid[0][:, i], visual, label='KDE', alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                    else:
                        # Off-diagonal: 2D scatter plot
                        ax[i, j].scatter(grid[0][:, i], grid[0][:, j], s=5, alpha=0.5)
                        ax[i, j].set_xlabel(f'$u_{i+1}$')
                        ax[i, j].set_ylabel(f'$u_{j+1}$')
        plt.tight_layout()
        if show:
            plt.show()

    def visualize_covariance_2d(self, visual, grid=None, title="Proposal distribution", ax=None, ranges=[], center=None, confidence_levels=[], show=True):
        """
        Visualizes the covariance contours over the posterior distribution.

        Parameters:
        - visual (Callable): The posterior function to visualize.
        - grid (tuple): Grid for posterior visualization (default is None).
        - title (str): Title for the plot (default is "Covariance").
        - ax (matplotlib.axes.Axes): Axis to plot on (default is None).
        - show (bool): If True, displays the plot (default is True).

        Returns:
        - None
        """
        if self.dimension != 2:
            raise ValueError("Covariance visualization is only supported for 2D cases.")
        elif visual is None:
            raise ValueError("Covariance must be provided for covariance visualization.")

        # Define the mean and covariance matrix
        mean = self.mean if (center is None and hasattr(self, "mean")) else (center if center is not None else np.zeros(self.dimension))
        cov = visual

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Calculate ellipse parameters for largest confidence interval (95%)
        chi2_95 = chi2.ppf(0.95, df=2)
        width_95 = 2 * np.sqrt(chi2_95 * eigenvalues[0])
        height_95 = 2 * np.sqrt(chi2_95 * eigenvalues[1])
        
        # Calculate ellipse vertices coordinates
        ellipse = plt.matplotlib.patches.Ellipse(mean, width_95, height_95, angle=angle)

        # Get the extents of the ellipse
        verts = ellipse.get_verts()
        x_min_ellipse, x_max_ellipse = verts[:, 0].min(), verts[:, 0].max()
        y_min_ellipse, y_max_ellipse = verts[:, 1].min(), verts[:, 1].max()
        
        # Add margin to the extents
        x_span = x_max_ellipse - x_min_ellipse
        y_span = y_max_ellipse - y_min_ellipse
        x_min = x_min_ellipse - 0.1 * x_span
        x_max = x_max_ellipse + 0.1 * x_span
        y_min = y_min_ellipse - 0.1 * y_span
        y_max = y_max_ellipse + 0.1 * y_span
        
        # Combine with user-specified ranges if any
        if ranges:
            x_min = min(x_min, ranges[0][0]) if ranges[0] else x_min
            x_max = max(x_max, ranges[0][1]) if ranges[0] else x_max
            y_min = min(y_min, ranges[1][0]) if len(ranges) > 1 else y_min
            y_max = max(y_max, ranges[1][1]) if len(ranges) > 1 else y_max
        
        # Create new ranges array
        new_ranges = [(x_min, x_max), (y_min, y_max)]
        
        # Create grid with adjusted ranges
        grid = self.create_grid(ranges=new_ranges) if grid is None else grid
        
        # Rest of the visualization code (same as before)
        fig, ax = plt.subplots(figsize=(5, 5)) if ax is None else (None, ax)
        self.visualize_posterior(visual=self.posterior, grid=grid, ax=ax, show=False)
        ax.set_title(title)
        ax.grid(True, color='gray', linestyle='solid', linewidth=0.8, alpha=0.8)

        # # Reshape the grid for contour plotting
        # x = grid[0][:, 0].reshape(grid[2][0], grid[2][1])
        # y = grid[0][:, 1].reshape(grid[2][0], grid[2][1])
        # pos = np.dstack((x, y))

        # # Compute the probability density function
        # rv = multivariate_normal(mean, visual)
        # z = rv.pdf(pos)

        # # Plot the contours
        # ax.contour(x, y, z, levels=3, colors='red')

        # Chi-squared quantiles for 2 degrees of freedom (2D)
        confidence_levels = [0.35, 0.65, 0.85, 0.95] if confidence_levels == [] else confidence_levels
        chi2_quantiles = [chi2.ppf(level, df=2) for level in confidence_levels]

        # Plot ellipses for confidence levels
        for quantile in chi2_quantiles:
            # Scale eigenvalues by sqrt(quantile) to get radii
            width = 2 * np.sqrt(quantile * eigenvalues[0])
            height = 2 * np.sqrt(quantile * eigenvalues[1])
            ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, edgecolor='red', facecolor='none', linestyle='solid', linewidth=1,)
            ax.add_patch(ellipse)

        ax.scatter(mean[0], mean[1], c='red', s=50, marker='x')
        ax.set_xlabel("$u_1$")
        ax.set_ylabel("$u_2$")

        if show:
            plt.show()

    def create_grid(self, visual = None, max_points: int = 100, min_points: int = 10, margin: float = 0.1, ranges: list = []):
        """
        Creates a grid for comparing samples or posterior, with adaptive num_points for each dimension
        based on relative ranges and scaling for total number of dimensions.
        
        Parameters:
        - visual (np.ndarray or Callable): Samples or posterior function to visualize.
        - max_points (int): Maximum number of grid points for the longest range (default is 100).
        - min_points (int): Minimum number of grid points for the shortest range (default is 10).
        - margin (float): Margin to extend the grid beyond the sample bounds (default is 10%).
        - ranges (list): List of min and max bounds for each dimension (default is None)

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

        # Case 1: If samples are provided, use them to calculate the grid bounds
        if ranges != []:
            if len(ranges) != self.dimension:
                raise ValueError("Ranges must be provided for each dimension.")
            else:
                dim_ranges = [r[1] - r[0] for r in ranges]

        elif isinstance(visual, np.ndarray):
            # Check if samples come from multiple chains
            if visual.ndim == 3:
                visual = visual.reshape(-1, self.dimension)
            elif visual.ndim > 3:
                raise ValueError("Samples array must be 2D or 3D")

            for dim in range(self.dimension):
                dim_min = np.min(visual[:, dim])
                dim_max = np.max(visual[:, dim])
                dim_range = dim_max - dim_min
                dim_ranges.append(dim_range)

                # Add margin to extend the bounds
                range_min = dim_min - (margin * dim_range)
                range_max = dim_max + (margin * dim_range)
                ranges.append((range_min, range_max))
        
        # Case 2: If a posterior function is provided, use default heuristic ranges
        elif (visual is None or callable(visual)):
            for dim in range(self.dimension):
                ranges.append((-5, 5))  # Use default range for posterior
                dim_ranges.append(10)  # Example range for posterior
        else:
            raise ValueError("Either samples, distribution or ranges must be provided.")

        # Find the longest range and normalize other ranges relative to it
        max_range = max(dim_ranges)

        # Calculate num_points for each dimension, relative to the max range and the number of dimensions
        dim_num_points = []
        for dim_range in dim_ranges:
            relative_scale = dim_range / max_range
            num_points = int(relative_scale * (max_points - min_points) + min_points)
            #num_points = int(num_points * (1 / self.dimension))  # Scale down based on number of dimensions
            num_points = max(min_points, min(max_points, num_points))  # Ensure within bounds
            dim_num_points.append(num_points)

        # Create a grid using the determined ranges and num_points per dimension
        grid_ranges = [np.linspace(r[0], r[1], n) for r, n in zip(ranges, dim_num_points)]
        mesh = np.meshgrid(*grid_ranges, indexing='ij')

        # Stack the meshgrid into a list of points in shape (num_points^dimension, dimension)
        grid = np.vstack([m.ravel() for m in mesh]).T

        return grid, ranges, dim_num_points

    def animate_chain_movement(self, samples: np.ndarray, subsample_rate: int = 1, time: int = 60, html: bool = True):
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

        if samples is None:
            if hasattr(self, 'samples') and self.samples is not None:
                samples = self.samples
            else:
                raise ValueError("No samples provided for animation.")

        # Ensure the correct shape and dimensionality
        if samples.ndim != 2:
            raise ValueError("Expected samples array of shape (N_gen, dimension)")

        N_gen, dimension = samples.shape

        if dimension != 2:
            raise ValueError("Currently only 2D sample space is supported for animation.")

        # Subsample the data
        samples = samples[::subsample_rate]

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
        if html:
            return HTML(anim.to_jshtml())  # Return the HTML representation for display
        else:
            plt.show()

    def sampling_quality(self, samples: np.ndarray, ranges=[], visualise: bool = False):
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

        # Check if samples come from multiple chains
        if samples.ndim == 3:
            samples = samples.reshape(-1, self.dimension)
        elif samples.ndim > 3:
            raise ValueError("Samples array must be 2D or 3D")

        # Create a grid using the samples (adjustable for N dimensions)
        grid = self.create_grid(visual=samples, ranges=ranges, max_points=100)

        # Apply KDE to the samples
        kde = gaussian_kde(samples.T)  # Transpose to match expected shape

        # Evaluate the KDE on the grid
        kde_approximation = kde(grid[0].T)  # Evaluate KDE on the grid points
        kde_approximation_flat = kde_approximation.flatten()

        # Evaluate the target posterior on the grid
        target_pdf = np.array([np.exp(self.posterior(u)) if self.log else self.posterior(u) for u in grid[0]])
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
            self.kde = kde_approximation_flat
            self.visualize(visuals=["kde", "posterior"], grid=grid)

        # Return KL Divergence value
        return kl_divergence

class MH(Sampling):
    """
    Random Walk Metropolis algorithm for sampling from a target distribution.
    Uses a Gaussian proposal distribution.

    Attributes:
    - epsilon (float): Small value to ensure positive definiteness of the covariance matrix.
    - scale_factor (float): Scaling factor for the covariance matrix.
    - mean (np.ndarray): Mean of the samples.
    - burnin (float): Fraction of the total number of samples to discard as burn-in.
    - C (list): List of covariance matrices.
    - acc_rate (float): Acceptance rate of the samples.
    - samples (np.ndarray): Generated samples.

    Example usage:
    ```
    distribution = Sampling(posterior=my_posterior, dimension=2)
    sampler = MG(distribution, initial_cov=np.eye(2), epsilon=1e-5, burnin=0.2)
    samples = sampler.sample(N=10000)
    ```
    """

    def __init__(self, distribution, initial_cov=None, scale_factor=None, burnin=0.2):
        """
        Initialize the Random Walk Metropolis Hastings algorithm.

        Parameters:
        - distribution (Sampling): Sampling object with the target distribution.
        - initial_cov (np.ndarray): Initial covariance matrix for the proposal distribution.
        - epsilon (float): Small value to ensure positive definiteness of the covariance matrix.
        - scale_factor (float): Scaling factor for the covariance matrix.
        - burnin (float): Fraction of the total number of samples to discard as burn-in.
        """

        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for MH
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.mean = np.zeros(self.dimension)
        self.burnin = burnin
        self.prop_dist = multivariate_normal # Proposal distribution
        self.C = self.scale_factor * np.eye(self.dimension) if initial_cov is None else self.scale_factor * initial_cov
        self.acc_rate = None
        self.samples = None

    def print_info(self):
        """
        Print information about the sampling method and its parameters.
        """
        print(f"Sampling Method: {self.__class__.__name__}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"Covariance Matrix: {self.C}")
        print(f"Burn-in: {self.burnin}")
        print(f"Mean of Samples: {self.mean}")
        print(f"Acceptance Rate: {(self.acc_rate * 100):.2f}%")

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_covs is None):
            initial = np.random.randn(self.dimension)
        else:
            mode_idx = np.random.choice(len(self.prior_means))
            initial = multivariate_normal.rvs(mean=self.prior_means[mode_idx], cov=self.prior_covs[mode_idx])

        return np.array(initial)
    
    def _reset(self):
        """
        Reset the sampler to its initial state.
        """

        self.mean = np.zeros(self.dimension)
        self.acc_rate = None
        self.samples = None

    def sample(self, initial=None, N=10000):
        """
        Run the Adaptive Metropolis algorithm for n_samples iterations.

        Parameters:
        - x0 (np.ndarray): Initial point for the chain.
        - N (int): Number of samples to generate.

        Returns:
        - None
        """

        self._reset()  # Reset the sampler state
        current = self._initial_sample() if initial is None else initial
        acc = 0
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))
        
        for t in range(N + burnin_index):
            # Propose a new point
            proposal = self.prop_dist.rvs(mean=current, cov=self.C)
            proposal_posterior = self.posterior(proposal)
            alpha = min(0, proposal_posterior - current_posterior) if self.log else min(1, proposal_posterior / current_posterior)
            u = np.log(np.random.rand()) if self.log else np.random.rand()
            # Accept the proposal with probability alpha
            if u < alpha: 
                current = proposal  
                current_posterior = proposal_posterior
                if t >= burnin_index:
                    acc += 1
            if t >= burnin_index:
                self.samples[t-burnin_index, :] = current
        
        self.acc_rate = acc / N
        self.mean = np.mean(self.samples, axis=0)
        return self.samples

class AM(Sampling):
    """
    Adaptive Metropolis algorithm for sampling from a target distribution.
    Uses a Gaussian proposal distribution.

    Attributes:

    Example usage:
    ```
    distribution = Sampling(posterior=my_posterior, dimension=2)
    sampler = AM(distribution, initial_cov=np.eye(2), epsilon=1e-5, scale_factor=None, burnin=0.2, update_step=1)
    samples = sampler.sample(N=10000)
    ```
    """

    def __init__(self, distribution, initial_cov=None, scale_factor=None, t0=0.01, burnin=0.2, eps=1e-5, update_step=1):
        """
        Initialize the Adaptive Metropolis algorithm.

        Parameters:
        """

        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for AM
        self.eps = eps
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.mean = np.zeros(self.dimension)  # Track mean of all samples (including t0)
        self.t0 = t0  # Initial fixed period for covariance
        self.burnin = burnin
        self.update_step = update_step
        self.prop_dist = multivariate_normal
        self.C0 = self.scale_factor * initial_cov if initial_cov is not None else self.scale_factor * np.eye(self.dimension)
        self.C = None
        self.acc_rate = None
        self.samples = None
        self.mean = np.zeros(self.dimension)  # Track mean of all samples
        self.outer = np.zeros((self.dimension, self.dimension))

    def print_info(self):
        """
        Print information about the sampling method and its parameters.
        """
        print(f"Sampling Method: {self.__class__.__name__}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"t0: {self.t0}")
        print(f"Initial covariance (C0): {self.C0}")
        print(f"Covariance Matrix: {self.C[-1]}")
        print(f"Epsilon: {self.eps}")
        print(f"Burn-in: {self.burnin}")
        print(f"Update Step: {self.update_step}")
        print(f"Mean of Samples: {self.mean}")
        print(f"Acceptance Rate: {(self.acc_rate * 100):.2f}%")

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_covs is None):
            initial = np.random.randn(self.dimension)
        else:
            mode_idx = np.random.choice(len(self.prior_means))
            initial = multivariate_normal.rvs(mean=self.prior_means[mode_idx], cov=self.prior_covs[mode_idx])

        return np.array(initial)
    
    def _reset(self):
        """
        Reset the sampler to its initial state.
        """
        self.mean = np.zeros(self.dimension)
        self.outer = np.zeros((self.dimension, self.dimension))
        self.C = None
        self.acc_rate = None
        self.samples = None

    def sample(self, initial=None, N=10000):
        """
        Run the Adaptive Metropolis algorithm for n_samples iterations.

        Parameters:
        - initial (np.ndarray): Initial point for the chain.
        - N (int): Number of samples to generate.

        Returns:
        - None
        """

        self._reset()  # Reset the sampler state
        current = self._initial_sample() if initial is None else initial
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N + burnin_index, self.dimension))
        self.samples[0, :] = current
        self.C = np.zeros((N + burnin_index, self.dimension, self.dimension))
        self.C[0,:,:] = self.C0
        acc = 0
        t0 = int(self.t0 * N) if self.t0 > 0 else 1

        # Reset running statistics
        self.mean = current.copy()
        self.outer[:] = 0
        
        for t in range(1, N + burnin_index):
            # Propose a new point
            proposal = self.prop_dist.rvs(mean=current, cov=self.C[t-1])
            proposal_posterior = self.posterior(proposal)
            alpha = min(0, proposal_posterior - current_posterior) if self.log else min(1, proposal_posterior / current_posterior)
            u = np.log(np.random.rand()) if self.log else np.random.rand()
            # Accept the proposal with probability alpha
            if u < alpha:
                current = proposal
                current_posterior = proposal_posterior
                if t >= burnin_index:
                    acc += 1
            self.samples[t, :] = current

            # Update running statistics
            delta = current - self.mean
            self.mean += delta / (t + 1)
            self.outer += np.outer(delta, current - self.mean)

            if t <= t0:
                self.C[t] = self.C[t-1]
            else:
                self.C[t] = ((self.scale_factor / t) * self.outer) + (self.eps * np.eye(self.dimension))
        
        self.samples = self.samples[burnin_index:, :]
        self.acc_rate = acc / N
        return self.samples

class DRAM(Sampling):
    """
    Delayed Rejection Adaptive Metropolis (DRAM) algorithm for sampling from a target distribution.
    Uses a Gaussian proposal distribution.
    
    Attributes:

    Example usage:
    ```
    distribution = Sampling(posterior=my_posterior, dimension=2)
    sampler = DRAM(distribution, scale_factor=0.1, gammas=[1.0, 0.5], epsilon=1e-8, burnin=0.2, update_step=1, num_stages=2)
    samples = sampler.sample(N=10000)
    ```
    """

    def __init__(self, distribution, initial_cov=None, scale_factor=None, burnin=0.2, t0=0.01, eps=1e-5, update_step=1, gammas=None, num_stages=3):
        """
        DRAM initialization.
        
        Parameters:
        """

        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for DRAM
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.eps = eps  # Small constant for numerical stability
        self.burnin = burnin  # Non-adaptive period
        self.t0 = t0  # Initial fixed period for covariance
        self.num_stages = num_stages  # Number of delayed rejection stages
        self.update_step = update_step  # Update covariance matrix every n samples
        self.gammas = [1.0] + [0.5**i for i in range(1, num_stages)] if gammas is None else gammas
        self.prop_dist = multivariate_normal # Proposal distribution
        self.C0 = self.scale_factor * initial_cov if initial_cov is not None else self.scale_factor * np.eye(self.dimension)
        self.C = None
        self.acc = None  # Count of accepted samples
        self.acc_rate = None  # Acceptance rate of the samples
        self.samples = None
        self.mean = np.zeros(self.dimension)  # Track mean of all samples
        self.outer = np.zeros((self.dimension, self.dimension))

    def print_info(self):
        """
        Print information about the sampling method and its parameters.
        """
        print(f"Sampling Method: {self.__class__.__name__}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"t0: {self.t0}")
        print(f"Initial covariance (C0): {self.C0}")
        print(f"Covariance Matrix: {self.C[-1]}")
        print(f"Epsilon: {self.eps}")
        print(f"Burn-in: {self.burnin}")
        print(f"Update Step: {self.update_step}")
        print(f"Number of Stages: {self.num_stages}")
        print(f"Gammas: {self.gammas}")
        print(f"Mean of Samples: {self.mean}")
        for i, acc_rate in enumerate(self.acc_rate):
            print(f"Acceptance rate for stage {i+1}: {(acc_rate * 100):.2f}%")
        print("-----------------------------------------------------------------------------------------")
        N = self.samples.shape[1] if self.samples.ndim == 3 else self.samples.shape[0]
        print(f"Acceptance rate: {(sum(self.acc[:,1])/N) * 100:.2f}%")

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_covs is None):
            initial = np.random.randn(self.dimension)
        else:
            mode_idx = np.random.choice(len(self.prior_means))
            initial = multivariate_normal.rvs(mean=self.prior_means[mode_idx], cov=self.prior_covs[mode_idx])

        return np.array(initial)

    def _acceptance_probability(self, stage_posterior, proposals, t):
        """
        Calculate the acceptance probability for a given likelihood history.
        
        Parameters:
        - lh: Likelihood history for the current stage.

        Returns:
        - alpha: Acceptance probability for the likelihood history.
        """

        stage = len(stage_posterior) - 2

        if self.log:
            alpha = (stage_posterior[-1] - stage_posterior[0]) if stage_posterior[0] > -np.inf else 0.0
            numerator_alpha = 0.0
            denominator_alpha = 0.0
        else:
            alpha = (stage_posterior[-1] / stage_posterior[0]) if stage_posterior[0] > 0 else 1.0
            numerator_alpha = 1.0
            denominator_alpha = 1.0

        for i in range(stage, 0, -1):
            if self.log:
                numerator_alpha += self.prop_dist.logpdf(proposals[i], mean=proposals[-1], cov=self.gammas[stage - i] * self.C[t-1])
                numerator_alpha += np.log1p(-np.exp(self._acceptance_probability(stage_posterior=stage_posterior[i:][::-1], proposals=proposals[i:][::-1], t=t)))

                denominator_alpha += self.prop_dist.logpdf(proposals[i], mean=proposals[0], cov=self.gammas[i-1] * self.C[t-1])
                denominator_alpha += np.log1p(-np.exp(self._acceptance_probability(stage_posterior=stage_posterior[:len(stage_posterior) - i], proposals=proposals[:len(stage_posterior) - i], t=t)))
            else:
                numerator_alpha *= self.prop_dist.pdf(proposals[i], mean=proposals[-1], cov=self.gammas[stage] * self.C[t-1])
                numerator_alpha *= (1 - self._acceptance_probability(stage_posterior=stage_posterior[i:][::-1], proposals=proposals[i:][::-1], t=t))

                denominator_alpha *= self.prop_dist.pdf(proposals[i], mean=proposals[0], cov=self.gammas[stage] * self.C[t-1])
                denominator_alpha *= (1 - self._acceptance_probability(stage_posterior=stage_posterior[:len(stage_posterior) - i], proposals=proposals[:len(stage_posterior) - i], t=t))
        
        if self.log:
            alpha += (numerator_alpha - denominator_alpha) if denominator_alpha > -np.inf else 0.0
        else:
            alpha *= (numerator_alpha / denominator_alpha) if denominator_alpha > 0 else 1.0
        
        return min(0, alpha) if self.log else min(1, alpha)
    
    def _reset(self):
        """
        Reset the sampler to its initial state.
        
        Returns:
        - None
        """
        
        self.acc = None
        self.acc_rate = None
        self.samples = None
        self.mean = np.zeros(self.dimension)
        self.outer = np.zeros((self.dimension, self.dimension))
        self.C = None

    def sample(self, initial=None, N=10000):
        """
        Run the DRAM algorithm with multiple proposal stages.

        Parameters:
        - x0: Initial point for the chain.
        - N: Number of samples to generate.

        Returns:
        - None
        """

        self._reset()  # Reset the sampler to its initial state
        current = self._initial_sample() if initial is None else initial
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N + burnin_index, self.dimension))
        self.samples[0, :] = current
        self.acc = np.zeros((self.num_stages, 2))
        self.C = np.zeros((N + burnin_index, self.dimension, self.dimension))
        self.C[0,:,:] = self.C0
        t0 = int(self.t0 * N) if self.t0 > 0 else 1

        # Reset running statistics
        self.mean = current.copy()
        self.outer[:] = 0

        for t in range(1, N + burnin_index):
            stage_posterior = [current_posterior]
            proposals = [current]
            for stage in range(self.num_stages):
                if t >= burnin_index:
                    self.acc[stage, 0] += 1

                # Propose a new point for the current stage
                proposal = self.prop_dist.rvs(mean=current, cov=self.gammas[stage] * self.C[t-1])
                proposal_posterior = self.posterior(proposal)
                proposals.append(proposal)
                stage_posterior.append(proposal_posterior)

                # Calculate acceptance probability for the current stage
                alpha = self._acceptance_probability(stage_posterior=stage_posterior, proposals=proposals, t=t)
                u = np.log(np.random.rand()) if self.log else np.random.rand()

                # Accept the proposal with probability alpha
                if u < alpha:
                    current = proposal
                    current_posterior = proposal_posterior
                    if t >= burnin_index:
                        self.acc[stage, 1] += 1
                    break 

            self.samples[t, :] = current

            # Update running statistics
            delta = current - self.mean
            self.mean += delta / (t + 1)
            self.outer += np.outer(delta, current - self.mean)

            if t <= t0:
                self.C[t] = self.C[t-1]
            else:
                self.C[t] = ((self.scale_factor / t) * self.outer) + (self.eps * np.eye(self.dimension))
        
        self.samples = self.samples[burnin_index:, :]
        self.acc_rate = self.acc[:, 1] / self.acc[:, 0]
        return self.samples

class DREAM(Sampling):
    def __init__(self, distribution, chains=None, scale_factor=None, burnin=0.2, nCR=3, max_pairs=3, eps=1e-5, outlier_detection=True):
        """
        Initializes the DREAM sampling algorithm.
        
        Parameters:
        - distribution: Sampling object with the target distribution.
        - chains: Number of chains to use in the algorithm.
        - scale_factor: Scaling factor for the proposal covariance matrix.
        - burnin: Fraction of the total number of samples to discard as burn-in.
        - nCR: Number of crossover probabilities to explore.
        - max_pairs: Maximum number of pairs to use for the differential evolution.
        - eps: Small constant for numerical stability.
        - num_stages: Number of proposal stages for delayed rejection.
        - outlier_detection: Flag to enable outlier detection.
        """
        
        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for DREAM
        self.chains = max(10, self.dimension//2) if chains is None else chains
        self.scale_factor = 2.38 if scale_factor is None else scale_factor
        self.burnin = burnin
        self.max_pairs = max_pairs if max_pairs < (self.chains - 1) else self.chains - 1
        self.outlier_threshold = 2.0
        self.outlier_detection = outlier_detection
        self.eps = eps
        self.samples = None
        self.mean = None
        self.acc = None
        self.acc_rate = None
        self.posterior_history = None
        self.R_hat = None
        self.outlier_resets = 0

        # Initialize CR probabilities
        self.nCR = nCR # Number of crossover probabilities to explore
        self.p_a = np.ones(nCR) / nCR  # Equal probability for each crossover probability
        self.h_a = np.zeros(nCR)  # Counts of usage for each CR value
        self.Delta_a = np.zeros(nCR)  # Sum of squared jumping distances for each CR value

    def print_info(self):
        """
        Print information about the sampling method and its parameters.
        """
        print(f"Sampling Method: {self.__class__.__name__}")
        print(f"Number of Chains: {self.chains}")
        print(f"Scale Factor: {self.scale_factor}")
        print(f"Burn-in: {self.burnin}")
        print(f"Max Pairs: {self.max_pairs}")
        print(f"Epsilon: {self.eps}")
        print(f"Outlier Detection: {self.outlier_detection}")
        print(f"Outlier Resets: {self.outlier_resets}")
        print(f"R_hat: {self.R_hat}")
        print(f"nCR: {self.nCR}")
        print(f"p_a: {self.p_a}")
        print(f"h_a: {self.h_a}")
        print(f"Delta_a: {self.Delta_a}")
        print(f"Mean of Samples: {self.mean}")
        for chain in range(self.chains):
            print(f"Acceptance rate for chain {chain+1}: {(self.acc_rate[chain]*100):.2f}%")
        print("-----------------------------------------------------------------------------------------")
        print(f"Acceptance rate: {(np.mean(self.acc_rate)*100):.2f}%")

    def _initialize_population(self):
        """
        Initializes a population of chains using the given prior distribution.¨

        Returns:
        - initial: Initial population of points for the chains.
        """
        
        if (self.prior_means is None) or (self.prior_covs is None):
            initial = np.random.randn(self.chains, self.dimension)
        else:
            num_modes = len(self.prior_means)
            initial = np.zeros((self.chains, self.dimension))

            for chain in range(self.chains):
                mode_idx = np.random.choice(num_modes)
                initial[chain] = multivariate_normal.rvs(mean=self.prior_means[mode_idx], cov=self.prior_covs[mode_idx])

        return initial

    def _propose_point(self, current, chain, gen, CR):
        """
        Generates a candidate point for the given chain using differential evolution.

        Parameters:
        - current: Current population of points.
        - chain: Index of the chain to generate a proposal for.
        - gen: Current generation number.
        - CR: Crossover probability for the proposal.
        - stage: Current proposal stage.

        Returns:
        - proposal_point: Proposed point for the given chain
        """
        
        d = self.dimension
        proposal_point = current[chain].copy()
        num_pairs = np.random.randint(1, self.max_pairs + 1)

        randomized_dimensions = np.random.permutation(self.dimension)

        for i in randomized_dimensions:
            if np.random.rand() <= CR:
                if gen % 5 == 0:
                    beta = 1.0
                else:
                    beta = self.scale_factor / np.sqrt(2*num_pairs*d)

                diff = 0
                for _ in range(num_pairs):
                    indices = np.random.choice([x for x in range(self.chains) if x != chain] , 2, replace=False)
                    diff += current[indices[0]][i] - current[indices[1]][i]

                e = np.random.uniform(-self.eps, self.eps)
                eps = np.random.normal(0, self.eps)
                proposal_point[i] = current[chain][i] + ((1 + e) * (beta * diff)) + eps
            else:
                d -= 1

        return proposal_point

    def _outlier_detection(self, current, current_posterior, gen):
        """
        Detects outlier chains and resets them to the best chain.

        Parameters:
        - current: Current population of points.
        - current_posterior: Posterior values for the current population.
        - gen: Current generation number.

        Returns:
        - None
        """
        
        # IQR statistics for outlier detection
        q75, q25 = np.percentile(self.posterior_history[(gen // 2):gen].flatten(), [75, 25])
        iqr = q75 - q25
        
        # Identify outlier chains
        if self.log:
            chain_means = np.mean(self.posterior_history[(gen // 2):gen], axis=0)
        else:
            chain_means = np.mean(np.log(self.posterior_history[(gen // 2):gen]), axis=0)

        outlier_threshold = q25 - abs(self.outlier_threshold * iqr)
        outliers = chain_means < outlier_threshold
        outlier_indices = np.where(outliers)[0]

        # Reset outlier chains to the currently best chain
        best_chain_index = np.argmax(chain_means)
        for idx in outlier_indices:
            self.outlier_resets += 1
            current[idx] = current[best_chain_index]
            current_posterior[idx] = current_posterior[best_chain_index]

    def _compute_gelman_rubin(self):
        """
        Computes the Gelman-Rubin convergence diagnostic R-hat for each dimension.
        Uses the last 50% of samples in each chain.
        """

        N = self.samples.shape[0] // 2
        samples_half = self.samples[N:, :, :]

        R_hat = np.zeros(self.dimension)

        for j in range(self.dimension):
            # Mean across chains for each sample in dimension j
            chain_means = np.mean(samples_half[:, :, j], axis=0)  # Shape (num_chains,)
            #overall_mean = np.mean(chain_means)

            # Between-chain variance B
            B = N * np.var(chain_means, ddof=1)

            # Within-chain variance W
            W = np.mean([np.var(samples_half[:, chain, j], ddof=1) for chain in range(self.chains)])

            # Gelman-Rubin R-hat for dimension j
            R_hat[j] = np.sqrt((((N - 1) * W) + B) / (W * N))

        self.R_hat = R_hat

    def _reset(self):
        """
        Resets the DREAM algorithm by reinitializing the population and other parameters.
        """
        
        self.samples = None
        self.mean = None
        self.acc = None
        self.acc_rate = None
        self.posterior_history = None
        self.R_hat = None
        self.outlier_resets = 0
        self.h_a = np.zeros(self.nCR)  # Counts of usage for each CR value
        self.Delta_a = np.zeros(self.nCR)  # Sum of squared jumping distances for each CR value
        self.p_a = np.ones(self.nCR) / self.nCR  # Equal probability for each crossover probability

    def sample(self, initial=None, N=10000):
        """
        Runs the DREAM algorithm over a specified number of generations.
        
        Parameters:
        - initial: Initial population of points for the chains.
        - N: Number of generations to run the algorithm.
        
        Returns:
        - Array of sampled points from the posterior distribution.
        """

        self._reset()  # Reset the algorithm before sampling
        current = self._initialize_population() if initial is None else initial
        current_posterior = [self.posterior(u) for u in current]
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.chains, self.dimension))
        self.posterior_history = np.zeros((N + burnin_index, self.chains))
        self.acc = np.zeros(self.chains)

        for gen in range(N + burnin_index):
            # Propose new points for each chain
            proposal = current.copy()
            for chain in range(self.chains):
                # Sample crossover probability CR = a / nCR
                a = np.random.choice(np.arange(1, self.nCR + 1), p=self.p_a)
                CR = a / self.nCR
                self.h_a[a - 1] += 1  # Count usage of this CR value

                # Proposal generation using differential evolution
                proposal_chain = self._propose_point(current, chain, gen, CR)
                proposal_posterior = self.posterior(proposal_chain)

                alpha = min(0, proposal_posterior - current_posterior[chain]) if self.log else min(1, proposal_posterior / current_posterior[chain])
                u = np.log(np.random.rand()) if self.log else np.random.rand()
                # Accept the proposal with probability alpha
                if u < alpha:
                    proposal[chain] = proposal_chain
                    current_posterior[chain] = proposal_posterior
                    self.acc[chain] += 1 if gen >= burnin_index else 0

                    # Calculate the squared jumping distance ∆m
                    delta_m = np.sum(((proposal[chain] - current[chain]) / np.std(proposal, axis=0)) ** 2)
                    self.Delta_a[a - 1] += delta_m

            current = proposal        
            self.posterior_history[gen] = current_posterior
            # Store samples post-burn-in
            if gen >= burnin_index:
                self.samples[gen-burnin_index, :] = current
            elif self.outlier_detection and gen > 0 and gen % 100 == 0:
                self._outlier_detection(current, current_posterior, gen)
            
            if gen == (burnin_index - 1):
                # Update the probabilities p_m based on ∆m
                total_Delta = np.sum(self.Delta_a / self.h_a)
                for a in range(self.nCR):
                    self.p_a[a] = (self.Delta_a[a] / self.h_a[a]) / total_Delta if (total_Delta > 0 and self.h_a[a] > 0) else self.p_a[a]

        self.acc_rate = self.acc / N
        self._compute_gelman_rubin()
        self.samples = self.samples.transpose(1, 0, 2)  # Change shape to (chains, N, dimension)
        self.mean = np.mean(self.samples, axis=(1, 0))  # Mean of samples across chains
        return self.samples

def compute_autocorrelation_old(samples, max_lag=None):
    """
    Compute the autocorrelation function (ACF) and the autocorrelation length (s_f)
    for an N-dimensional time series by averaging the per-dimension autocorrelations.
    
    Parameters:
      samples (array-like): 2D array of shape (num_samples, d) for a single chain.
      max_lag (int): Maximum lag to consider (default is min(100, num_samples//2)).
      
    Returns:
      acfs (np.ndarray): 1D array of averaged autocorrelations for lags 1,2,...,max_lag.
      s_f (float): Autocorrelation length computed as 1 + 2 * sum_{lag: acf(lag) > 0} acf(lag).
    """

    N, d = samples.shape
    max_lag = min(1000, N // 2) if max_lag is None else max_lag
    acfs = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acfs[lag - 1] = np.mean([np.corrcoef(samples[:-lag, j], samples[lag:, j])[0, 1] for j in range(d)])
    s_f = 1 + 2 * np.sum(acfs[acfs > 0])
    return acfs, s_f

def compute_autocorrelation(samples):
    """
    Compute the autocorrelation function (ACF) and the integrated autocorrelation 
    length (s_f) for multi-chain samples.
    
    Parameters:
      samples (np.ndarray): Array of shape (num_chains, N, d) of MCMC samples.
      
    Returns:
      s_f_mean (float): Mean integrated autocorrelation time over dimensions.
      s_f_max (float): Maximum integrated autocorrelation time over dimensions.
    """
    samples=samples[np.newaxis, :, :] if samples.ndim == 2 else samples
    no_chains, N, d = samples.shape

    # Compute the ACF for each chain and each dimension.
    acf_list = []
    for i in range(no_chains):
        acf_chain = np.empty((N, d))
        for j in range(d):
            acf_chain[:, j] = emcee.autocorr.function_1d(samples[i, :, j])
        acf_list.append(acf_chain)
    
    # Determine the maximum length among the computed ACFs
    lengths = np.array([acf.shape[0] for acf in acf_list])
    max_length = lengths.max()

    # Create a 3D array (no_chains x max_length x d) and pad with NaN where needed.
    acf_array = np.full((no_chains, max_length, d), np.nan)
    for i in range(no_chains):
        L = lengths[i]
        acf_array[i, :L, :] = acf_list[i]

    # Compute the mean ACF over chains for each lag and dimension, ignoring NaNs.
    mean_acf = np.nanmean(acf_array, axis=0)  # shape: (max_length, d)

    # Compute the integrated autocorrelation time for each dimension.
    s_f_all = np.zeros(d)
    for j in range(d):
        s_f_all[j] = autocorr_FM(mean_acf[:, j], c=5)

    s_f_mean = np.mean(s_f_all)
    s_f_max = np.max(s_f_all)

    return s_f_mean, s_f_max, np.mean(mean_acf[:(2*int(s_f_max)),:], axis=1)

def autocorr_FM(f, c: int = 5):
    taus = 2.0 * np.cumsum(f) - 1.0
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        window = np.argmin(m)
    else:
        window = len(taus) - 1

    return taus[window]

def compute_gelman_rubin(samples):
        """
        Computes the Gelman-Rubin convergence diagnostic R-hat for each dimension.
        Uses the last 50% of samples in each chain.
        """
        samples = np.stack(samples, axis=0) if isinstance(samples, list) else samples
        chains, N, d = samples.shape
        N = N // 2
        samples_half = samples[:, N:, :]

        R_hat = np.zeros(d)

        for j in range(d):
            # Mean across chains for each sample in dimension j
            chain_means = np.mean(samples_half[:, :, j], axis=1)  # Shape (num_chains,)

            # Between-chain variance B
            B = N * np.var(chain_means, ddof=1)

            # Within-chain variance W
            W = np.mean([np.var(samples_half[chain, :, j], ddof=1) for chain in range(chains)])

            # Gelman-Rubin R-hat for dimension j
            R_hat[j] = np.sqrt((((N - 1) * W) + B) / (W * N))

        return (np.mean(R_hat), np.max(R_hat))

def normalized_euclidean_distance(samples, true_mean):
    """
    Compute the normalized Euclidean distance between the estimated mean 
    from MCMC samples and the true mean of the target distribution.
    
    Parameters:
        samples (np.ndarray): MCMC samples, shape (num_samples,) for 1D 
                              or (num_samples, dim) for multi-dimensional.
        true_mean (float or np.ndarray): The true mean of the target distribution.
    
    Returns:
        float: Normalized Euclidean distance.
    """

    est_mean = np.mean(samples, axis=0)
    print(f"Estimated mean: {est_mean}, True mean: {true_mean}")

    euclidean_dist = np.linalg.norm(est_mean - true_mean)
    norm_factor = np.linalg.norm(true_mean) if np.linalg.norm(true_mean) > 0 else 1.0
    normalized_dist = euclidean_dist / norm_factor

    return normalized_dist

def compute_function_evaluations(len_samples, sampler):

    if isinstance(sampler, DRAM):
        function_evaluations = float(np.sum(sampler.acc[:, 0]))
    else:
        function_evaluations = float(len_samples)

    return function_evaluations

def benchmark(sampler, chains=10, num_samples=10000, initial=None, ranges=[], max_lag=None, visualize=False,
              Acc=True, distance=False, autocorrelation=True, ESS=True, cost=True, R_hat=True, KL=False, old=False):
    """
    Benchmark a given sampling algorithm by running it multiple times, computing diagnostics, and (optionally) visualizing.
    
    Parameters:
        sampler (Sampling): The sampling algorithm to benchmark.
        chains (int): Number of chains to run.
        num_samples (int): Number of samples to generate per chain.
        initial (np.ndarray): Initial point for the chain.
        ranges (list): Ranges for the sampling algorithm.
        max_lag (int): Maximum lag for autocorrelation computation.
        visualize (bool): Whether to visualize the samples and diagnostics.
        Acc (bool): Whether to compute acceptance rates.
        distance (bool): Whether to compute the distance from the true mean.
        autocorrelation (bool): Whether to compute autocorrelation statistics.
        ESS (bool): Whether to compute effective sample size.
        cost (bool): Whether to compute cost statistics.
        R_hat (bool): Whether to compute R-hat statistics.
        KL (bool): Whether to compute KL divergence.
        old (bool): Whether to use the old autocorrelation method.
    
    Returns:
        df_stats (pd.DataFrame): DataFrame containing the computed statistics.
        all_samples (np.ndarray): Array of all samples generated by the sampler.
    """

    all_samples = []
    acc_rates = []
    if isinstance(sampler, DREAM):
        sampler.chains = chains
        sampler.sample(N=num_samples, initial=initial)
        all_samples = [np.asarray(sampler.samples[i,:,:]) for i in range(sampler.chains)]
        acc_rates = list(sampler.acc_rate)
    elif isinstance(sampler, MH) or isinstance(sampler, AM) or isinstance(sampler, DRAM):
        for i in range(chains):
            samples = sampler.sample(N=num_samples, initial=initial)
            all_samples.append(samples)
            if isinstance(sampler, DRAM):
                acc_rates.append((sum(sampler.acc[:,1])/len(samples)))
            else:
                acc_rates.append(sampler.acc_rate)
    else:
        raise ValueError("Unknown sampler type.")
    
    ESS = True if cost else ESS
    autocorrelation = True if ESS else autocorrelation
    
    all_stats = []
    total_evaluations = 0.0
    for i, samples in enumerate(all_samples):
        stats = {"Chains": f"Chain {i+1}"}
        if distance and hasattr(sampler, "true_mean"):
            stats["Distance"] = round(normalized_euclidean_distance(samples=samples, true_mean=sampler.true_mean), 3)

        if autocorrelation:
            IAT_avg, IAT_max, acfs = compute_autocorrelation(samples=samples)
            stats["IAT_avg"] = round(IAT_avg, 3)
            stats["IAT_max"] = round(IAT_max, 3)

        if old:
            _, old_IAT_avg = compute_autocorrelation_old(samples=samples, max_lag=max_lag)
            stats["old_IAT_avg"] = round(old_IAT_avg, 3)

        if ESS:
            stats["ESS"] = len(samples) // IAT_avg if IAT_avg > 0 else np.nan

        if cost:
            function_evaluations = compute_function_evaluations(len_samples=len(samples), sampler=sampler)
            total_evaluations += function_evaluations
            stats["Cost_per_sample"] = round(function_evaluations / len(samples), 3) if len(samples) > 0 else np.inf
            stats["Cost_per_ESS"] = round(function_evaluations / stats["ESS"], 3) if stats["ESS"] > 0 else np.inf

        if Acc and acc_rates:
            stats["Acc"] = round(acc_rates[i], 2)

        if R_hat:
            stats["R_hat_avg"] = np.nan
            stats["R_hat_max"] = np.nan

        if KL:
            stats["KL"] = round(sampler.sampling_quality(samples, ranges=ranges), 4)

        all_stats.append(stats)
        
    keys = all_stats[0].keys()
    stats = {"Chains": "Overall"}
    for key in keys:
        if key == "Distance":
            stats[key] = round(normalized_euclidean_distance(samples=np.stack(all_samples, axis=0).reshape(-1, 2), true_mean=sampler.true_mean), 3)
        elif key == "R_hat_avg":
            R_hat_avg, R_hat_max = compute_gelman_rubin(all_samples)
            stats["R_hat_avg"] = round(R_hat_avg, 4)
            stats["R_hat_max"] = round(R_hat_max, 4)
        elif key == "IAT_avg":
            IAT_avg, IAT_max, acfs = compute_autocorrelation(samples=np.stack(all_samples, axis=0))
            stats["IAT_avg"] = round(IAT_avg, 3)
            stats["IAT_max"] = round(IAT_max, 3)
        elif key == "ESS":
            stats[key] = round((len(samples) * chains) // stats["IAT_avg"], 3) if autocorrelation else np.nan
        elif key == "Cost_per_sample":
            stats["Cost_per_sample"] = round(total_evaluations / (len(samples) * chains), 3) if (len(samples) * chains) > 0 else np.inf
            stats["Cost_per_ESS"] = round(total_evaluations / stats["ESS"], 3) if stats["ESS"] > 0 else np.inf
        elif key == "KL":
            stats[key] = round(sampler.sampling_quality(samples=np.stack(all_samples, axis=0), ranges=ranges, visualise=visualize), 4)
        elif key not in ["Chains", "R_hat_max", "IAT_max", "Cost_per_ESS"]:
            stats[key] = round(np.mean([stat[key] for stat in all_stats]), 3)
        
    all_stats.append(stats)
    df_stats = pd.DataFrame(all_stats)
    
    if visualize:
        axes = 2 if autocorrelation else 1
        fig, axs = plt.subplots(axes, 1, figsize=(15, 5*axes))
        axs = [axs] if axes == 1 else axs
        sampler.visualize_samples_trace(visual=np.stack(all_samples, axis=0), ax=axs[0], show=False)
        if autocorrelation:
            sampler.visualize_samples_acf(visual=None, ax=axs[1], IAT_avg=IAT_avg, IAT_max=IAT_max, acfs=acfs, show=False)
        plt.tight_layout()
        plt.show()
    
    return df_stats, np.stack(all_samples, axis=0)

def benchmark_average(sampler, runs=10, chains=10, num_samples=10000, initial=None, ranges=[], max_lag=None, visualize=False,
                      Acc=True, distance=False, autocorrelation=True, ESS=True, cost=True, R_hat=True, KL=False, old=False):
    """
    Run the benchmark function multiple times and combine the statistics into a single DataFrame.

    Parameters:
        sampler (Sampling): The sampling algorithm to benchmark.
        runs (int): Number of runs to perform.
        chains (int): Number of chains to use.
        num_samples (int): Number of samples per run.
        initial (np.ndarray): Initial point for the chain.
        ranges (list): List of ranges for KL divergence calculation.
        max_lag (int): Maximum lag for autocorrelation calculation.
        visualize (bool): Whether to visualize the results (only for the last run).
        Acc (bool): Whether to compute acceptance rates.
        distance (bool): Whether to compute normalized Euclidean distance.
        autocorrelation (bool): Whether to compute autocorrelation length and ESS.
        ESS (bool): Whether to compute effective sample size.
        cost (bool): Whether to compute cost metrics.
        R_hat (bool): Whether to compute Gelman-Rubin convergence diagnostic.
        KL (bool): Whether to compute KL divergence.
        old (bool): Whether to compute old autocorrelation metrics.

    Returns:
        pd.DataFrame: DataFrame containing combined statistics for all runs and an overall average row.
    """
    chain_stats = []
    runs_stats = []
    all_samples = []
    for run in range(runs):
        visualize_run = visualize and (run == runs - 1)
        df_stats, samples = benchmark(sampler, chains=chains, num_samples=num_samples, initial=initial, ranges=ranges, max_lag=max_lag,
                                    visualize=visualize_run, Acc=Acc, distance=distance, autocorrelation=autocorrelation,
                                    ESS=ESS, cost=cost, R_hat=R_hat, KL=KL, old=old)
        all_samples.append(samples)
        chain_stats.append(df_stats.copy())
        overall_row = df_stats[df_stats["Chains"] == "Overall"].copy()
        overall_row["Chains"] = f"Run {run + 1}"
        runs_stats.append(overall_row)

    # Combine all runs into a single DataFrame
    runs_stats = pd.concat(runs_stats, ignore_index=True)
    runs_stats.rename(columns={"Chains": "Runs"}, inplace=True)

    # Add an overall row with averages
    overall_row = runs_stats.iloc[:, 1:].mean(numeric_only=True).round(4).to_dict()
    overall_row["Runs"] = "Average"
    runs_stats = pd.concat([runs_stats, pd.DataFrame([overall_row])], ignore_index=True)

    return runs_stats, all_samples, chain_stats

def benchmark_parametrs(sampler, param_dict, benchmark_inputs):
    """
    Benchmark different settings of sampler-specific parameters.

    Parameters:
        sampler (Sampling): The sampling algorithm to benchmark.
        param_dict (dict): Dictionary of parameter names and their list of values to try.
        benchmark_inputs (dict): Dictionary of inputs for the "benchmark_average" function.

    Returns:
        pd.DataFrame: DataFrame containing benchmark results for each parameter combination.
    """

    # Validate that all keys in param_dict are attributes of the sampler
    invalid_keys = [key for key in param_dict if not hasattr(sampler, key)]
    if invalid_keys:
        raise ValueError(f"The sampler does not have the following attributes: {invalid_keys}")
    
    results = []

    # Generate all combinations of parameter values
    param_names = list(param_dict.keys())
    param_values = list(itertools.product(*param_dict.values()))

    for values in param_values:
        # Set the parameters for the sampler
        for name, value in zip(param_names, values):
            if value is not None:
                setattr(sampler, name, value)

        # Run the benchmark
        runs_stats, _, _ = benchmark_average(sampler, **benchmark_inputs)

        # Extract the last row (average row) and remove the "Runs" column
        avg_row = runs_stats.iloc[-1].drop("Runs").to_dict()

        # Add parameter values to the row
        for name, value in zip(param_names, values):
            avg_row[name] = value

        # Append the row to the results
        results.append(avg_row)

    # Create a DataFrame from the results
    columns = param_names + list(runs_stats.columns[1:])
    return pd.DataFrame(results, columns=columns)