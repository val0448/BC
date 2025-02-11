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
    - likelihood (Callable): The likelihood function for a given problem.
    - forward_model (Callable): The forward model function for a given problem.
    - observed_data (np.ndarray): The observed data for a given problem.
    - noise_std (float): The standard deviation of the noise in the data.
    - prior (Callable): The prior function for a given problem.
    - prior_means (np.ndarray): The means of the prior distribution.
    - prior_stds (np.ndarray): The standard deviations of the prior distribution.
    - log (bool): If True, the function returns the log-probability density.

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

    def __init__(self, dimension=2, posterior=None, likelihood=None, forward_model=None, observed_data=None, noise_std=None, 
                 prior=None, prior_means=None, prior_stds=None, log=False):
        """
        Constructor for the Sampling class.

        Parameters:
        - dimension (int): The dimension of the parameter.
        - posterior (Callable): The posterior function for a given problem.
        - prior (Callable): The prior function for a given problem.
        - prior_means (np.ndarray): The means of the prior distribution.
        - prior_stds (np.ndarray): The standard deviations of the prior distribution.
        - likelihood (Callable): The likelihood function for a given problem.
        - forward_model (Callable): The forward model function for a given problem.
        - observed_data (np.ndarray): The observed data for a given problem.
        - noise_std (float): The standard deviation of the noise in the data.
        - log (bool): If True, the function returns the log-probability density.

        Returns:
        - None

        Raises:
        - None

        Example usage:
        ```
        sampler = Sampling(dimension=2, posterior=posterior_function)
        ```
        """

        self.dimension = dimension
        self.log = log
        self.likelihood = self._setup_likelihood(likelihood, forward_model, observed_data, noise_std, posterior)
        self.prior = self._setup_prior(prior, prior_means, prior_stds, posterior)
        self.posterior = self._setup_posterior(posterior)

    def _setup_likelihood(self, likelihood, forward_model, observed_data, noise_std, posterior):
        """
        Creates the likelihood of a forward model given the data.
        
        Parameters:
        - forward_model (Callable): Forward model function that takes a point and returns a prediction.
        - observed_data (np.ndarray): Observed data.
        - std_dev (float): Standard deviation of the noise in the data.
        
        Returns:
        - float: Likelihood of the forward model given the data.
        """

        def likelihood_func(point):
            """
            Computes the likelihood of the forward model given the data.

            Parameters:
            - point (np.ndarray): The point at which to evaluate the likelihood.

            Returns:
            - float: The likelihood of the forward model given the data.
            """

            y_pred = self.forward_model(point)
            likelihood = -((self.observed_data - y_pred) ** 2) / (2 * self.noise_std ** 2)
            if self.log:
                return likelihood
            else:
                return np.exp(likelihood)

        self.forward_model = forward_model
        self.observed_data = observed_data
        self.noise_std = noise_std

        if (likelihood is None) and (posterior is None):
            if self.forward_model is None or self.observed_data is None or self.noise_std is None:
                raise ValueError("`forward_model`, `observed_data`, and `noise_std` are required to set up the likelihood.")
            return likelihood_func
        else:
            return likelihood       
    
    def _setup_prior(self, prior, prior_means, prior_stds, posterior):
        """
        Creates the prior function.

        Parameters:
        - prior_means (np.ndarray): The means of the prior distribution.
        - prior_stds (np.ndarray): The standard deviations of the prior distribution.

        Returns:
        - function: A function that takes a point and returns the prior probability density.
        """

        self.prior_means = prior_means
        self.prior_stds = prior_stds

        if (prior is None) and (posterior is None):
            if (self.prior_means) is None or (self.prior_stds) is None:
                raise ValueError("`prior_means` and `prior_stds` are required to set up the prior.")
            if len(self.prior_means) != len(self.prior_stds):
                raise ValueError("`prior_means` and `prior_stds` must have the same length.")

            return self._create_gaussian_pdf()
        else:
            return prior

    def _setup_posterior(self, posterior):
        """
        Creates the posterior function for a given point.

        Parameters:
        - None

        Returns:
        - function: A function that takes a point and returns the posterior probability density.
        """

        def posterior_func(point):
            """
            Computes the posterior probability density for a given point.

            Parameters:
            - point (np.ndarray): The point at which to evaluate the posterior.

            Returns:
            - float: The posterior probability density at the given point.
            """

            if self.log:
                return self.likelihood(point) + self.prior(point)
            else:
                return self.likelihood(point) * self.prior(point)

        if posterior is None:
            if (self.likelihood is None) or (self.prior is None):
                raise ValueError("`likelihood` and `prior` are required to set up the posterior.")
            return posterior_func
        else:
            return posterior

    def _create_gaussian_pdf(self):
        """
        Creates a Gaussian probability density function for a given mean and standard deviation.

        Parameters:
        - None

        Returns:
        - function: A function that takes a point and returns the Gaussian PDF.
        """  

        def gaussian_pdf(point = None, mean=None, std=None):
            """
            Computes the Gaussian probability density function for a given mean and standard deviation.

            Parameters:
            - point (np.ndarray): The point at which to evaluate the PDF.
            - mean (np.ndarray): The mean of the Gaussian distribution.
            - std (float): The standard deviation of the Gaussian distribution.

            Returns:
            - float: The Gaussian PDF at the given point.
            """

            if point is None:
                point = np.zeros(self.dimension)
            if mean is None:
                mean = self.prior_means[0]
            if std is None:
                std = self.prior_stds[0]
            
            point = np.array(point)
            mean = np.array(mean)
            std = np.array(std)

            # Construct the covariance matrix with std deviations per dimension
            inv_covariance = np.diag(1 / (std ** 2))

            # Compute the Gaussian PDF in log space
            pos = point - mean
            exponent = -0.5 * (np.dot(pos.T, inv_covariance).dot(pos))
            #normalization_constant_log = -0.5 * self.dimension * np.log(2 * np.pi * std ** 2)
            pdf = exponent #+ normalization_constant_log

            return pdf if self.log else np.exp(pdf)

        def combined_gaussian_pdf(point):
            """
            Computes the combined Gaussian probability density function for multiple means and standard deviations.

            Parameters:
            - point (np.ndarray): The point at which to evaluate the PDF.

            Returns:
            - float: The combined Gaussian PDF at the given point.
            """

            densities = [gaussian_pdf(point, mean, std) for mean, std in zip(self.prior_means, self.prior_stds)]

            if self.log:
                max_density = np.max(densities)
                combined_log_density = max_density + np.log(np.sum(np.exp(densities - max_density)))
                return combined_log_density
            else:
                return np.sum(densities)# / len(self.prior_means)

        if len(self.prior_means) > 0:
            return combined_gaussian_pdf
        else:
            raise ValueError("Invalid number of prior means and standard deviations.")

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

        # visuals_map = {"prior": self.prior, 
        #                "posterior": self.posterior, 
        #                "likelihood": self.likelihood, 
        #                "forward_model": self.forward_model, 
        #                "samples": self.samples, 
        #                "kde": self.kde}

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
                    grid = self.create_grid(visual=visual, ranges=ranges, max_points=max_points//2)
                # Visualize samples in histograms and scatter plots (2 separate plots)
                self.visualize_samples_hist(visual, grid, ax=next(axis_iter), show = False)
                if self.dimension < 3:
                    self.visualize_samples_scatter(visual, ax=next(axis_iter), show = False)
            elif isinstance(visual, dict) and visual.get("is_kde"):
                # If it's a KDE approximation
                self.visualize_kde(visual['data'], grid, ax=next(axis_iter))
            elif callable(visual):
                if grid is None:
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
        if self.log:
            posterior_values = np.exp(posterior_values)  # Convert log posterior to standard form for visualization

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
        if ranges is None:
            ranges = []

        # Case 1: If samples are provided, use them to calculate the grid bounds
        if isinstance(visual, np.ndarray):
            # Check if samples come from multiple chains
            if visual.ndim == 3:
                visual = visual.transpose(1, 0, 2).reshape(-1, self.dimension)
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
        if html:
            return HTML(anim.to_jshtml())  # Return the HTML representation for display
        else:
            plt.show()

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

        # Check if samples come from multiple chains
        if samples.ndim == 3:
            samples = samples.transpose(1, 0, 2).reshape(-1, self.dimension)
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
        Initialize the Random Walk Metropolis algorithm.

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
        self.C = np.eye(self.dimension) if initial_cov is None else initial_cov
        self.acc_rate = None
        self.samples = None

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_stds is None):
            initial = np.random.randn(self.dimension)
        else:
            num_modes = len(self.prior_means)
            mode_idx = np.random.choice(num_modes)
            initial = [np.random.normal(mean, std) for mean, std in zip(self.prior_means[mode_idx], self.prior_stds[mode_idx])]

        return np.array(initial)

    def sample(self, initial=None, N=10000):
        """
        Run the Adaptive Metropolis algorithm for n_samples iterations.

        Parameters:
        - x0 (np.ndarray): Initial point for the chain.
        - N (int): Number of samples to generate.

        Returns:
        - None
        """

        current = self._initial_sample() if initial is None else initial
        acc = 0
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))
        
        for t in range(N + burnin_index):
            # Propose a new point
            proposal = np.random.multivariate_normal(current, self.scale_factor * self.C)
            proposal_posterior = self.posterior(proposal)
            if self.log:
                alpha = min(0, proposal_posterior - current_posterior)
                u = np.log(np.random.rand())
            else:
                alpha = min(1, proposal_posterior / current_posterior)
                u = np.random.rand()
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

class AM(Sampling):
    """
    Adaptive Metropolis algorithm for sampling from a target distribution.
    Uses a Gaussian proposal distribution.

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

    def __init__(self, distribution, initial_cov=None, scale_factor=None, burnin=0.2, eps=1e-5, update_step=1):
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

        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for AM
        self.eps = eps
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.mean = np.zeros(self.dimension)
        self.burnin = burnin
        self.update_step = update_step
        self.C = [np.eye(self.dimension)] if initial_cov is None else [initial_cov]
        self.acc_rate = None
        self.samples = None

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_stds is None):
            initial = np.random.randn(self.dimension)
        else:
            num_modes = len(self.prior_means)
            mode_idx = np.random.choice(num_modes)
            initial = [np.random.normal(mean, std) for mean, std in zip(self.prior_means[mode_idx], self.prior_stds[mode_idx])]

        return np.array(initial)    

    def sample(self, initial=None, N=10000):
        """
        Run the Adaptive Metropolis algorithm for n_samples iterations.

        Parameters:
        - x0 (np.ndarray): Initial point for the chain.
        - N (int): Number of samples to generate.

        Returns:
        - None
        """

        current = self._initial_sample() if initial is None else initial
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))
        acc = 0
        
        for t in range(N + burnin_index):
            # Propose a new point
            proposal = np.random.multivariate_normal(current, self.scale_factor * self.C[-1])
            proposal_posterior = self.posterior(proposal)
            if self.log:
                alpha = min(0, proposal_posterior - current_posterior)
                u = np.log(np.random.rand())
            else:
                alpha = min(1, proposal_posterior / current_posterior)
                u = np.random.rand()
            # Accept the proposal with probability alpha
            if u < alpha:
                current = proposal  
                current_posterior = proposal_posterior
                if t >= burnin_index:
                    acc += 1
            if t >= burnin_index:
                self.samples[t-burnin_index, :] = current
                if (t >= burnin_index + 1) and (t - burnin_index) % self.update_step == 0:
                    # Update covariance matrix with the recursive formula
                    old_mean = self.mean
                    self.mean = old_mean + (current - old_mean) / t
                    diff = np.outer(current - self.mean, current - self.mean)
                    self.C.append((t - 1) / t * self.C[-1] + (self.scale_factor / t) * (diff + self.eps * np.eye(self.dimension)))
        
        self.acc_rate = acc / N

class DRAM(Sampling):
    """
    Delayed Rejection Adaptive Metropolis (DRAM) algorithm for sampling from a target distribution.
    Uses a Gaussian proposal distribution.
    
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

    def __init__(self, distribution, initial_cov=None, scale_factor=None, burnin=0.2, eps=1e-5, update_step=1, gammas=None, num_stages=3):
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

        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for DRAM
        self.scale_factor = (2.4**2) / self.dimension if scale_factor is None else scale_factor
        self.eps = eps  # Small constant for numerical stability
        self.burnin = burnin  # Non-adaptive period
        self.num_stages = num_stages  # Number of delayed rejection stages
        self.update_step = update_step  # Update covariance matrix every n samples
        self.gammas = [1.0] + [0.5**i for i in range(1, num_stages)] if gammas is None else gammas
        self.C = [np.eye(self.dimension)] if initial_cov is None else [initial_cov] # list of covariance matrices
        self.samples = None  # To store past samples
        self.mean = np.zeros(self.dimension)  # Mean of the samples
        self.acc = None  # Count of accepted samples
        self.acc_rate = None  # Acceptance rate of the samples

    def _initial_sample(self):
        """
        Generate an initial sample.

        Returns:
        - np.ndarray: Initial sample.
        """
        
        if (self.prior_means is None) or (self.prior_stds is None):
            initial = np.random.randn(self.dimension)
        else:
            num_modes = len(self.prior_means)
            mode_idx = np.random.choice(num_modes)
            initial = [np.random.normal(mean, std) for mean, std in zip(self.prior_means[mode_idx], self.prior_stds[mode_idx])]

        return np.array(initial)

    def _acceptance_probability(self, stage_posterior):
        """
        Calculate the acceptance probability for a given likelihood history.
        
        Parameters:
        - lh: Likelihood history for the current stage.

        Returns:
        - alpha: Acceptance probability for the likelihood history.
        """

        if self.log:
            alpha = (stage_posterior[-1] - stage_posterior[0]) if stage_posterior[0] > -np.inf else 0.0
            numerator = 0.0
            denominator = 0.0
        else:
            alpha = (stage_posterior[-1] / stage_posterior[0]) if stage_posterior[0] > 0 else 1.0
            numerator = 1.0
            denominator = 1.0

        for i in range(len(stage_posterior) - 2, 0, -1):
            if self.log:
                numerator += np.log1p(-np.exp(self._acceptance_probability(stage_posterior[i:][::-1])))
                denominator += np.log1p(-np.exp(self._acceptance_probability(stage_posterior[:len(stage_posterior) - i])))
            else:
                numerator *= (1 - self._acceptance_probability(stage_posterior[i:][::-1]))
                denominator *= (1 - self._acceptance_probability(stage_posterior[:len(stage_posterior) - i]))
        
        if self.log:
            alpha += (numerator - denominator) if denominator > -np.inf else 0.0
        else:
            alpha *= (numerator / denominator) if denominator > 0 else 1.0
        
        return min(0, alpha) if self.log else min(1, alpha)

    def sample(self, initial=None, N=10000):
        """
        Run the DRAM algorithm with multiple proposal stages.

        Parameters:
        - x0: Initial point for the chain.
        - N: Number of samples to generate.

        Returns:
        - None
        """
    
        current = self._initial_sample() if initial is None else initial
        current_posterior = self.posterior(current)
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.dimension))
        self.acc = np.zeros((self.num_stages, 2))

        for t in range(N + burnin_index):
            stage_posterior = [current_posterior]
            for stage in range(self.num_stages):
                if t >= burnin_index:
                    self.acc[stage, 0] += 1

                # Propose a new point for the current stage
                proposal = np.random.multivariate_normal(current, self.scale_factor * self.gammas[stage] * self.C[-1])
                proposal_posterior = self.posterior(proposal)
                stage_posterior.append(proposal_posterior)

                # Calculate acceptance probability for the current stage
                if self.log:
                    alpha = self._acceptance_probability(stage_posterior=stage_posterior)
                    u = np.log(np.random.rand())
                else:
                    alpha = self._acceptance_probability(stage_posterior=stage_posterior)
                    u = np.random.rand()
                # Accept the proposal with probability alpha
                if u < alpha:
                    current = proposal
                    current_posterior = proposal_posterior
                    if t >= burnin_index:
                        self.acc[stage, 1] += 1
                    break 

            if t >= burnin_index:
                self.samples[t-burnin_index, :] = current
                if (t >= burnin_index + 1) and (t - burnin_index) % self.update_step == 0:
                    # Update covariance matrix with the recursive formula
                    old_mean = self.mean
                    self.mean = old_mean + (current - old_mean) / t
                    diff = np.outer(current - self.mean, current - self.mean)
                    self.C.append((t - 1) / t * self.C[-1] + (self.scale_factor / t) * (diff + self.eps * np.eye(self.dimension)))
        
        self.acc_rate = self.acc[:, 1] / self.acc[:, 0]

class DREAM(Sampling):
    def __init__(self, distribution, chains=None, scale_factor=None, burnin=0.2, nCR=3, max_pairs=3, eps=1e-5, num_stages=3, outlier_detection=True):
        """
        Initializes the DREAM sampling algorithm.
        
        Parameters:
        - posterior: Target posterior distribution.
        - dimension: Dimensionality of the parameter space.
        - num_chains: Number of chains for sampling.
        - crossover_prob (CR): Crossover probability for subspace sampling.
        - scale_factor: Scaling factor for proposal step sizes.
        - burnin_ratio: Fraction of the total steps for burn-in.
        - nCR: Number of crossover probabilities to explore during burn-in.
        """
        
        # Initialize using Sampling's constructor with attributes from the `distribution` instance
        super().__init__(**distribution.__dict__)

        # Additional attributes for DREAM
        self.chains = max(10, self.dimension//2) if chains is None else chains
        self.scale_factor = 2.38 if scale_factor is None else scale_factor
        self.burnin = burnin
        self.max_pairs = max_pairs
        self.outlier_threshold = 2.0
        self.outlier_detection = outlier_detection
        self.num_stages = num_stages  # Number of delayed rejection stages
        self.gammas = [1.0] + [0.5**i for i in range(1, num_stages)]  # Scaling factors for each proposal stage
        self.eps = eps
        self.samples = None
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

    def _initialize_population(self):
        """
        Initializes a population of chains using the given prior distribution.¨

        Returns:
        - initial: Initial population of points for the chains.
        """
        
        if (self.prior_means is None) or (self.prior_stds is None):
            initial = np.random.randn(self.chains, self.dimension)
        else:
            num_modes = len(self.prior_means)
            initial = np.zeros((self.chains, self.dimension))

            for chain in range(self.chains):
                mode_idx = np.random.choice(num_modes)

                initial_point = [np.random.normal(mean, std) for mean, std in zip(self.prior_means[mode_idx], self.prior_stds[mode_idx])]

                initial[chain] = initial_point

        return initial

    def _propose_point(self, current, chain, gen, CR, stage):
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
            if np.random.rand() < CR:
                if gen % 5 == 0:
                    beta = 1.0
                else:
                    beta = self.scale_factor / np.sqrt(2*num_pairs*d)

                diff = 0
                for _ in range(num_pairs):
                    indices = np.random.choice([x for x in range(self.chains) if x != chain] ,self.dimension, replace=False)
                    diff += current[indices[0]][i] - current[indices[1]][i]

                e = np.random.uniform(-self.eps, self.eps)
                eps = np.random.normal(0, self.eps)
                proposal_point[i] = current[chain][i] + ((1 + e) * (beta * diff) * self.gammas[stage]) + eps
            else:
                d -= 1

        return proposal_point

    def _acceptance_probability(self, stage_posterior):
        """
        Calculate the acceptance probability for a given posterior history.
        
        Parameters:
        - stage_posterior: Posterior history in each stage for the current chain.

        Returns:
        - alpha: Acceptance probability for the posterior history in each stage.
        """

        if self.log:
            alpha = (stage_posterior[-1] - stage_posterior[0]) if stage_posterior[0] > -np.inf else 0.0
            numerator = 0.0
            denominator = 0.0
        else:
            alpha = (stage_posterior[-1] / stage_posterior[0]) if stage_posterior[0] > 0 else 1.0
            numerator = 1.0
            denominator = 1.0

        for i in range(len(stage_posterior) - 2, 0, -1):
            if self.log:
                numerator += np.log1p(-np.exp(self._acceptance_probability(stage_posterior[i:][::-1])))
                denominator += np.log1p(-np.exp(self._acceptance_probability(stage_posterior[:len(stage_posterior) - i])))
            else:
                numerator *= (1 - self._acceptance_probability(stage_posterior[i:][::-1]))
                denominator *= (1 - self._acceptance_probability(stage_posterior[:len(stage_posterior) - i]))
        
        if self.log:
            alpha += (numerator - denominator) if denominator > -np.inf else 0.0
        else:
            alpha *= (numerator / denominator) if denominator > 0 else 1.0
        
        return min(0, alpha) if self.log else min(1, alpha)

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

    def sample(self, initial=None, N=10000):
        """
        Runs the DREAM algorithm over a specified number of generations.
        
        Parameters:
        - prior: Prior distribution to initialize the population.
        - num_generations: Number of generations to run.
        
        Returns:
        - Array of sampled points from the posterior distribution.
        """
        current = self._initialize_population() if initial is None else initial
        current_posterior = [self.posterior(u) for u in current]
        burnin_index = int(self.burnin * N)
        self.samples = np.zeros((N, self.chains, self.dimension))
        self.posterior_history = np.zeros((N + burnin_index, self.chains))
        self.acc = np.zeros((self.chains, self.num_stages, 2))

        for gen in range(N + burnin_index):
            proposal = np.zeros((self.chains, self.dimension))
            proposal_posterior = np.zeros(self.chains)

            for chain in range(self.chains):
                # Initialize stage_posterior for the current chain
                stage_posterior = [current_posterior[chain]]

                for stage in range(self.num_stages):
                    if gen >= burnin_index:
                        self.acc[chain, stage, 0] += 1
                    # Sample crossover probability CR = m / nCR
                    a = np.random.choice(np.arange(1, self.nCR + 1), p=self.p_m)
                    CR = a / self.nCR
                    self.h_a[a - 1] += 1  # Count usage of this CR value

                    # Proposal generation using differential evolution
                    proposal[chain] = self._propose_point(current, chain, gen, CR, stage)
                    proposal_posterior[chain] = self.posterior(proposal[chain])
                    stage_posterior.append(proposal_posterior[chain])

                    if self.log:
                        alpha = self._acceptance_probability(stage_posterior=stage_posterior)
                        u = np.log(np.random.rand())
                    else:
                        alpha = self._acceptance_probability(stage_posterior=stage_posterior)
                        u = np.random.rand()
                    # Accept the proposal with probability alpha
                    if u < alpha:
                        current_posterior[chain] = proposal_posterior[chain]
                        if gen >= burnin_index:
                            self.acc[chain, stage, 1] += 1

                        # Calculate the squared jumping distance ∆m
                        delta_m = np.sum(((proposal[chain] - current[chain]) / np.std(proposal, axis=0)) ** 2)
                        self.Delta_a[a - 1] += delta_m
                        break
                    else:
                        proposal[chain] = current[chain]

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

        self.acc_rate = self.acc[:, :, 1] / self.acc[:, :, 0]
        self._compute_gelman_rubin()
