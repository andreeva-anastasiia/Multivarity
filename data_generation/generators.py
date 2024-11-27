"""Augmentation Methods"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# TODO Define Data Augementation Methods that inherit 

class DataGenerator(ABC):
    """Base Class for Data Augmentation and Synthesis Methods"""
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, data: pd.DataFrame)->pd.DataFrame:
        """
        Generates new data from the input data 
        Note: data can be None for Synthesis Methods 
        Args:
            data: Input Data (for Augmentation Methods) or None (for Synthesis Methods)
        Returns: A Pandas DataFrame with the generated data
        """
        pass

    def __str__(self):
        return self.__class__.__name__ 
    


# Example
class TSMixup(DataGenerator):
    """Time Series Mixup Data Augmentation Method"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: pd.DataFrame)->pd.DataFrame:
        # TODO Generate Data
        raise NotImplementedError("Method not implemented")


# ! Recommendation:
# ! Start with simple methods like Random Noise for univariate Time Series
# ! Then introduce correlation via Cholesky Decomposition 
# ! Then go over to evaluating those methods first (To get the evaluation Pipeline up and running)

# TODO Random Noise
class RandomNoise(DataGenerator):
    def __init__(self, noise_level=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:

        # Ensure data is in DataFrame or NumPy array format
        if isinstance(data, pd.DataFrame):
            data = data.values  # DataFrame to NumPy array
        elif not isinstance(data, np.ndarray):
            raise ValueError(f"Expected data to be a Pandas DataFrame or NumPy array, got {type(data)}")

        noisy_data = data + np.random.normal(0, self.noise_level, data.shape)
        # print(noisy_data[0][0].tolist())
        # print(noisy_data)
        return pd.DataFrame(noisy_data[0][0])  # Return as DataFrame


# TODO Transformation Methods
    # TODO Jittering ~ Noise Injection
    # TODO Rotation
    # TODO Magnitude Warping
    # TODO Time Warping
    # TODO Window Slicing
    # TODO Frequency Warping

# TODO Define Recombination Methods
    # TODO Permutation

# TODO Define Pattern Mixing Methods
    # TODO TSMix: Time Series Mixup https://arxiv.org/html/2403.07815v1

# TODO Generative models
    # TODO KernelsSynth https://arxiv.org/html/2403.07815v1
    # TODO TimeGAN (The GAN we spoke about)
        # TODO TrendGan ~ Only Generates Trends
        # TODO SeasonGan ~ Only Generates Seasons
        # TODO LocalGan ~ Trained for regenerating a single TS

# TODO Cholesky Decomposition (Here the question is should we make it an addition to other methods or a standalone method)
# ...


