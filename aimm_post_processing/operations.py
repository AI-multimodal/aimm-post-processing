"""Module for housing post-processing operations."""

from abc import ABC
from datetime import datetime
from uuid import uuid4

from monty.json import MSONable
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from aimm_post_processing import utils
from copy import deepcopy
from abc import ABC, abstractmethod

class Operator(MSONable, ABC):
    """Base operator class. Tracks everything required through a combination
    of the MSONable base class and by using an additional datetime key to track
    when the operator was logged into the metadata of a new node.

    .. important::

        The __call__ method must be derived for every operator. In particular,
        this operator should take as arguments at least one other data point
        (node).
    """

    def __init__( # Initialize common parameters for all operators
        self,
        x_column="energy",
        y_columns=["mu"],
    ):
        self.x_column = x_column
        self.y_columns = y_columns
        self.operator_id = str(uuid4()) # UID for the defined operator.


    def __call__(self, dataDict):
        # meke a copy, otherwise python will make modification to input dataDict instead.
        copy_dataDict = deepcopy(dataDict)

        new_metadata = self._process_metadata(copy_dataDict["metadata"])
        new_df = self._process_data(copy_dataDict["data"])

        return {"data": new_df, "metadata": new_metadata}

    def _process_metadata(self, metadata):
        """Preliminary pre-processing of the dictionary object that contains data and metadata. 
        Takes the:class:`dict` object as input and returns the untouched data in addition to an 
        augmented metadata dictionary.
        
        Parameters
        ---------
        dataDict : dict
            The data dictionary that contains data and metadata
        local_kwargs : tuple
            A tuple (usually `locals().items()` where it is called) of local variables in operator
        space.

        Notes
        -----
        1. Measurement `id` is suspiciously `_id` that sits under `sample` in the metadata.
        Need to check if this hierchy is universal for all data.
        """

        # parents are the uid of the last processed data, or the original sample id otherwise.
        try: 
            parent_id = metadata["post_processing"]["id"]
        except: 
            parent_id = metadata['_tiled']['uid']

        dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        metadata["post_processing"] = {
            "id": str(uuid4()),
            "parent": parent_id,
            "operator": self.as_dict(),
            "kwargs": self.__dict__,
            "datetime": f"{dt} UTC",
        }

        return metadata
    
    @abstractmethod
    def _process_data(self, df) -> dict:
        """User must override this method.
        """
        raise NotImplementedError


class Pull(Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, dfClient):
        """ This operator does nothing but return the data/metadata dictionary for a given 
        `tiled.client.dataframe.DataFrameClient`.
        """
        metadata = deepcopy(dict(dfClient.metadata))
        new_metadata = self._process_metadata(metadata)
        df = deepcopy(dfClient.read())
        new_df = self._process_data(df)

        return {"data": new_df, "metadata": new_metadata}

    def _process_data(self, df):
        return df


class Identity(Operator):
    """The identity operation. Does nothing. Used for testing purposes."""

    def __init__(self):
        super().__init__()

    def _process_data(self, df):
        """
        Parameters
        ----------
        dataDict : pandas.DataFrame
            The data is a :class:`pd.DataFrame`

        Returns
        -------
        dataDict : dict
            Same as input
        """
        return df


class StandardizeGrid(Operator):
    """Interpolates specified columns onto a common grid."""

    def __init__(
        self,
        x0,
        xf,
        nx,
        interpolated_univariate_spline_kwargs=dict(),
        x_column="energy",
        y_columns=["mu"]
    ):
        """Interpolates the provided DataFrameClient onto a grid as specified
        by the provided parameters.

        Parameters
        ----------
        x0 : float
            The lower bound of the grid to interpolate onto.
        xf : float
            The upper bound of the grid to interpolate onto.
        nx : int
            The number of interpolation points.
        interpolated_univariate_spline_kwargs : TYPE, optional
            Keyword arguments to be passed to the
            :class:`InterpolatedUnivariateSpline` class. See
            [here](https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.interpolate.InterpolatedUnivariateSpline.html) for the
            documentation on this class. Default is {}.
        x_column : str, optional
            References a single column in the DataFrameClient (this is the
            "x-axis"). Default is "energy".
        y_columns : list, optional
            References a list of columns in the DataFrameClient (these are the
            "y-axes"). Default is ["mu"].
        
        Returns:
        -------
        An instance of StandardGrid operator.
        """
        super().__init__(x_column, y_columns)
        self.x0 = x0
        self.xf = xf
        self.nx = nx
        self.interpolated_univariate_spline_kwargs = interpolated_univariate_spline_kwargs
        
    def _process_data(self, df):

        """Takes in a dictionary of the data amd metadata. The data is a
        :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        Returns the same dictionary with processed data and metadata.
        """

        new_grid = np.linspace(self.x0, self.xf, self.nx)
        new_data = {self.x_column: new_grid}
        for column in self.y_columns:
            ius = InterpolatedUnivariateSpline(
                df[self.x_column],
                df[column],
                **self.interpolated_univariate_spline_kwargs,
            )
            new_data[column] = ius(new_grid)

        return pd.DataFrame(new_data)


class RemoveBackground(Operator):
    """Fit the pre-edge region to a victoreen function and subtract it from the spectrum.
    """
    def __init__(
        self,
        *,
        x0,
        xf,
        x_column="energy",
        y_columns=["mu"],
        victoreen_order=0
    ):
        """Subtract background from data.
        Fit the pre-edge data to a line with slope, and subtract slope info from data.

        Parameters
        ----------
        dfClient : tiled.client.dataframe.DataFrameClient
        x0 : float
            The lower bound of energy range on which the background is fitted.
        xf : flaot
            The upper bound of energy range on which the background is fitted.
        x_column : str, optional
            References a single column in the DataFrameClient (this is the
            "x-axis"). Default is "energy".
        y_columns : list, optional
            References a list of columns in the DataFrameClient (these are the
            "y-axes"). Default is ["mu"].
        victoreen_order : int
            The order of Victoreen function. The selected data is fitted to Victoreen pre-edge 
            function (in which one fits a line to μ(E)*E^n for some value of n. Default is 0,
            which is a linear fit.
        
        Returns
        -------
        An instance of RemoveBackground operator
        """
        super().__init__(x_column, y_columns)
        self.x0 = x0
        self.xf = xf
        self.victoreen_order = victoreen_order

    def _process_data(self, df):
        """
        Takes in a dictionary of the data amd metadata. The data is a
        :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        Returns the same dictionary with processed data and metadata.
        
        Notes
        -----
        `LinearRegression().fit()` takes 2-D arrays as input. This can be explored
        for batch processing of multiple spectra
        """

        bg_data = df.loc[(df[self.x_column] >= self.x0) * (df[self.x_column] < self.xf)]

        new_data = {self.x_column: df[self.x_column]}
        for column in self.y_columns:
            y = bg_data[column] * bg_data[self.x_column]**self.victoreen_order
            reg = LinearRegression().fit(
                bg_data[self.x_column].to_numpy().reshape(-1,1), 
                y.to_numpy().reshape(-1,1)
            )
            background = reg.predict(df[self.x_column].to_numpy().reshape(-1,1))
            new_data[column] = df.loc[:,column].to_numpy() - background.flatten()

        return pd.DataFrame(new_data)


# class Normalize(Operator):
#     """
#     """
#     def __init__(
#         self,
#         x_column="energy",
#         y_columns=["mu"]
#     ):
#         super().__init__(x_column, y_columns)

#     def _process_data(self, df):
#         xas_ds = XASDataSet(name="Shift XANES", energy=grid, mu=dd) 
#         xas_ds.norm1 = norm1 # update atribute for force_normalization
#         xas_ds.normalize_force() # force the normalization again with updated atribute        


class StandardizeIntensity(Operator):
    """ Scale the intensity so they vary in similar range.
    """
    def __init__(
        self,
        *,
        x0 = None,
        xf = None,
        x_column="energy",
        y_columns=["mu"]
    ):
        """Align the intensity to the mean of a selected range, and scale the intensity up to standard
        deviation.

        Parameters
        ----------
        dfClient : tiled.client.dataframe.DataFrameClient
        x0 : float
            The lower bound of energy range for which the mean is calculated. If None, the first 
            point in the energy grid is used. Default is None.
        yf : float
            The upper bound of energy range for which the mean is calculated. If None, the last 
            point in the energy grid is used. Default is None.
        x_column : str, optional
                References a single column in the DataFrameClient (this is the
                "x-axis"). Default is "energy".
        y_columns : list, optional
            References a list of columns in the DataFrameClient (these are the
            "y-axes"). Default is ["mu"].
        
        Returns
        -------
        An instance of StandardizeIntensity operator
        """
        super().__init__(x_column, y_columns)
        self.x0 = x0
        self.xf = xf

    def _process_data(self, df):
        """
        Takes in a dictionary of the data amd metadata. The data is a
        :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        Returns the same dictionary with processed data and metadata.

        """

        grid = df.loc[:, self.x_column]
        if self.x0 is None: self.x0 = grid[0]
        if self.xf is None: self.xf = grid[-1]
        assert self.x0 < self.xf, "Invalid range, make sure x0 < xf"
        select_mean_range = (grid > self.x0) & (grid < self.xf)
        
        new_data = {self.x_column: df[self.x_column]}
        for column in self.y_columns:
            mu = df.loc[:, column]
            mu_mean = mu[select_mean_range].mean()
            mu_std = mu.std()
            new_data.update({column: (mu-mu_mean)/mu_std})
        
        return pd.DataFrame(new_data)


class Smooth(Operator):
    """Return the simple moving average of spectra with a rolling window.
        Parameters
        ----------
        
        window : float, in eV.
            The rolling window in eV over which the average intensity is taken.
        x_column : str, optional
                References a single column in the DataFrameClient (this is the
                "x-axis"). Default is "energy".
        y_columns : list, optional
            References a list of columns in the DataFrameClient (these are the
            "y-axes"). Default is ["mu"].
    """
    def __init__(
        self,
        *,
        window=10,
        x_column='energy',
        y_columns=['mu']
    ):
        super().__init__(x_column, y_columns)
        self.window = window

    def _apply(self, df):
        """
        Takes in a dictionary of the data amd metadata. The data is a
        :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        Returns the same dictionary with processed data and metadata.
        
        Returns:
        --------
        dict
            A dictionary of the data and metadata. The data is a :class:`pd.DataFrame`, 
            and the metadata is itself a dictionary.
        """
        
        grid = df.loc[:,self.x_column]
        new_data = {self.x_column: df[self.x_column]}
        for column in self.y_columns:
            y = df.loc[:, column]
            y_smooth = utils.simple_moving_average(grid, y, window=self.window)
            new_data.update({column: y_smooth})
            mse = mean_squared_error(y, y_smooth)
            n2s = mse / y_smooth.std()

        return pd.DataFrame(new_data)


class Classify(Operator):
    """ Label the spectrum as "good", "noisy" or "discard" based on the quality of the spectrum.
    """
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def _process_data(self, df):
        """
        Parameters
        ----------
        dfClient : tiled.client.dataframe.DataFrameClient
        classifier : Callable
            The classifier that takes in the spectrum and output a label.

        """
        return df



class PreNormalize(Operator):
    """
    """