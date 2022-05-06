"""Module for housing post-processing operations."""

from datetime import datetime
from uuid import uuid4

from monty.json import MSONable
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.linear_model import LinearRegression



class Operator(MSONable):
    """Base operator class. Tracks everything required through a combination
    of the MSONable base class and by using an additional datetime key to track
    when the operator was logged into the metadata of a new node.

    .. important::

        The __call__ method must be derived for every operator. In particular,
        this operator should take as arguments at least one other data point
        (node).
    """

    def __call__(self):
        raise NotImplementedError

    def _preprocess_DataFrameClient(self, x):
        """Preliminary pre-processing of the DataFrameClient object. Takes the
        :class:`tiled.client.dataframe.DataFrameClient` object as input and
        returns the read data in addition to an augmented metadata dictionary.
        """

        data = x.read()
        metadata = dict(x.metadata)
        dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        metadata["uid"] = str(uuid4())
        metadata["post_processing"] = {
            "operator": self.as_dict(),
            "kwargs": dict(),
            "parents": [x.uri],
            "datetime": f"{dt} UTC",
        }

        return data, metadata


class Identity(Operator):
    """The identity operation. Does nothing. Used for testing purposes."""

    def __call__(self, dfClient):
        """
        Parameters
        ----------
        dfClient : tiled.client.dataframe.DataFrameClient

        Returns
        -------
        dict
            A dictionary of the data and metadata. The data is a
            :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        """

        data, metadata = self._preprocess_DataFrameClient(dfClient)
        return {"data": data, "metadata": metadata}


class StandardizeGrid(Operator):
    """Interpolates specified columns onto a common grid."""

    def __call__(
        self,
        dfClient,
        *,
        x0,
        xf,
        nx,
        interpolated_univariate_spline_kwargs=dict(),
        x_column="energy",
        y_columns=["mu"],
    ):
        """Interpolates the provided DataFrameClient onto a grid as specified
        by the provided parameters.

        Parameters
        ----------
        dfClient : tiled.client.dataframe.DataFrameClient
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

        Returns
        -------
        dict
            A dictionary of the data and metadata. The data is a
            :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        """

        kwargs = {
            key: value
            for key, value in locals().items()
            if key not in ["self", "dfClient"]
        }

        data, metadata = self._preprocess_DataFrameClient(dfClient)
        metadata["post_processing"]["kwargs"] = kwargs

        new_grid = np.linspace(x0, xf, nx)
        new_data = {x_column: new_grid}
        for column in y_columns:
            ius = InterpolatedUnivariateSpline(
                data[x_column],
                data[column],
                **interpolated_univariate_spline_kwargs,
            )
            new_data[column] = ius(new_grid)

        return {"data": pd.DataFrame(new_data), "metadata": metadata}


class RemoveBackground(Operator):
    """Fit the pre-edge region to a victoreen function and subtract it from the spectrum.
    """
    def __call__(
        self,
        dfClient,
        *,
        x0,
        xf, 
        x_column="energy",
        y_columns=["mu"],
        victoreen_order = 0
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
        dict
            A dictionary of the data and metadata. The data is a
            :class:`pd.DataFrame`, and the metadata is itself a dictionary.
        
        Notes
        -----
        `LinearRegression().fit()` takes 2-D arrays as in put. This can be explored
        for batch processing of multiple spectra

        """
        kwargs = {
            key: value
            for key, value in locals().items()
            if key not in ["self", "dfClient"]
        }

        data, metadata = self._preprocess_DataFrameClient(dfClient)
        metadata["post_processing"]["kwargs"] = kwargs

        bg_data = data.loc[(data[x_column] >= x0) * (data[x_column] < xf)]

        new_data = {x_column: data[x_column]}
        for column in y_columns:
            y = bg_data[column] * bg_data[x_column]**victoreen_order
            reg = LinearRegression().fit(
                bg_data[x_column].to_numpy().reshape(-1,1), 
                y.to_numpy().reshape(-1,1)
            )
            background = reg.predict(data[x_column].to_numpy().reshape(-1,1))
            new_data[column] = data.loc[:,column].to_numpy() - background.flatten()

        return {"data": pd.DataFrame(new_data), "metadata": metadata}