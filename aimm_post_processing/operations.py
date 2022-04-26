"""Module for housing postprocessing operations."""

from datetime import datetime
from monty.json import MSONable
import numpy as np
import pandas as pd
from aimm_post_processing import utils



class Operator(MSONable):
    """Base operator class. Tracks everything required through a combination
    of the MSONable base class and by using an additional datetime key to track
    when the operator was logged into the metadata of a new node.

    .. important::

        The __call__ method must be derived for every operator. In particular,
        this operator should take as arguments at least one other data point
        (node).
    """

    def as_dict(self):
        d = super().as_dict()
        d["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return d

    @classmethod
    def from_dict(cls, d):
        if "datetime" in d.keys():
            d.pop("datetime")
        return super().from_dict(d)

    def __call__(self):
        raise NotImplementedError


class Identity(Operator):
    """The identity operation. Does nothing. Used for testing purposes."""

    def __call__(self, x):
        """
        Parameters
        ----------
        x : tiled.client.dataframe.DataFrameClient
        """

        # Note this will throw a TypeError, as setting attributes is not
        # allowed!
        x.metadata["derived"] = {**self.as_dict(), "parents": [x.uri]}

        return x

class Standardization(Operator):
    
    def __call__(self, DataFrameClient, start, end, step):
        """
        Takes as an input `tiled.client.dataframe.DataFrameClient`, process it and
        return a dataframe and a dictionary.
        
        Parameters
        ----------
        DataFrameClient : tiled.client.dataframe.DataFrameClient
        start : float
            The lower limit of a standard energy grid.
        end : float
            The upper limit of a standard energy grid.
        step : float
            The spacing of energy grid.
        
        Returns
        -------
        dict_standard : dict
            The dictionary that contains the standardized mu.
        df_standard : Pandas.DataFrame
            The dataframe that contains the standardized mu.
        
        Notes
        -----
        1. This by default only accepts DataFrameClient that has mu data, 
        intensity data needs to be transformed to mu.
        2. So far it does not check if data contains NaN values.

        """

        df = DataFrameClient.read() # read as datafarme

        dict_raw = {
            col: df.loc[:,col].to_numpy()
            for col in ['energy'] + [s for s in df.columns[df.columns.str.contains('mu')]]
        } # Extract the energy~mu information into a dictionary
        
        dict_standard = {
            'energy': np.arange(start, end, step=step, dtype=np.float32)
        } # Initialize the dictionary that contains standardized mu

        for col in dict_raw:
            if col in ['mutrans', 'mufluor', 'murefer', 'mu']:
                grid_reduced, spec_reduced = utils.merge_duplicate_energy(
                    dict_raw['energy'], dict_raw[col]
                ) # merge duplicate measurements
                spec_standard = utils.featurize(
                    spec_reduced, grid_reduced, to_grid=dict_standard['energy'], 
                    kind='linear', fill_value='both_ends'
                ) # map spectra onto standard energy grid

                dict_standard.update({col: spec_standard})
        
        df_standard = pd.DataFrame(dict_standard)

        return (dict_standard, df_standard)


class Reduction(Operator):
    """Subtract pre-edge background by fitting the pre-edge region to a Victoreen function.
    Implement the background reduction method introduced in Chapter 5.1 of the book: 
        Fundamentals of XAFS by Matthew Newville.
    """
    def __call__(self, x):
        raise NotImplementedError