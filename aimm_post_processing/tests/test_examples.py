from os import remove
from numpy import identity
import tiled
from tiled.client import from_uri
from aimm_post_processing.operations import (
    Identity,
    RemoveBackground, 
    StandardizeGrid,
)
import pandas as pd


class TestDataInterface:
    client = from_uri("https://aimm.lbl.gov/api")
    view = client['dataset']['newville']['uid']['cCv65Ngs86N']
    
    def test_data_interface(self):
        assert isinstance(self.view, tiled.client.dataframe.DataFrameClient)


class TestOperator:
    client = from_uri("https://aimm.lbl.gov/api")
    view = client['dataset']['newville']['uid']['cCv65Ngs86N']


    def test_Identity_extends_metadata(self):
        
        metadata = dict(self.view.metadata)
        assert not "uid" in metadata.keys()
        assert not "post_processing" in metadata.keys()
        
        identity_operator = Identity()
        data_dictionary = identity_operator(self.view)
        
        assert isinstance(data_dictionary, dict)
        data = data_dictionary["data"]
        metadata = data_dictionary["metadata"]
        assert isinstance(data, pd.core.frame.DataFrame)
        assert isinstance(metadata, dict)

        assert "post_processing" in metadata.keys()
        

    def test_StandardizeGrid_works(self):
        standardizegrid_operator = StandardizeGrid()
        data_dictionary = standardizegrid_operator(
            self.view, 
            x0=15700, 
            xf=17000, 
            nx=1301,
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"]
        )
        assert isinstance(data_dictionary, dict)
        data = data_dictionary["data"]
        metadata = data_dictionary["metadata"]
        assert isinstance(data, pd.core.frame.DataFrame)
        assert isinstance(metadata, dict)

        assert data.columns.to_list() == ['energy', 'mutrans', 'mufluor', 'murefer']
        assert len(data) == 1301
        assert metadata["post_processing"]["kwargs"]["x0"] == 15700
        assert metadata["post_processing"]["kwargs"]["xf"] == 17000
        assert metadata["post_processing"]["kwargs"]["nx"] == 1301

    
    def test_RemoveBackground_works(self):
        
        removebackground_operator = RemoveBackground()
        data_dictionary = removebackground_operator(
            self.view, 
            x0=15700, 
            xf=15800, 
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"],
            victoreen_order=0
        )

        assert isinstance(data_dictionary, dict)
        data = data_dictionary["data"]
        metadata = data_dictionary["metadata"]
        assert isinstance(data, pd.core.frame.DataFrame)
        assert isinstance(metadata, dict)




if __name__ == "__main__":
    # TestDataInterface().test_data_interface()
    # TestOperator().test_StandardizeGrid_works()
    TestOperator().test_RemoveBackground_works()
