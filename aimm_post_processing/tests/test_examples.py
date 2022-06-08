from os import remove
from numpy import identity
import tiled
from tiled.client import from_uri
from aimm_post_processing.operations import (
    Pull,
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
    pull_operator = Pull()
    data_dict = pull_operator(view) 
    
    def test_Pull_from_dfClient(self):    
        assert isinstance(self.data_dict["data"], pd.DataFrame)
        assert isinstance(self.data_dict["metadata"], dict)

    def test_Identity_adds_parent_uid(self):
        identity_operator = Identity()
        data_dict1 = identity_operator(self.data_dict)
        assert data_dict1["metadata"]["post_processing"]["parents"] == \
            self.data_dict["metadata"]["post_processing"]["uid"]
        
    def test_StandardizeGrid_works(self):
        standardizegrid_operator = StandardizeGrid(
            x0=15700, 
            xf=17000, 
            nx=1301,
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"]
        )
        data_dict1 = standardizegrid_operator(self.data_dict)

        data = data_dict1["data"]
        metadata = data_dict1["metadata"]
        assert isinstance(data, pd.core.frame.DataFrame)
        assert isinstance(metadata, dict)

        assert data.columns.to_list() == ['energy', 'mutrans', 'mufluor', 'murefer']
        assert len(data) == 1301
        assert metadata["post_processing"]["kwargs"]["x0"] == 15700
        assert metadata["post_processing"]["kwargs"]["xf"] == 17000
        assert metadata["post_processing"]["kwargs"]["nx"] == 1301
        assert metadata["post_processing"]["parents"] == \
            self.data_dict["metadata"]["post_processing"]["uid"]

    def test_RemoveBackground_works(self):
        
        removebackground_operator = RemoveBackground(
            x0=15700, 
            xf=15800, 
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"],
            victoreen_order=0
        )
        data_dict1 = removebackground_operator(self.data_dict)
    
        data = data_dict1["data"]
        metadata = data_dict1["metadata"]
        assert isinstance(data, pd.core.frame.DataFrame)
        assert metadata["post_processing"]["parents"] == \
            self.data_dict["metadata"]["post_processing"]["uid"]

    def test_Operator_Chains(self):
        pull_operator = Pull()
        identity_operator = Identity()
        standardizegrid_operator = StandardizeGrid(
            x0=15700, 
            xf=17000, 
            nx=1301,
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"]
        )
        removebackground_operator = RemoveBackground(
            x0=15700, 
            xf=15800, 
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"],
            victoreen_order=0
        )
        data_dict = pull_operator(self.view)
        data_dict0 = identity_operator(data_dict)
        data_dict1 = identity_operator(data_dict0)
        data_dict2 = standardizegrid_operator(data_dict1)
        data_dict3 = removebackground_operator(data_dict2)

        assert data_dict["metadata"]["post_processing"]["parents"] == \
            self.view.metadata['sample']['_id']
        assert data_dict0["metadata"]["post_processing"]["parents"] == \
            data_dict["metadata"]["post_processing"]["uid"]
        assert data_dict1["metadata"]["post_processing"]["parents"] == \
            data_dict0["metadata"]["post_processing"]["uid"]
        assert data_dict2["metadata"]["post_processing"]["parents"] == \
            data_dict1["metadata"]["post_processing"]["uid"]
        assert data_dict3["metadata"]["post_processing"]["parents"] == \
            data_dict2["metadata"]["post_processing"]["uid"]

    def test_pipeline_works(self):
        pull = Pull()
        identity = Identity()
        standardizegrid = StandardizeGrid(
            x0=15700, 
            xf=17000, 
            nx=1301,
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"]
        )
        removebackground = RemoveBackground(
            x0=15700, 
            xf=15800, 
            x_column="energy",
            y_columns=["mutrans", "mufluor", "murefer"],
            victoreen_order=0
        )
        pipeline = [
            identity, 
            standardizegrid, 
            removebackground
        ]
        
        data_dict = pull(self.view)
        for operator in pipeline:
            data_dict = operator(data_dict)


    def test_Smooth_works(self):
        
        smooth_operator = RemoveBackground()
        data_dictionary = smooth_operator(
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
    TestOperator().test_Operator_Chains()
