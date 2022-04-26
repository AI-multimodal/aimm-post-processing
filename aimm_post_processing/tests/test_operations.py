from tiled.client import from_uri
from aimm_post_processing import operations as op


class Test_Standardization():
    client = from_uri("https://aimm.lbl.gov/api")

    def test_standardization_works(self):
        view = self.client['dataset']['newville']['uid']['cCv65Ngs86N']
        standardize = op.Standardization()
        dict_standardize, df_standardize = standardize(view, 15700, 17001, 1)
        # assert len(df_standardize) == 1300
        # To be finished

if __name__ == "__main__":
    Test_Standardization().test_standardization_works()