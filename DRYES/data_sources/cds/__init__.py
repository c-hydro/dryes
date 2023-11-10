# functions to be used for all CDS data sources

import cdsapi
import sys
import os

class CDSDownloader:
    def __init__(self) -> None:
        self.cds = cdsapi.Client()

    def download(self, dataset: str, request: dict, output: str) -> None:
        """
        Downloads data from the CDS API based on the request.
        dataset: the name of the dataset to download from
        request: a dictionary with the request parameters
        output: the name of the output file
        """
        # send request to the CDS withouth printing the output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.cds.retrieve(dataset, request, output)
        sys.stdout = original_stdout