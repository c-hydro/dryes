# functions to be used for all CDS data sources
import cdsapi
from .. import DRYESDataSource

class CDSDownloader(DRYESDataSource):
    def __init__(self) -> None:
        super().__init__()
        self.cds = cdsapi.Client(progress=False)#, quiet=True)

    def download(self, dataset: str, request: dict, output: str) -> None:
        """
        Downloads data from the CDS API based on the request.
        dataset: the name of the dataset to download from
        request: a dictionary with the request parameters
        output: the name of the output file
        """
        # send request to the CDS withouth printing the output
        self.cds.retrieve(dataset, request, output)