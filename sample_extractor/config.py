import os


class Config:

    STAC_HOST = os.getenv("STAC_HOST", "https://data.inpe.br/bdc/stac/v1/")