import ab.nn.util.db.Read as DB_Read
from pandas import DataFrame

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas dataframe
    for args see :ref:`ab.nn.util.db.Read.data()`
    """
    dt: tuple[dict,...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch)
    return DataFrame.from_records(dt)
