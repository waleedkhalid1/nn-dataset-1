import ab.nn.util.db.Read as DB_Read
import ab.nn.util.db.Write as DB_Write
from pandas import DataFrame

def data(only_best_accuracy=False, task=None, dataset=None, metric=None, nn=None, epoch=None) -> DataFrame:
    """
    Get the NN model code and all related statistics as a pandas dataframe
    for argument description see :ref:`ab.nn.util.db.Read.data()`
    """
    dt: tuple[dict,...] = DB_Read.data(only_best_accuracy, task=task, dataset=dataset, metric=metric, nn=nn, epoch=epoch)
    return DataFrame.from_records(dt)

def save_nn(nn_code : str, task : str, dataset : str, metric : str) -> str:
    """
    Saving a new NN model and its default training configuration into database
    for argument description see :ref:`ab.nn.util.db.Write.save_nn()`
    :return: Automatically generated name of NN model.

    """
    return DB_Write.save_nn(nn_code, task, dataset, metric)
