
class Loader:
    @staticmethod
    def load_dataset(loader_path, transform_path=None, **kwargs):
        """
        Dynamically load dataset and transformation based on the provided paths.
        :param loader_path: Path to the dataset loader (e.g., 'ab.loader.cifar10.loader').
        :param transform_path: Path to the dataset transformation (e.g., 'ab.transform.cifar10_norm.transform').
        :param kwargs: Additional parameters for the loader and transform.
        :return: Train and test datasets.
        """
        # Dynamically load the transform function if provided
        transform = None
        if transform_path:
            transform_module, transform_func = transform_path.rsplit('.', 1)
            transform = getattr(__import__(transform_module, fromlist=[transform_func]), transform_func)(**kwargs)

        # Dynamically load the loader function
        loader_module, loader_func = loader_path.rsplit('.', 1)
        loader = getattr(__import__(loader_module, fromlist=[loader_func]), loader_func)
        # Call the loader function with the dynamically loaded transform
        return loader(transform=transform, **kwargs)
