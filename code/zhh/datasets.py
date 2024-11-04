import jax.numpy as jnp
import jax.random as jr
from zhh.debug import get_mode
from zhh.transforms import Transform, ToTensor

class Dataset:
    """
    A general dataset class. You should implement your dataset class by inheriting this class.

    ### Methods:

    `__len__`: return the length of the dataset

    `__getitem__`: given a **sequence of indices** `indices`, return the corresponding elements.
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, indices):
        raise NotImplementedError

class LabeledImageDataset(Dataset):
    """
    A TFDS dataset.

    Args:
        dataset: a dictionary containing `image` and `label` keys.
        transforms: a zhh.transforms.Transform to apply to the data, default is just `ToTensor()`

    `__getitem__` returns a tuple of (x, y) given a **sequence of indices** , where x is the transformed image and y is the label.
    """

    def __init__(self,dataset, transforms:Transform = None):
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        if self.transforms is not None:
            return self.transforms(self.dataset['image'][idx]), self.dataset['label'][idx]
        return self.dataset['image'][idx], self.dataset['label'][idx]

class DataLoader:
    """
    A DataLoader class which is similar to PyTorch DataLoader.

    Note that by default, the dataset will be **truncated** to make it a multiple of `batch_size`. This isn't our specific choice, it is due to TPU's inefficiency in handling the last batch (since the shape is not fixed).

    Args:
        dataset: a Dataset object.
        batch_size: the batch size.
        shuffle: whether to shuffle the dataset.

    >>> dataset = jnp.zeros((128, 1, 28, 28))
    >>> dl = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for x in dl:
    ...     print(x.shape) # (32, 1, 28, 28)
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        dataset_len = len(dataset)

        # we truncate the dataset to make it a multiple of batch_size
        self.length = dataset_len // batch_size
        self.shuffle = shuffle
        self.key = jr.PRNGKey(0)
        
    def _shuffle(self):
        if (not self.shuffle) or get_mode():
            if get_mode():
                print('[WARNING] In debug mode, shuffling is disabled.')
            self.perm = jnp.arange(self.length * self.batch_size).reshape((self.length, self.batch_size))
        else:
            print('Shuffling the dataset, this may take a while...')
            self.key, use = jr.split(self.key)
            self.perm = jr.permutation(use, self.length * self.batch_size).reshape((self.length, self.batch_size))
            print('Shuffling done!')

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(self.length):
            if i == 0:
                self._shuffle()
            yield self.dataset[self.perm[i]]

class MNIST:
    """
    MNIST data class implemented by TFDS (Tensorflow Datasets).

    Args:
        train_transform: a zhh.transforms.Transform to apply to the training data, default is just `ToTensor()`
        valid_transform: a zhh.transforms.Transform to apply to the validation data, default is just `ToTensor()`

    >>> mnist = MNIST()
    >>> train_ds = mnist.train_ds
    >>> test_ds = mnist.test_ds
    >>> print(len(train_ds), len(test_ds)) # 60000 10000
    >>> x,y = train_ds[0]
    >>> print(x.shape, y) # (28, 28, 1) 4
    >>> print(x.min(), x.max()) # 0.0 0.99999994
    """

    def __init__(self, train_transform:Transform = None, valid_transform:Transform=None):
        import tensorflow_datasets as tfds
        ds_builder = tfds.builder('mnist')
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
        if train_transform is None:
            train_transform = ToTensor()
        if valid_transform is None:
            valid_transform = ToTensor()
        self.train_ds = LabeledImageDataset(train_ds, transforms=train_transform)
        self.test_ds = LabeledImageDataset(test_ds, transforms= valid_transform)

class CIFAR10:
    """
    CIFAR10 data class implemented by TFDS (Tensorflow Datasets).

    Args:
        train_transform: a zhh.transforms.Transform to apply to the training data, default is just `ToTensor()`
        valid_transform: a zhh.transforms.Transform to apply to the validation data, default is just `ToTensor()`

    >>> cifar10 = CIFAR10()
    >>> train_ds = cifar10.train_ds
    >>> test_ds = cifar10.test_ds
    >>> print(len(train_ds), len(test_ds)) # 50000 10000
    >>> x,y = train_ds[0]
    >>> print(x.shape, y) # (32, 32, 3) 7
    >>> print(x.min(), x.max()) # 0.0 0.97647053
    """

    def __init__(self, train_transform:Transform = None, valid_transform:Transform=None):
        import tensorflow_datasets as tfds
        ds_builder = tfds.builder('cifar10')
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
        if train_transform is None:
            train_transform = ToTensor()
        if valid_transform is None:
            valid_transform = ToTensor()
        self.train_ds = LabeledImageDataset(train_ds, transforms=train_transform)
        self.test_ds = LabeledImageDataset(test_ds, transforms=valid_transform)
