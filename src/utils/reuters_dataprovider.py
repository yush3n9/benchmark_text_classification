from src.utils.dataset_provider_base import IterableDataProvider
from sklearn.datasets import fetch_rcv1


class ReutersTfIdfVectors(IterableDataProvider):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def inspect_dataset(self):
        pass

    def fetch_dataset_train(self):
        return fetch_rcv1(subset='train', random_state=self.random_state)

    def fetch_dataset_test(self):
        return fetch_rcv1(subset='test', random_state=self.random_state)
