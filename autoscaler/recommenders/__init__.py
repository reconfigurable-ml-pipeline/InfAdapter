from .hpa import HPARecommender
from .bi_lstm.bi_lstm import BiLSTMRecommender


class _Recommenders:
    HPARecommender = HPARecommender
    BiLSTMRecommender = BiLSTMRecommender


recommenders = _Recommenders()
