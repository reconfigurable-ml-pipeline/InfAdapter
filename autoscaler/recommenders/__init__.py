from .hpa import HPARecommender


class _Recommenders:
    HPARecommender = HPARecommender


recommenders = _Recommenders()
