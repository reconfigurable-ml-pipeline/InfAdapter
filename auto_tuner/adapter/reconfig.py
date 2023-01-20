import numpy as np
from joblib import load


class Reconfiguration:
    def __init__(
        self, model_versions: list, max_cpu: int, capacity_models_paths: dict, baseline_accuracies: dict, alpha: float
    ):
        self.__max_cpu = max_cpu
        self.__model_versions = model_versions
        self.__capacity_models = {v:load(capacity_models_paths[v]) for v in self.__model_versions}
        self.__baseline_accuracies = baseline_accuracies
        self.__alpha = alpha
    
    def regression_model(self, model_version, cpu):
        assert cpu >= 0, "cpu is a non-negative parameter"
        if cpu == 0:
            return 0
        X = np.array([cpu]).reshape(-1, 1)
        return int(self.__capacity_models[model_version].predict(X))
    
    def find_all_valid_options(self, max_cpu, rate, models=None, option=None, options=None):
        if models is None:
            models = self.__model_versions
            
        if option is None:
            option = []
        
        if options is None:
            options = []
        
        if rate <= 0:
            return option[:]
        
        for mi in range(len(models)):
            m = models[mi]
            ms = models[mi+1:]
            for c in range(2, max_cpu + 1):
                p = self.find_all_valid_options(
                    max_cpu-c, rate - self.regression_model(m, c), ms, option + [(m, c)], options
                )
                if p:
                    options.append(p)
        if models == self.__model_versions:
            return options
        
    def assign_shares_to_options_models(self, options, lmbda):
        options_with_shares = []
        for option in options:
            rate = lmbda
            with_share = []
            for model_choice in sorted(option, lambda x: x[0], reverse=True):
                # if rate <= 0:
                #     break
                model, cpu = model_choice
                capacity = min(self.regression_model(model, cpu), rate)
                share = capacity / lmbda
                rate -= capacity
                with_share.append((model, cpu, share))
            # if len(with_share) == len(option):
            #     options_with_shares.append(with_share)
            options_with_shares.append(with_share)
        return options_with_shares

    def find_best_option(self, options_with_shares, prev_option):
        # Todo: add the effect of prev_option
        max_f = -1000
        best_option = None
        for option_with_shares in options_with_shares:
            accuracy = 0
            cost = 0
            for model_choice in option_with_shares:
                model, cpu, share = model_choice
                accuracy += share * self.__baseline_accuracies[model]
                cost += cpu
            f = accuracy - self.__alpha * cost
            if f > max_f:
                max_f = f
                best_option = option_with_shares
        return best_option
    
    def reconfig(self, lmbda, current_option):
        all_options = self.find_all_valid_options(self.__max_cpu, lmbda)
        all_options_with_shares = self.assign_shares_to_options_models(all_options, lmbda)
        best_option = self.find_best_option(all_options_with_shares, current_option)
        return best_option
