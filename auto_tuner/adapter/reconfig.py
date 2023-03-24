import os
# import numpy as np
from joblib import load
import gurobipy as gp
import pandas as pd


class Reconfiguration:
    def __init__(
        self,
        model_versions: list,
        max_cpu: int,
        capacity_models_paths: dict,
        baseline_accuracies: dict,
        load_times: dict,
        alpha: float,
        beta = 0.2,
    ):
        self.__max_cpu = max_cpu
        self.__model_versions = sorted(model_versions)
        capacity_models = {v: load(capacity_models_paths[v]) for v in self.__model_versions}
        self.__baseline_accuracies = baseline_accuracies
        self.__load_times = load_times
        self.__alpha = alpha
        self.__beta = beta
        self.__min_cpu = 2
        self.__cap_coef = float(os.environ["CAPACITY_COEF"])
        self.x = 0
        self.__regression_coefficients = []
        for i in range(len(self.__model_versions)):
            reg = capacity_models[self.__model_versions[i]]
            coef1 = pd.DataFrame(
                reg.coef_, ["coef"], columns=['Coefficients']
            )["Coefficients"][0]
            example_cpu = 4
            p1 = coef1 * example_cpu
            actual = float(reg.predict([[example_cpu]]))
            coef2 = actual - p1
            coef2 = min(coef2, 0.99)
            coef2 = int(coef2)
            self.__regression_coefficients.append([coef1, coef2])
        
        self.__options = []  # global list for the recursive generator function
        
        # # Caching regresison predictions
        # self.__capacities = {v: {} for v in self.__model_versions}
        # for c in range(2, self.__max_cpu + 1):
        #     X = np.array([c]).reshape(-1, 1)
        #     for m in self.__model_versions:
        #         self.__capacities[m][c] = int(self.__cap_coef * capacity_models[m].predict(X))
    
    
    def regression_model(self, model_version, cpu):
        assert cpu >= 0, "cpu is a non-negative parameter"
        if cpu == 0:
            return 0
        return int(self.regression_model_gurobi(self.__model_versions.index(model_version), cpu))
    
    def regression_model_gurobi(self, model_idx, cpu):
        coef1, coef2 = self.__regression_coefficients[model_idx]
        return self.__cap_coef * (coef1 * cpu + coef2)
    
    
    def find_all_valid_options(self, idx, max_cpu, rate, option=None):
        if option is None:
            self.x = 0
            option = []
        
        if rate <= 0:
            return self.__options.append(option)
        
        if idx == len(self.__model_versions):
            return
        
        self.x += 1
        m = self.__model_versions[idx]
        
        self.find_all_valid_options(
            idx+1, max_cpu, rate, option
        )
        
        for c in range(self.__min_cpu, max_cpu + 1):
            self.find_all_valid_options(
                idx+1, max_cpu-c, rate - self.regression_model(m, c), option + [(m, c)]
            )
        
    def assign_shares_to_options_models(self, options, lmbda):
        options_with_shares = []
        for option in options:
            rate = lmbda
            with_share = []
            for model_choice in sorted(option, key=lambda x: x[0], reverse=True):
                model, cpu = model_choice
                capacity = min(self.regression_model(model, cpu), rate)
                share = capacity / lmbda
                rate -= capacity
                with_share.append((model, cpu, share))
            options_with_shares.append(with_share)
        return options_with_shares

    @staticmethod
    def convert_config_to_dict(config):
        d = {}
        for ms in config:
            m, c, _ = ms
            d[m] = c
        return d
    
    def transition_cost(self, model, current_option, new_option):
        current = self.convert_config_to_dict(current_option)
        new = self.convert_config_to_dict(new_option)
        if model not in new.keys():
            return 0
        if current.get(model) != new.get(model):
            return 1
        return 0

    def load_cost(self, current_option, new_option):
        lc = {v: self.transition_cost(v, current_option, new_option) * self.__load_times[v] for v in self.__model_versions}
        return max(lc.values()) / max(self.__load_times.values())
                

    def find_best_option(self, options_with_shares, prev_option):
        # Todo: add the effect of prev_option
        max_f = -100
        best_option = None
        for option_with_shares in options_with_shares:
            accuracy = 0
            cost = 0
            for model_choice in option_with_shares:
                model, cpu, share = model_choice
                accuracy += share * self.__baseline_accuracies[model]
                cost += cpu
            load_cost = self.load_cost(prev_option, option_with_shares)
            resource_cost = cost / self.__max_cpu
            f = accuracy - (self.__alpha * resource_cost + self.__beta * load_cost)
            if f > max_f:
                max_f = f
                best_option = option_with_shares
        return best_option
    
    def reconfig(self, lmbda, current_option):
        max_cpu = self.__max_cpu
        # mamv = sorted(self.__model_versions)[-1]
        # for cpu in range(self.__min_cpu, max_cpu + 1):
        #     if self.regression_model(mamv, cpu) >= lmbda:
        #         max_cpu = cpu
        #         break
        use_brute_force = False
        if use_brute_force:
            self.find_all_valid_options(0, max_cpu, lmbda)
            all_options_with_shares = self.assign_shares_to_options_models(self.__options, lmbda)
            best_option = self.find_best_option(all_options_with_shares, current_option)
            self.__options = []
        else:
            best_option = self.run_optimizer(lmbda, current_option)
        return best_option
    
    def reconfig_msp(self, lmbda, current_option):
        max_cpu = self.__max_cpu
        mamv = sorted(self.__model_versions)[-1]
        for cpu in range(self.__min_cpu, max_cpu + 1):
            if self.regression_model(mamv, cpu) >= lmbda:
                max_cpu = cpu
                break
        all_options = []
        for mv in self.__model_versions:
            for c in range(2, max_cpu + 1):
                if self.regression_model(mv, c) >= lmbda:
                    all_options.append([(mv, c, 1)])
                    break
        best_options = self.find_best_option(all_options, current_option)
        return best_options
    
    def run_optimizer(self, lmbda, current_option):
        current = self.convert_config_to_dict(current_option)
        prev_state = []
        for v in self.__model_versions:
            prev_state.append(current.get(v, 0))
        num_ms = len(self.__model_versions)
        gp.setParam("LogToConsole", 0)
        gp.setParam('OutputFlag', 0)
        gp.setParam('Threads', 1)
        model = gp.Model("Inference")
        
        n = model.addVars(num_ms, lb=0, ub=self.__max_cpu, vtype=gp.GRB.INTEGER, name="n")
        reg = model.addVars(num_ms, lb=-10, vtype=gp.GRB.CONTINUOUS, name="reg")
        tp = model.addVars(num_ms, lb=0, vtype=gp.GRB.INTEGER, name="tp")
        tp_float = model.addVars(num_ms, lb=0, vtype=gp.GRB.CONTINUOUS, name="tp_float")
        rl = model.addVars(num_ms, lb=0, vtype=gp.GRB.INTEGER, name="rl")
        l = model.addVars(num_ms, lb=0, vtype=gp.GRB.INTEGER, name="l")
        LC = model.addVar(name='LC', lb=0, vtype=gp.GRB.CONTINUOUS)
        
        forObj = model.addVars(num_ms, vtype = gp.GRB.CONTINUOUS)
        check_v = model.addVars(num_ms, vtype=gp.GRB.BINARY, name='check_v')
        vIs0 = model.addVars(num_ms, vtype=gp.GRB.BINARY, name='vIs0')
        vIsP = model.addVars(num_ms, vtype=gp.GRB.BINARY, name='vIsP')
        # if check_v is 0 then n=0 or n=n' (this implies if n!=0 and n!=n' then check_v=1)
        model.addConstrs((vIs0[i] == 1) >> (n[i] == 0) for i in range(num_ms))
        model.addConstrs((vIsP[i] == 1) >> (n[i] == prev_state[i]) for i in range(num_ms))
        model.addConstrs(check_v[i] == gp.or_(vIs0[i], vIsP[i]) for i in range(num_ms))
        model.addConstrs(forObj[i] == (1-check_v[i]) * list(self.__load_times.values())[i] for i in range(num_ms))
        model.addConstr(LC == gp.max_(forObj[i] for i in range(num_ms)))
        model.addConstrs((vIs0[i] == 0) >> ((n[i] - 1) >= 1) for i in range(num_ms))
        model.addConstrs((reg[i] == self.regression_model_gurobi(i, n[i]) for i in range(num_ms)), name="regAssign")
        model.addConstrs(tp_float[i] == gp.max_(reg[i], 0) for i in range(num_ms))
        model.addConstrs(tp[i] <= tp_float[i] for i in range(num_ms))
        model.addConstrs(tp[i] >= tp_float[i] - 0.999999 for i in range(num_ms))
        model.addConstrs(
            rl[i] == lmbda - (gp.quicksum(tp[j] for j in range(i+1, num_ms)))
            for i in range(num_ms)
        )
        model.addConstrs(l[i] == gp.min_(tp[i], rl[i]) for i in range(num_ms))
        RC = gp.quicksum(n[i] for i in range(num_ms))
        AA = gp.quicksum((l[i] / lmbda) * list(self.__baseline_accuracies.values())[i] for i in range(num_ms))
        model.addConstr(RC <= self.__max_cpu)       
        model.addConstr(gp.quicksum(tp[i] for i in range(num_ms)) >= lmbda)
        model.setObjective(
            AA - 
            self.__alpha * (RC / self.__max_cpu) -
            self.__beta * (LC / max(self.__load_times.values())),
            gp.GRB.MAXIMIZE
        )
        
        model.optimize()
        # model.computeIIS()
        # model.write("test.ilp")
        n_list, l_list = [], []
        for v in model.getVars():
            if v.varName in [f'n[{i}]' for i in range(num_ms)]:
                n_list.append(int(v.x))
            elif v.varName in [f'l[{i}]' for i in range(num_ms)]:
                l_list.append(int(v.x))
            
            # print(v.varName, v.x)
        
        new_config = []
        for i in range(num_ms):
            v = self.__model_versions[i]
            if n_list[i] > 0:
                new_config.append((v, n_list[i], l_list[i] / lmbda))
        return new_config
