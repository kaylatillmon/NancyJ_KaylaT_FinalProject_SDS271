import pandas as pd
import numpy as np 

class SalesModeler: 

    def __init__(self, sales_data):
        self.sales_data = sales_data
        self.models = {}
        #self.bootstap_results = {}

    def fit_all(self):
        x = self.sales_data.get_predictor()
        outcomes_data = self.sales_data.get_outcome()
        
        for outcome in self.sales_data.outcome_variable:
            y = outcomes_data[outcome]

            slope, intercept = np.polyfit(x, y, 1)

            y_pred = slope * x + intercept

            ssr = np.sum((y - y_pred)**2) 
            sst = np.sum((y - y.mean()) ** 2)

            r2 = 1 - (ssr/sst)

            self.models[outcome] = {"slope": slope, "intercept": intercept, "R^2":r2}

    def get_model_parameters(self, outcome):
        return self.models.get(outcome)
    
    #def bootstrap_slopes():

    def summary_stats_table(self):
        rows = []

        for outcome, params in self.models.items():
            row = {"outcome": outcome, "slope": params["slope"], "intercept": params["intercept"], "R^2": params["R^2"]}

            rows.append(row)
        
        return pd.DataFrame(rows)

    



