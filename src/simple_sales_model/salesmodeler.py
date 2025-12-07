import pandas as pd
import numpy as np 

class SalesModeler: 
    """
    SalesModeler fits and summarizes simple linear regression models for all outcome variables using the same predictor variable.

    Attributes:
        sales_data (object): Contains predictor and outcome variables.
        models (dict): Dictionary with the keys being the outcome variable name, and each value being seperate dictionaries containing regression results.
    """
    def __init__(self, sales_data):
        """
        This function sets up the storage for the class SalesModeler.

        Parameters:
            sales_data (object): The stored provided data that will be analyzed. 

        Attributes:
            sales_data (object): Stores imputed sales data. 
            models (dict): Initialized as an empty dictionary, intended to store outcome regression models by each outcome variable name.
        """
        self.sales_data = sales_data
        self.models = {}
        #self.bootstap_results = {}

    def fit_all(self):
        """
        This function fits simple linear regression models for every outcome variable, all using the same predictor variable. 

        Attributes:
            models (dict): A dictionary storing the calculated regression models. Each key is a variable name with each value being another dictionary containing:
                'slope' (float): Calculated slope of the regression line.
                'intercept' (float): Calculated intercept of the regression line.
                'R^2' (float): Calculated "goodness" of fit.
        """
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
        """
        A getter, the function returns stored regression model information for the imputed outcome variable name.

        Parameters:
            outcome (str): Name of outcome variable that will have their information retrieved and returned.

        Returns:
            self.models (dict): Assuming the imputed outcome variable exists, the function will return the dictionary containing the outcome variables information. If the outcome variable does not exist in self.model, the function returns None.
        """
        return self.models.get(outcome)
    
    #def bootstrap_slopes():

    def summary_stats_table(self):
        """
        This function compiles a summary statistics table for all outcome variable fitted models. 

        Returns:
            rows (pd.DataFrame): The returned dataframe with each row corresponding to an outcome variable. Each row contains:
                'outcome' (str): Name of outcome variable.
                'slope' (float): Calculated slope of the regression line.
                'intercept' (float): Calculated intercept of the regression line.
                'R^2' (float): Calculated "goodness" of fit.
        """
        rows = []

        for outcome, params in self.models.items():
            row = {"outcome": outcome, "slope": params["slope"], "intercept": params["intercept"], "R^2": params["R^2"]}

            rows.append(row)
        
        return pd.DataFrame(rows)

    



