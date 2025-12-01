import pandas as pd
import numpy as np 

class SalesModeler: 

    def __init__(self, sales_data):
        self.sales_data = sales_data
        self.models = []

    def fit_all(self):
        x = self.sales_data.get_predictor()
