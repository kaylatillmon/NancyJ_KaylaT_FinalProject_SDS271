import pandas as pd

class SalesData:
    """A class used for validating and storing Sales data"""

    def __init__(self, data, predictor_variable):
        self.data = data
        self.predictor_variable = predictor_variable
        
        self.outcome_variable = None
    
    def column_validator(self):
        """Checks if the predictor and outcome columns given exist in the data given"""
        if self.predictor_variable not in self.data.columns:
            print("Predictor column", {self.predictor_variable}, " does not exist in this data frame.")

        if pd.api.types.is_numeric_dtype(self.data[self.predictor_variable]) == False:
            print("Predictor variable is not numeric, predictor variable must be numeric")
        
        """Get the names of all numerical variables in the data frame and stores it in a new variable"""
        numeric_column_names = self.data.select_dtypes(include = "number").columns.tolist()
        
        """Removes the predictor variable from the numeric variable column"""

        if self.predictor_variable in numeric_column_names:
            numeric_column_names.remove(self.predictor_variable)
        
        """Checks if there are numeric variables other than the predictor variable"""
        if len(numeric_column_names) == 0:
            print("No numeric columns found, data must have numerical data")

        self.outcome_variable = numeric_column_names
        self.predictor = self.data[self.predictor_variable]
        self.outcome = self.data[self.outcome_variable]
        

    def get_predictor(self):
        """ A getter, it returns the predictor variable """
        return self.predictor
    
    def get_outcome(self):
        """ A getter, it returns the outcome variable"""
        return self.outcome
    
    def summary_stats(self):
        """ Returns a "concatenated version of our predictor column and our outcome variable dataframe to return the summary statistics of them"""
        concat_dataframe = pd.concat([self.predictor, self.outcome], axis = 1)

        return concat_dataframe.describe()