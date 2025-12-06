import pandas as pd

class SalesData:
    """
    A class used for validating and storing Sales data.

    Attributes:
        data (pandas.DataFrame): User provided data set. This data set must include the predictor variable and at least one numerical variable.
        predictor_variable (str): The user-chosen variable that will serve as a predictor for following modeling and/or analysis.
        outcome_variables(str or None): Assigned after running column_validator(), list of all  numerical varibables given in 'data', not including predictor_varibale
        predictor (pandas.Series): Assigned after running column_validator(), the extracted predictor column. 
        outcome (pandas.Series): Assigned after running column_validator(), a dataframe that contains all calculated outcome variables.
    """

    def __init__(self, data, predictor_variable):
        """ 
        This function initilizes the class with a provided dataset and user-chosen predictor variable. 

        Parameters:
            data (pandas.DataFrame): The provided data set that the function will use its variables for modeling and/or analysis.
            predictor_variable (str): The user-chosen variable that will serve as a predictor for following modeling and/or analysis.

        Attributes:
            data (pandas.DataFrame): Function stores inputed dataset. 
            predictor_variable (str): Function stores predictor variable.
            outcome_variables(str or None): Empty variable, placeholder.
        """
        self.data = data
        self.predictor_variable = predictor_variable
        
        self.outcome_variable = None
    
    def column_validator(self):
        """
        Checks if the predictor and outcome columns given exist in 'data', that both the provided and outcome variables are numerical. 

        Attributes:
            predictor (pandas.Series): Extracts the predictor variable. 
            outcome (pandas.Series): Compiles of all remaining numerical outcomes variables in a dataframe.

        """
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
        """
        A getter, it returns the predictor variable.

        Returns:
            self.predictor (pandas.Series): The stored predictor variable.
        """
        return self.predictor
    
    def get_outcome(self):
        """ 
        A getter, it returns the outcome variable.
        
        Returns:
            self.outcome_variable (pandas.DataFrame): The stored outcome variables.
        """
        return self.outcome
    
    def summary_stats(self):
        """
        Returns a concatenated version of our predictor column and our outcome variable dataframe to return the summary statistics of them.

        Returns:
            concat_dataframe (pandas.DataFrame): A dataframe containing all summary statistics for all columns.
        """
        concat_dataframe = pd.concat([self.predictor, self.outcome], axis = 1)

        return concat_dataframe.describe()