#INSTRUCTIONS + EXAMPLE
# To start, you must import pandas and import SalesData and SalesModeler from their respective files.
import pandas as pd
from salesdatavalidator import SalesData
from salesmodeler import SalesModeler

# You import a data frame through pandas. In this example, we created an extremely simple one. 
df = pd.DataFrame({
    "discount": [5, 10, 15],
    "revenue": [100, 150, 200],
    "profit": [10, 25, 40]
})

# You create an object from the class SalesData by passing in your Data Frame as an argument and defining your predictor variable.

sd = SalesData(df, predictor_variable="discount")

# You then call column validator on your object. This method validates your drata frame by making sure your predictor variable is numerical, 
# and finding all other numerical columns, then removing your predictor from that.

print(sd.column_validator())

#This returns your predictor column
print(sd.get_predictor())

#This returns your outcome columns
print(sd.get_outcome())

#This returns the summary statistics of your predictor and outcome variables 
print(sd.summary_stats())

#To define a model, you pass it through the SalesModeler Class with it's argument being the validated object from SalesData's function, column_validated.

models = SalesModeler(sd)

# This fits all of your outcome variables by the predictor variable and prints all of the parameters from all of the models. 
print(models.fit_all())

# If you want to see the parameters for one specific outcome, you call get_model_parameters from your defined model object and pass in the outcome variable.

print(models.get_model_parameters("revenue"))
print(models.get_model_parameters("profit"))

#To get a prettier/easier way to read the models you can create another object that calls your specifed model object and call the summary_stats_table() method.
summary = models.summary_stats_table()
print(summary)
