from simple_sales_model.salesmodeler import SalesModeler
from simple_sales_model.salesdatavalidator import SalesData
import pandas as pd

def test_SalesModelerandMethods():
    """
    Tests SalesData and SalesModler classes.

    Attributes:
        SalesData (object): Tests that the object accurately validates and stores the imputed data.
        SalesModeler (class): Tests that the validated SalesData can fit regression models successfully, returning slope, intercept and R^2 for each regression model.
        summary_stats_table (tbl): Tests the accuracy of the outcome varaible names and the table column names.
    """
    sales = pd.DataFrame({"amount_ordered": [5, 10, 15], "revenue": [100, 200, 300], "profit": [300, 600, 900], "product_type": ["lip gloss", "blush", "mascara"]})

    data = SalesData(sales, predictor_variable = "amount_ordered")
    data.column_validator()

    models = SalesModeler(data)
    models.fit_all()

    assert set(models.models.keys()) == {"revenue", "profit"}

    revenue_model = models.get_model_parameters("revenue")
    assert "slope" in revenue_model
    assert "intercept" in revenue_model
    assert "R^2" in revenue_model

    profit_model = models.get_model_parameters("profit")
    assert "slope" in profit_model
    assert "intercept" in profit_model
    assert "R^2" in profit_model

    summary = models.summary_stats_table()

    assert set (summary["outcome"]) == {"revenue", "profit"}
    assert set(summary.columns) == {"outcome", "slope", "intercept", "R^2"}