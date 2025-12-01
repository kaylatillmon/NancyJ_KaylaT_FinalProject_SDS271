import pandas as pd
from simple_sales_model.salesdatavalidator import SalesData

def test_SalesDataandMethods():
    sales = pd.DataFrame({"amount_ordered": [5, 10, 15], "revenue": [100, 200, 300], "profit": [300, 600, 900], "product_type": ["lip gloss", "blush", "mascara"]})
    
    #creates object, chooses predictor variable , validates columns and sets them up for modeling
    sample = SalesData(sales, predictor_variable="amount_ordered")
    sample.column_validator()

    #tests if the outcome variable are the rest of the numeric variables
    assert set(sample.outcome_variable) == {"revenue", "profit"}

    predictor = sample.get_predictor()
    outcome = sample.get_outcome()

    assert list(predictor.values) == [5, 10, 15]
    assert list(outcome.columns) == ["revenue", "profit"]


    stats = sample.summary_stats()
    assert set(stats.columns) ==  {"amount_ordered", "revenue", "profit"}