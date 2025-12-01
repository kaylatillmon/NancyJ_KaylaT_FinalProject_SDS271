#test

import pandas as pd
from salesdatavalidator import SalesData
from salesmodeler import SalesModeler

df = pd.DataFrame({
    "discount": [5, 10, 15],
    "revenue": [100, 150, 200],
    "profit": [10, 25, 40]
})

sd = SalesData(df, predictor_variable="discount")
sd.column_validator()

modeler = SalesModeler(sd)
modeler.fit_all()

print(modeler.models)
summary = modeler.summary_stats_table()
print(summary)
