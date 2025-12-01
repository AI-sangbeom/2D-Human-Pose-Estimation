import core.metrics as metrics
from models import * 

metric = getattr(metrics, 'PCK')
print(metric)
