## Library GeoMLE

#### Data generation
```python
from geomle import DataGenerator
DG = DataGenerator()
data = DG.gen_data('Sphere', 1000, 2, 1)
```
#### Levina-Bickel dimension
```python
from geomle import mle
mle(data)
```
#### GeoMLE dimension
```python
from geomle import geomle
geomle(data)
```

