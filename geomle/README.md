## Library intdim
#### Data generation
```python
from intdim.data import DataGenerator
DG = DataGenerator()
data = DG.gen_data('Sphere', 1000, 2, 1)
```
#### Levina-Bickel dimension
```python
from intdim import mle
mle(data)
```
#### GeoMLE dimension
```python
from intdim import geomle
geomle(data)
```

