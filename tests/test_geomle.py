import pytest
from geomle import DataGenerator, mle, geomle

def test_mle():
	DG = DataGenerator()
	data = DG.gen_data('Sphere', 1000, 2, 1)
	res = mle(data)[0].mean()
	assert abs(res - 1) < 0.2

def test_geomle():
	DG = DataGenerator()
	data = DG.gen_data('Sphere', 1000, 2, 1)
	res = geomle(data).mean()
	assert abs(res - 1) < 0.2
