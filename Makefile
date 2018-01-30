test:
	python -m pytest --pyargs --doctest-modules approxposterior

test-coverage:
	python -m pytest --pyargs --doctest-modules --cov=approxposterior --cov-report term approxposterior

test-coverage-html:
	python -m pytest --pyargs --doctest-modules --cov=approxposterior --cov-report html approxposterior
