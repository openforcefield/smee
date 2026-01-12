PACKAGE_NAME  := smee

EXAMPLES_SKIP := examples/md-simulations.ipynb
EXAMPLES := $(filter-out $(EXAMPLES_SKIP), $(wildcard examples/*.ipynb))

.PHONY: env lint format test test-examples docs docs-deploy

env:
	pixi install
	pixi run pre-commit install || true

lint:
	pixi run lint

format:
	pixi run format

test:
	pixi run test

test-examples:
	pixi run test-examples

docs:
	pixi run docs

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	pixi run mike deploy --push --update-aliases $(VERSION)
