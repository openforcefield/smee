# Development

To create a development environment, you must have [`pixi` installed](https://pixi.sh/latest/#installation).

A development environment can be created with:

```shell
pixi install
pixi run pre-commit install
```

This will install all dependencies and set up pre-commit hooks. The pixi environment is automatically activated when you run commands through `pixi run`.

To format the codebase:

```shell
pixi run format
```

To run the unit tests:

```shell
pixi run test
```

To check code style:

```shell
pixi run lint
```

To run the example notebooks:

```shell
pixi run test-examples
```

To build the documentation:

```shell
pixi run docs
```

To serve the documentation locally:

```shell
pixi run docs-serve
```

## Multiple Environments

The project includes two environments to test compatibility with different pydantic versions:

- **default**: Uses pydantic >=2 (includes all features)
- **pydantic-1**: Uses pydantic <2 (excludes `absolv` which requires pydantic >=2 due to use of `femto`)

To run tests in the pydantic-1 environment:

```shell
pixi run --environment pydantic-1 test
```
