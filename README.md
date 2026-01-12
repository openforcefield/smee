<h1 align="center">SMIRNOFF Energy Evaluations</h1>

<p align="center">Differentiably evaluate energies of molecules using SMIRNOFF force fields</p>

<p align="center">
  <a href="https://github.com//actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/SimonBoothroyd/smee/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/SimonBoothroyd/smee/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/SimonBoothroyd/smee/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `smee` framework aims to offer a simple API for differentiably evaluating the energy of [SMIRNOFF](https://openforcefield.github.io/standards/standards/smirnoff/)
force fields applied to molecules using `pytorch`.

The package currently supports evaluating the energy of force fields that contain:

* `Bonds`, `Angles`, `ProperTorsions` and `ImproperTorsions`
* `vdW`, `Electrostatics`, `ToolkitAM1BCC`, `LibraryCharges`
* `VirtualSites`

parameter handlers in addition to limited support for registering custom handlers.

It further supports a number of functional forms included in `smirnoff-plugins`, namely:

* `DoubleExponential`

## Installation

This package can be installed using `pixi`:

```shell
pixi add smee
```

or using `conda` / `mamba`:

```shell
mamba install -c conda-forge smee
```

The example notebooks further require you install `jupyter`, `nglview`, and `smirnoff-plugins`:

```shell
pixi add jupyter nglview "smirnoff-plugins >=0.0.4"
```

or with conda/mamba:

```shell
mamba install -c conda-forge jupyter nglview "smirnoff-plugins >=0.0.4"
```

## Getting Started

To get started, see the [examples](examples).

## Copyright

Copyright (c) 2023, Simon Boothroyd
