import importlib
import inspect
import typing

import networkx
import openff.interchange.components.potentials
import openff.interchange.models
import openff.interchange.smirnoff
import openff.toolkit
import openff.units
import torch

import smee
import smee.geometry


class _Converter(typing.NamedTuple):
    fn: typing.Callable
    """The function that will convert the parameters of a handler into tensors."""
    units: dict[str, openff.units.Unit]
    """The default units of each parameter in the handler."""
    depends_on: list[str] | None
    """The names of other converters that this converter should be run after."""


_CONVERTERS: dict[str, _Converter] = {}

_ANGSTROM = openff.units.unit.angstrom
_RADIANS = openff.units.unit.radians

_V_SITE_DEFAULT_VALUES = {
    "distance": 0.0 * _ANGSTROM,
    "inPlaneAngle": torch.pi * _RADIANS,
    "outOfPlaneAngle": 0.0 * _RADIANS,
}


def smirnoff_parameter_converter(
    type_: str,
    default_units: dict[str, openff.units.Unit],
    depends_on: list[str] | None = None,
):
    """A decorator used to flag a function as being able to convert a parameter handlers
    parameters into tensors.

    Args:
        type_: The type of parameter handler that the decorated function can convert.
        default_units: The default units of each parameter in the handler.
        depends_on: The names of other handlers that this handler depends on. When set,
            the convert function should additionally take in a list of the already
            converted potentials and return a new list of potentials that should either
            include or replace the original potentials.
    """

    def parameter_converter_inner(func):
        if type_ in _CONVERTERS:
            raise KeyError(f"A {type_} converter is already registered.")

        _CONVERTERS[type_] = _Converter(func, default_units, depends_on)

        return func

    return parameter_converter_inner


def _get_value(
    potential: openff.interchange.components.potentials.Potential,
    parameter: str,
    default_units: dict[str, openff.units.Unit],
    default_value: openff.units.Quantity | None = None,
) -> float:
    """Returns the value of a parameter in its default units"""
    default_units = default_units[parameter]
    
    # Handle missing parameters by using default values
    if parameter not in potential.parameters:
        if default_value is None:
            # Set specific defaults for known multipole parameters
            if parameter == "thole":
                default_value = 0.39 * openff.units.unit.dimensionless
            elif parameter == "dampingFactor":
                # Will be computed from polarizability if not provided
                return 0.0  # Placeholder, will be computed later
            elif parameter.startswith("dipole"):
                default_value = 0.0 * openff.units.unit.elementary_charge * openff.units.unit.angstrom
            elif parameter.startswith("quadrupole"):
                default_value = 0.0 * openff.units.unit.elementary_charge * openff.units.unit.angstrom**2
            elif parameter in ["axisType", "multipoleAtomZ", "multipoleAtomX", "multipoleAtomY"]:
                default_value = -1 * openff.units.unit.dimensionless  # -1 indicates undefined
            else:
                raise KeyError(f"Parameter '{parameter}' not found and no default provided")
        return default_value.m_as(default_units)
    
    value = potential.parameters[parameter]
    return (value if value is not None else default_value).m_as(default_units)


def _handlers_to_potential(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    handler_type: str,
    parameter_cols: tuple[str, ...],
    attribute_cols: tuple[str, ...] | None,
) -> smee.TensorPotential:
    potential_fns = {handler.expression for handler in handlers}
    assert len(potential_fns) == 1, "multiple handler functions found"
    potential_fn = next(iter(potential_fns))

    parameters_by_key = {
        parameter_key: parameter
        for handler in handlers
        for parameter_key, parameter in handler.potentials.items()
    }

    parameter_keys = [*parameters_by_key]
    parameters = torch.tensor(
        [
            [
                _get_value(
                    parameters_by_key[parameter_key],
                    column,
                    _CONVERTERS[handler_type].units,
                )
                for column in parameter_cols
            ]
            for parameter_key in parameter_keys
        ],
        dtype=torch.float64,
    )

    attributes = None

    if attribute_cols is not None:
        attributes_by_column = {
            column: openff.units.Quantity(getattr(handler, column))
            for column in attribute_cols
            for handler in handlers
        }
        attributes_by_column = {
            k: v.m_as(_CONVERTERS[handler_type].units[k])
            for k, v in attributes_by_column.items()
        }
        attributes = torch.tensor(
            [attributes_by_column[column] for column in attribute_cols],
            dtype=torch.float64,
        )

    potential = smee.TensorPotential(
        type=handler_type,
        fn=potential_fn,
        parameters=parameters,
        parameter_keys=parameter_keys,
        parameter_cols=parameter_cols,
        parameter_units=tuple(
            _CONVERTERS[handler_type].units[column] for column in parameter_cols
        ),
        attributes=attributes,
        attribute_cols=attribute_cols,
        attribute_units=(
            None
            if attribute_cols is None
            else tuple(
                _CONVERTERS[handler_type].units[column] for column in attribute_cols
            )
        ),
    )
    return potential


def _convert_v_sites(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFVirtualSiteCollection],
    topologies: list[openff.toolkit.Topology],
) -> tuple[smee.TensorVSites, list[smee.VSiteMap | None]]:
    handler_types = {handler.type for handler in handlers}
    assert handler_types == {"VirtualSites"}, "invalid handler types found"

    assert all(
        isinstance(key, openff.interchange.models.VirtualSiteKey)
        for handler in handlers
        if handler is not None
        for key in handler.key_map
    ), "only v-site keys expected"

    # account for the fact the interchange doesn't track v-site parameter types
    parameter_key_to_type = {
        parameter_key: v_site_key.type
        for handler in handlers
        if handler is not None
        for v_site_key, parameter_key in handler.key_map.items()
    }
    parameters_by_key = {
        parameter_key: parameter
        for handler in handlers
        if handler is not None
        for parameter_key, parameter in handler.potentials.items()
    }
    parameter_keys = [*parameters_by_key]
    parameters = [
        [
            _get_value(
                parameters_by_key[parameter_key],
                column,
                smee.TensorVSites.default_units(),
                _V_SITE_DEFAULT_VALUES[column],
            )
            for column in ("distance", "inPlaneAngle", "outOfPlaneAngle")
        ]
        for parameter_key in parameter_keys
    ]

    v_site_frames = [
        smee.geometry.V_SITE_TYPE_TO_FRAME[parameter_key_to_type[parameter_key]]
        for parameter_key in parameter_keys
    ]
    v_sites = smee.TensorVSites(
        keys=parameter_keys, weights=v_site_frames, parameters=torch.tensor(parameters)
    )

    v_site_maps = []

    for topology, handler in zip(topologies, handlers, strict=True):
        if handler is None:
            v_site_maps.append(None)
            continue

        v_site_keys = sorted(
            handler.key_map,
            key=lambda k: handler.virtual_site_key_topology_index_map[k],
        )
        parameter_idxs = [
            parameter_keys.index(handler.key_map[v_site_key])
            for v_site_key in v_site_keys
        ]

        # check v-sites aren't interleaved between atoms, which isn't supported yet
        # TODO: interchange (#812) means we can't safely do this check currently
        # topology_indices = sorted(
        #     idx - topology.n_atoms
        #     for idx in handler.virtual_site_key_topology_index_map.values()
        # )
        # assert topology_indices == list(range(len(topology_indices)))

        v_site_map = smee.VSiteMap(
            keys=v_site_keys,
            key_to_idx={key: i + topology.n_atoms for i, key in enumerate(v_site_keys)},
            parameter_idxs=torch.tensor(parameter_idxs),
        )
        v_site_maps.append(v_site_map)

    return v_sites, v_site_maps


def _convert_constraints(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFConstraintCollection],
) -> list[smee.TensorConstraints | None]:
    handler_types = {handler.type for handler in handlers}
    assert handler_types == {"Constraints"}, "invalid handler types found"

    constraints = []

    for handler in handlers:
        if handler is None or len(handler.key_map) == 0:
            constraints.append(None)
            continue

        topology_keys = [*handler.key_map]

        constraint_idxs = torch.tensor([[*key.atom_indices] for key in topology_keys])
        assert constraint_idxs.shape[1] == 2, "only distance constraints supported"

        units = {"distance": _ANGSTROM}

        constraint_distances = [
            _get_value(handler.potentials[handler.key_map[key]], "distance", units)
            for key in topology_keys
        ]

        constraint = smee.TensorConstraints(
            idxs=constraint_idxs, distances=torch.tensor(constraint_distances)
        )
        constraints.append(constraint)

    return constraints


def convert_handlers(
    handlers: list[openff.interchange.smirnoff.SMIRNOFFCollection],
    topologies: list[openff.toolkit.Topology],
    v_site_maps: list[smee.VSiteMap | None] | None = None,
    potentials: (
        list[tuple[smee.TensorPotential, list[smee.ParameterMap]]] | None
    ) = None,
    constraints: list[smee.TensorConstraints | None] | None = None,
) -> list[tuple[smee.TensorPotential, list[smee.ParameterMap]]]:
    """Convert a set of SMIRNOFF parameter handlers into a set of tensor potentials.

    Args:
        handlers: The SMIRNOFF parameter handler collections for a set of interchange
            objects to convert.
        topologies: The topologies associated with each interchange object.
        v_site_maps: The v-site maps associated with each interchange object.
        constraints: Any distance constraints between atoms.
        potentials: Already converted parameter handlers that may be required as
            dependencies.

    Returns:
        The potential containing the values of the parameters in each handler
        collection, and a list of maps (one per topology) between molecule elements
        (e.g. bond indices) and parameter indices.

    Examples:

        >>> from openff.toolkit import ForceField, Molecule
        >>> from openff.interchange import Interchange
        >>>
        >>> force_field = ForceField("openff_unconstrained-2.0.0.offxml")
        >>> molecules = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        >>>
        >>> interchanges = [
        ...     Interchange.from_smirnoff(force_field, molecule.to_topology())
        ...     for molecule in molecules
        ... ]
        >>> vdw_handlers = [
        ...     interchange.collections["vdW"] for interchange in interchanges
        ... ]
        >>>
        >>> vdw_potential, applied_vdw_parameters = convert_handlers(interchanges)
    """
    importlib.import_module("smee.converters.openff.nonbonded")
    importlib.import_module("smee.converters.openff.valence")

    handler_types = {handler.type for handler in handlers}
    assert len(handler_types) == 1, "multiple handler types found"
    handler_type = next(iter(handler_types))

    assert len(handlers) == len(topologies), "mismatched number of topologies"

    if handler_type not in _CONVERTERS:
        raise NotImplementedError(f"{handler_type} handlers is not yet supported.")

    constraints = [None] * len(topologies) if constraints is None else constraints

    converter = _CONVERTERS[handler_type]
    converter_spec = inspect.signature(converter.fn)
    converter_kwargs = {}

    if "topologies" in converter_spec.parameters:
        converter_kwargs["topologies"] = topologies
    if "v_site_maps" in converter_spec.parameters:
        assert v_site_maps is not None, "v-site maps must be provided"
        converter_kwargs["v_site_maps"] = v_site_maps
    if "constraints" in converter_spec.parameters:
        constraint_idxs = [[] if v is None else v.idxs.tolist() for v in constraints]
        unique_idxs = [{tuple(sorted(idxs)) for idxs in v} for v in constraint_idxs]

        converter_kwargs["constraints"] = unique_idxs

    potentials_by_type = (
        {}
        if potentials is None
        else {potential.type: (potential, maps) for potential, maps in potentials}
    )

    dependencies = {}
    depends_on = converter.depends_on if converter.depends_on is not None else []

    if len(depends_on) > 0:
        missing_deps = {dep for dep in depends_on if dep not in potentials_by_type}
        assert len(missing_deps) == 0, "missing dependencies"

        dependencies = {dep: potentials_by_type[dep] for dep in depends_on}
        assert "dependencies" in converter_spec.parameters, "dependencies not accepted"

    if "dependencies" in converter_spec.parameters:
        converter_kwargs["dependencies"] = dependencies

    converted = converter.fn(handlers, **converter_kwargs)
    converted = [converted] if not isinstance(converted, list) else converted

    converted_by_type = {
        potential.type: (potential, maps) for potential, maps in converted
    }
    assert len(converted_by_type) == len(converted), "duplicate potentials found"

    potentials_by_type = {
        **{
            potential.type: (potential, maps)
            for potential, maps in potentials_by_type.values()
            if potential.type not in depends_on
            and potential.type not in converted_by_type
        },
        **converted_by_type,
    }

    return [*potentials_by_type.values()]


def _convert_topology(
    topology: openff.toolkit.Topology,
    parameters: dict[str, smee.ParameterMap],
    v_sites: smee.VSiteMap | None,
    constraints: smee.TensorConstraints | None,
) -> smee.TensorTopology:
    """Convert an OpenFF topology into a tensor topology.

    Args:
        topology: The topology to convert.
        parameters: The parameters assigned to the topology.
        v_sites: The v-sites assigned to the topology.

    Returns:
        The converted topology.
    """

    atomic_nums = torch.tensor([atom.atomic_number for atom in topology.atoms])

    formal_charges = torch.tensor(
        [atom.formal_charge.m_as(openff.units.unit.e) for atom in topology.atoms]
    )

    bond_idxs = torch.tensor(
        [
            (topology.atom_index(bond.atom1), topology.atom_index(bond.atom2))
            for bond in topology.bonds
        ]
    )
    bond_orders = torch.tensor([bond.bond_order for bond in topology.bonds])

    return smee.TensorTopology(
        atomic_nums=atomic_nums,
        formal_charges=formal_charges,
        bond_idxs=bond_idxs,
        bond_orders=bond_orders,
        parameters=parameters,
        v_sites=v_sites,
        constraints=constraints,
    )


def _resolve_conversion_order(handler_types: list[str]) -> list[str]:
    """Resolve the order in which the handlers should be converted, based on their
    dependencies with each other."""
    dep_graph = networkx.DiGraph()

    for handler_type in handler_types:
        dep_graph.add_node(handler_type)

    for handler_type in handler_types:
        converter = _CONVERTERS[handler_type]

        if converter.depends_on is None:
            continue

        for dep in converter.depends_on:
            dep_graph.add_edge(dep, handler_type)

    return list(networkx.topological_sort(dep_graph))


def convert_interchange(
    interchange: openff.interchange.Interchange | list[openff.interchange.Interchange],
) -> tuple[smee.TensorForceField, list[smee.TensorTopology]]:
    """Convert a list of interchange objects into tensor potentials.

    Args:
        interchange: The list of (or singile) interchange objects to convert into
            tensor potentials.

    Returns:
        The tensor force field containing the parameters of each handler, and a list
        (one per interchange) of objects mapping molecule elements (e.g. bonds, angles)
        to corresponding handler parameters.

    Examples:

        >>> from openff.toolkit import ForceField, Molecule
        >>> from openff.interchange import Interchange
        >>>
        >>> force_field = ForceField("openff_unconstrained-2.0.0.offxml")
        >>> molecules = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CC")]
        >>>
        >>> interchanges = [
        ...     Interchange.from_smirnoff(force_field, molecule.to_topology())
        ...     for molecule in molecules
        ... ]
        >>>
        >>> tensor_ff, tensor_topologies = convert_interchange(interchanges)
    """
    importlib.import_module("smee.converters.openff.nonbonded")
    importlib.import_module("smee.converters.openff.valence")

    interchanges = (
        [interchange]
        if isinstance(interchange, openff.interchange.Interchange)
        else interchange
    )
    topologies = []

    handler_types = {
        handler_type
        for interchange in interchanges
        for handler_type in interchange.collections
    }
    handlers_by_type = {handler_type: [] for handler_type in sorted(handler_types)}

    for interchange in interchanges:
        for handler_type in handlers_by_type:
            handler = (
                None
                if handler_type not in interchange.collections
                else interchange.collections[handler_type]
            )
            handlers_by_type[handler_type].append(handler)

        topologies.append(interchange.topology)

    v_sites, v_site_maps = None, [None] * len(topologies)

    if "VirtualSites" in handlers_by_type:
        v_sites, v_site_maps = _convert_v_sites(
            handlers_by_type.pop("VirtualSites"), topologies
        )

    constraints = [None] * len(topologies)

    if "Constraints" in handlers_by_type:
        constraints = _convert_constraints(handlers_by_type.pop("Constraints"))

    conversion_order = _resolve_conversion_order([*handlers_by_type])
    converted = []

    for handler_type in conversion_order:
        handlers = handlers_by_type[handler_type]

        if (
            sum(len(handler.potentials) for handler in handlers if handler is not None)
            == 0
        ):
            continue

        converted = convert_handlers(
            handlers, topologies, v_site_maps, converted, constraints
        )

    # handlers may either return multiple potentials, or condense multiple already
    # converted potentials into a single one (e.g. electrostatics into some polarizable
    # potential)
    potentials = []
    parameter_maps_by_handler = {}

    for potential, parameter_maps in converted:
        potentials.append(potential)
        parameter_maps_by_handler[potential.type] = parameter_maps

    tensor_topologies = [
        _convert_topology(
            topology,
            {
                potential.type: parameter_maps_by_handler[potential.type][i]
                for potential in potentials
            },
            v_site_maps[i],
            constraints[i],
        )
        for i, topology in enumerate(topologies)
    ]

    tensor_force_field = smee.TensorForceField(potentials, v_sites)
    return tensor_force_field, tensor_topologies
