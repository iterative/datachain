from datachain.dataset import DatasetDependency, DatasetDependencyNode


def build_dependency_hierarchy(
    dependency_nodes: list[DatasetDependencyNode | None],
) -> tuple[
    dict[int, DatasetDependency | None], dict[tuple[int, int | None], list[int]]
]:
    """
    Build dependency hierarchy from dependency nodes.

    Args:
        dependency_nodes: List of DatasetDependencyNode objects from the database

    Returns:
        Tuple of (dependency_map, children_map) where:
        - dependency_map: Maps dependency_id -> DatasetDependency
        - children_map: Maps (source_dataset_id, source_version_id) ->
          list of dependency_ids
    """
    dependency_map: dict[int, DatasetDependency | None] = {}
    children_map: dict[tuple[int, int | None], list[int]] = {}

    for node in dependency_nodes:
        if node is None:
            continue
        dependency = node.to_dependency()

        if dependency is not None:
            dependency_map[dependency.id] = dependency
            parent_key = (node.source_dataset_id, node.source_dataset_version_id)
            children_map.setdefault(parent_key, []).append(dependency.id)
        else:
            # Handle case where dependency creation failed
            dependency_map[node.id] = None

    return dependency_map, children_map


def populate_nested_dependencies(
    dependency: DatasetDependency,
    dependency_nodes: list[DatasetDependencyNode | None],
    dependency_map: dict[int, DatasetDependency | None],
    children_map: dict[tuple[int, int | None], list[int]],
) -> None:
    """
    Recursively populate nested dependencies for a given dependency.

    Args:
        dependency: The dependency to populate nested dependencies for
        dependency_nodes: All dependency nodes from the database
        dependency_map: Maps dependency_id -> DatasetDependency
        children_map: Maps (source_dataset_id, source_version_id) ->
        list of dependency_ids
    """
    # Find the target dataset and version for this dependency
    target_dataset_id, target_version_id = find_target_dataset_version(
        dependency, dependency_nodes
    )

    if target_dataset_id is None or target_version_id is None:
        return

    # Get children for this target
    target_key = (target_dataset_id, target_version_id)
    if target_key not in children_map:
        dependency.dependencies = []
        return

    child_dependency_ids = children_map[target_key]
    child_dependencies = [
        dependency_map[child_id]
        for child_id in child_dependency_ids
        if dependency_map[child_id] is not None
    ]

    dependency.dependencies = child_dependencies

    # Recursively populate children
    for child_dependency in child_dependencies:
        if child_dependency is not None:
            populate_nested_dependencies(
                child_dependency, dependency_nodes, dependency_map, children_map
            )


def find_target_dataset_version(
    dependency: DatasetDependency,
    dependency_nodes: list[DatasetDependencyNode | None],
) -> tuple[int | None, int | None]:
    """
    Find the target dataset ID and version ID for a given dependency.

    Args:
        dependency: The dependency to find target for
        dependency_nodes: All dependency nodes from the database

    Returns:
        Tuple of (target_dataset_id, target_version_id) or (None, None) if not found
    """
    for node in dependency_nodes:
        if node is not None and node.id == dependency.id:
            return node.dataset_id, node.dataset_version_id
    return None, None
