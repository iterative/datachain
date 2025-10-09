from typing import Any

from datachain.dataset import DatasetDependency


def extract_flat_ids(d: dict[str, Any]) -> list[int]:
    """
    Extracts all unique dataset IDs from a nested dictionary structure.
    """
    ids = set()
    for key, value in d.items():
        ids.add(int(key))
        if isinstance(value, dict):
            ids.update(extract_flat_ids(value))
    return sorted(ids)


def build_nested_dependencies(
    dataset_deps: dict[str, DatasetDependency],
    dependency_structure: dict[str, dict],
    indirect: bool = True,
    max_depth: int = 100,
) -> list[DatasetDependency | None]:
    """
    Recursively constructs a tree of dataset dependencies based on their relationships,
    with each dependency containing its own nested dependencies.

    Args:
        dataset_deps: Dictionary mapping dependency ids to DatasetDependency objects.
        dependency_structure: Dependency tree with nested dependencies.
        indirect: Whether to build nested dependencies indirectly.
        max_depth: Maximum recursion depth to prevent infinite loops.

    Returns:
        List of DatasetDependency objects with their nested dependencies.
    """

    def build_deps(dep_id, deps_dict, visited=None, depth=0):
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if depth > max_depth:
            return []

        # Prevent circular dependencies
        if dep_id in visited:
            return []

        if dep_id not in dataset_deps:
            return []

        deps = []
        if dep_id in deps_dict:
            # Add current node to visited set
            visited.add(dep_id)

            for child_id in deps_dict[dep_id]:
                if child_id in dataset_deps:
                    # Recursively build child dependencies if indirect=True
                    child_deps = (
                        build_deps(
                            child_id,
                            deps_dict[dep_id],
                            visited.copy(),
                            depth + 1,
                        )
                        if indirect
                        else []
                    )
                    child_dep = dataset_deps[child_id]
                    child_dep.dependencies = child_deps
                    deps.append(child_dep)
                else:
                    deps.append(None)

            # Remove current node from visited set after processing
            visited.discard(dep_id)
        return deps

    result: list[DatasetDependency | None] = []
    for root_id in dependency_structure:
        if root_id in dataset_deps:
            deps = build_deps(root_id, dependency_structure)
            root_dep = dataset_deps[root_id]
            root_dep.dependencies = deps
            result.append(root_dep)
        else:
            result.append(None)

    return result
