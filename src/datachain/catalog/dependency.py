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


def _populate_dependency_tree(
    dataset_deps: dict[str, DatasetDependency],
    dependency_structure: dict[str, dict],
    indirect: bool = True,
) -> list[DatasetDependency | None]:
    """
    It takes a flat structure and populates a tree by setting the dependencies
    attribute on each DatasetDependency object.

    Args:
        dataset_deps: Dictionary mapping dependency ids to DatasetDependency objects.
        dependency_structure: Dependency tree with nested dependencies.
        indirect: Whether to build nested dependencies recursively.

    Returns:
        List of DatasetDependency objects with their nested dependencies.
    """

    def _populate_node(
        dep_id: str, deps_dict: dict, visited: set[str]
    ) -> list[DatasetDependency | None]:
        if dep_id in visited or dep_id not in dataset_deps:
            return []

        visited.add(dep_id)
        deps: list[DatasetDependency | None] = []

        if dep_id in deps_dict:
            for child_id in deps_dict[dep_id]:
                if child_id in dataset_deps:
                    child_dep = dataset_deps[child_id]
                    if indirect:
                        child_dep.dependencies = _populate_node(
                            child_id, deps_dict[dep_id], visited.copy()
                        )
                    deps.append(child_dep)
                else:
                    deps.append(None)

        return deps

    result: list[DatasetDependency | None] = []
    for root_id in dependency_structure:
        if root_id in dataset_deps:
            root_dep = dataset_deps[root_id]
            root_dep.dependencies = _populate_node(root_id, dependency_structure, set())
            result.append(root_dep)
        else:
            result.append(None)

    return result
