# Organizing Datasets with Namespace and Project

DataChain allows you to organize datasets using optional namespaces and projects. These provide an additional structure for managing data across different workflows, use cases, or organizational structures.

A dataset in DataChain can be organized as:

```
<namespace>.<project>.<dataset>
```

For example:

```
dev.analytics.metrics
```

## Default Namespace and Project

If no namespace or project is specified, DataChain uses defaults depending on whether you're using **Studio** or the **CLI**.

### Studio

- **Namespace:** `users`
- **Project:** your username (e.g. `jondoe`)
- Saving without namespace/project:

```python
dc.read_values(scores=[1.2, 3.4, 2.5]).save("metrics")
# Saved as users.jondoe.metrics
```

### CLI

- **Namespace:** `local`
- **Project:** `local`
- Saving without namespace/project:

```python
dc.read_values(scores=[2.0, 2.2, 2.8]).save("metrics")
# Saved as local.local.metrics
```

In the CLI, you cannot create or use any namespaces or projects other than the default `local.local`.

## Creating a Project (Studio only)

In Studio, you can explicitly create a project and namespace using:

```python
import datachain as dc

dc.create_project("analytics", "dev")
```

This creates the `dev` namespace (if it doesn't exist) and a project called `analytics` inside it.

**Note:** Creating custom namespaces and projects is only supported in **Studio**. In the **CLI**, only the default `local` namespace and `local` project are available.

## Saving a Dataset Using a Fully Qualified Name

You can implicitly create and use namespaces and projects by saving a dataset using a fully qualified name:

```python
dc.read_values(scores=[1.2, 3.4, 2.5]).save("dev.analytics.metrics")
```

In Studio, this automatically creates the namespace and project if they don’t already exist.

In CLI, only `local.local.<dataset>` is supported. Using any other namespace or project will result in an error.

## Using `.settings()` to Set Namespace and Project

You can also set the namespace and project using `.settings()`:

```python
dc.read_values(scores=[1.2, 3.4, 2.5])
  .settings(namespace="dev", project="analytics")
  .save("metrics")
```

This is equivalent to saving to `dev.analytics.metrics`.

In CLI, `.settings()` is only supported when both `namespace` and `project` are set to `"local"`.

## Reading a Dataset from a Project

To read a dataset from a specific namespace and project:

```python
ds = dc.read_dataset("dev.analytics.metrics")
```

In CLI, this only works for datasets saved in the default `local.local` project.

## Summary

| Feature                         | Studio                        | CLI                      |
|--------------------------------|-------------------------------|--------------------------|
| Create custom namespace/project | Yes                           | No                       |
| Save to custom project          | Yes (via name or settings)    | No (only `local.local`)  |
| Use default project             | `users.<username>.<dataset>`  | `local.local.<dataset>`  |
| Read from specific project      | Yes                           | Only `local.local`       |

## Example (Studio)

```python
import datachain as dc

dc.create_project("analytics", "prod")

dc.read_csv("gs://bucket/metrics.csv") \
  .save("prod.analytics.metrics")

ds = dc.read_dataset("prod.analytics.metrics")
ds.describe()
```

## Example (CLI – default only)

```python
import datachain as dc

dc.read_values(scores=[0.8, 1.5, 2.1]).save("metrics")

ds = dc.read_dataset("local.local.metrics")
ds.show()
```
