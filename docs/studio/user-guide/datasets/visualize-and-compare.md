# Visualize and Compare

Learn how to visualize dataset contents and compare different versions or processing results in DataChain Studio.

## Overview

DataChain Studio provides powerful visualization and comparison tools to help you understand your data, track changes, and analyze processing results.

## Data Visualization

### Supported Data Types

#### Structured Data
- **Tables**: Tabular data with sortable columns
- **Charts**: Bar charts, line charts, scatter plots
- **Statistics**: Summary statistics and distributions
- **Correlations**: Correlation matrices and heatmaps

#### Unstructured Data
- **Images**: Image thumbnails and galleries
- **Text**: Text previews and word clouds
- **Audio**: Audio waveforms and spectrograms
- **Video**: Video thumbnails and metadata

#### Geospatial Data
- **Maps**: Interactive maps with data points
- **Heatmaps**: Geographical distribution visualizations
- **Trajectories**: Path and movement visualizations
- **Boundaries**: Geographical boundaries and regions

### Visualization Types

#### Statistical Visualizations
```python
# Example: Data distribution
dc = DataChain.from_storage("s3://data/")
dc.visualize(
    type="histogram",
    column="price",
    bins=50,
    title="Price Distribution"
)
```

#### Content Visualizations
```python
# Example: Image gallery
image_dc = DataChain.from_storage("s3://images/")
image_dc.visualize(
    type="gallery",
    columns=["image", "label"],
    thumbnail_size=(200, 200)
)
```

#### Time Series Visualizations
```python
# Example: Time series plot
timeseries_dc = DataChain.from_storage("s3://timeseries/")
timeseries_dc.visualize(
    type="timeseries",
    x_column="timestamp",
    y_column="value",
    group_by="sensor_id"
)
```

## Dataset Comparison

### Version Comparison

Compare different versions of the same dataset:

#### Schema Comparison
- **Column Changes**: Added, removed, or modified columns
- **Type Changes**: Data type modifications
- **Constraint Changes**: Constraint additions or removals
- **Index Changes**: Index modifications

#### Content Comparison
- **Row Count**: Changes in dataset size
- **Value Distribution**: Statistical distribution changes
- **New/Deleted Records**: Identify added or removed records
- **Modified Records**: Track changes to existing records

#### Visual Diff
```python
# Compare two dataset versions
v1 = DataChain.from_storage("s3://data/v1/")
v2 = DataChain.from_storage("s3://data/v2/")

comparison = v1.compare(v2)
comparison.visualize(
    type="diff",
    show_changes=True,
    highlight_differences=True
)
```

### Processing Results Comparison

Compare results from different processing jobs:

#### Performance Metrics
- **Processing Time**: Execution time comparison
- **Resource Usage**: CPU, memory, and storage usage
- **Error Rates**: Error and success rate comparison
- **Throughput**: Data processing throughput metrics

#### Output Comparison
- **Result Quality**: Quality metrics comparison
- **Output Size**: Size and volume comparisons
- **Content Changes**: Changes in processed content
- **Accuracy Metrics**: Model accuracy and performance metrics

#### Side-by-side Visualization
```python
# Compare job results
job1_results = DataChain.from_job("job-123")
job2_results = DataChain.from_job("job-456")

DataChain.compare_jobs([job1_results, job2_results]).visualize(
    type="side_by_side",
    metrics=["accuracy", "processing_time", "output_size"]
)
```

## Interactive Dashboards

### Dashboard Creation

Create custom dashboards for your datasets:

#### Dashboard Components
- **Charts**: Various chart types for data visualization
- **Tables**: Interactive data tables with filtering
- **Metrics**: Key performance indicators and statistics
- **Controls**: Filters, selectors, and input controls

#### Dashboard Example
```yaml
# dashboard.yaml
dashboard:
  title: "Customer Data Analysis"
  components:
    - type: "metric"
      title: "Total Customers"
      query: "SELECT COUNT(*) FROM customers"

    - type: "chart"
      title: "Customer Distribution by Region"
      chart_type: "bar"
      query: "SELECT region, COUNT(*) FROM customers GROUP BY region"

    - type: "table"
      title: "Recent Customers"
      query: "SELECT * FROM customers ORDER BY created_at DESC LIMIT 100"
```

### Real-time Updates

Dashboards can update automatically:
- **Live Data**: Connect to streaming data sources
- **Scheduled Updates**: Refresh data on a schedule
- **Event-triggered Updates**: Update when new data arrives
- **Manual Refresh**: User-initiated data refresh

## Advanced Visualizations

### Custom Visualizations

Create custom visualizations for specific use cases:

#### Custom Chart Types
```python
# Custom visualization function
def custom_scatter_plot(dc, x_col, y_col, color_col):
    import matplotlib.pyplot as plt

    data = dc.to_pandas()
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        data[x_col],
        data[y_col],
        c=data[color_col],
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    return plt

# Use custom visualization
dc = DataChain.from_storage("s3://data/")
plot = custom_scatter_plot(dc, "feature1", "feature2", "label")
plot.show()
```

#### Integration with Visualization Libraries
- **Matplotlib**: Static plots and charts
- **Plotly**: Interactive visualizations
- **Seaborn**: Statistical visualizations
- **Bokeh**: Web-based interactive plots

### 3D Visualizations

For complex multidimensional data:
```python
# 3D scatter plot
dc.visualize_3d(
    x="feature1",
    y="feature2",
    z="feature3",
    color="cluster",
    size="importance"
)
```

## Export and Sharing

### Export Options

Export visualizations in various formats:
- **Images**: PNG, JPG, SVG formats
- **Interactive**: HTML with embedded JavaScript
- **Data**: CSV, JSON, Parquet exports
- **Reports**: PDF reports with multiple visualizations

### Sharing Visualizations

Share visualizations with others:
- **Public Links**: Generate shareable links
- **Embedding**: Embed in websites or applications
- **Team Sharing**: Share within team workspaces
- **API Access**: Programmatic access to visualizations

## Performance Optimization

### Large Dataset Visualization

Handle large datasets efficiently:
- **Sampling**: Visualize representative samples
- **Aggregation**: Pre-aggregate data for performance
- **Lazy Loading**: Load data on demand
- **Caching**: Cache expensive computations

### Optimization Tips
```python
# Optimize for large datasets
large_dc = DataChain.from_storage("s3://big-data/")

# Use sampling for exploration
sample_dc = large_dc.sample(n=10000)
sample_dc.visualize(type="histogram", column="value")

# Use aggregation for summaries
aggregated = large_dc.group_by("category").agg({"value": ["mean", "count"]})
aggregated.visualize(type="bar")
```

## Next Steps

- Learn about [job monitoring](../jobs/monitor-jobs.md) for visualizing job results
- Explore [API integration](../../api/index.md) for programmatic visualization
- Set up [webhooks](../../webhooks.md) for automated visualization updates
- Check out [team collaboration](../team-collaboration.md) for sharing visualizations
