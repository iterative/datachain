# Experiments (DVC Integration)

DataChain Studio provides comprehensive ML experiment tracking through DVC integration, allowing you to track, compare, and manage your machine learning experiments with Git-based versioning.

## Overview

The experiments feature enables:

- **Experiment Tracking**: Track ML experiments with parameters, metrics, and artifacts
- **Model Registry**: Manage model lifecycle from development to production
- **Visualization**: Compare experiments with interactive charts and plots
- **Git Integration**: Version control for experiments using DVC and Git
- **Collaboration**: Share experiments and models with team members

## Key Features

### Experiment Management
- **Run Tracking**: Automatically track experiment runs and results
- **Parameter Logging**: Record hyperparameters and configuration
- **Metrics Collection**: Capture model performance metrics
- **Artifact Storage**: Store model files, plots, and other outputs

### Model Registry
- **Model Versioning**: Version your models with semantic versioning
- **Stage Management**: Promote models through stages (staging, production, etc.)
- **Model Comparison**: Compare different model versions and performance
- **Deployment Integration**: Integrate with deployment pipelines

### Visualization and Analysis
- **Metrics Plots**: Visualize training metrics and performance trends
- **Parameter Comparison**: Compare hyperparameters across experiments
- **Model Performance**: Analyze model accuracy, loss, and other metrics
- **Interactive Dashboards**: Explore experiments with interactive visualizations

## Getting Started with DVC Experiments

### Prerequisites
- Git repository with DVC tracking
- Python environment with DVC and dvclive
- DataChain Studio account with repository connected

### Basic Experiment Tracking

1. **Set up DVC in your repository**:
   ```bash
   dvc init
   git add .dvc
   git commit -m "Initialize DVC"
   ```

2. **Install experiment tracking dependencies**:
   ```bash
   pip install dvc[studio] dvclive
   ```

3. **Add experiment tracking to your training code**:
   ```python
   import dvclive
   
   with dvclive.Live() as live:
       for epoch in range(epochs):
           # Training code
           train_loss = train_model()
           val_loss = validate_model()
           
           # Log metrics
           live.log_metric("train/loss", train_loss)
           live.log_metric("val/loss", val_loss)
           live.next_step()
       
       # Log final model
       live.log_artifact("model.pkl", type="model")
   ```

4. **Run experiments**:
   ```bash
   dvc exp run python train.py
   ```

5. **Push experiments to Studio**:
   ```bash
   dvc studio start
   dvc exp push origin
   ```

### Model Registry Integration

1. **Register models from experiments**:
   ```bash
   # Register model from experiment
   dvc artifacts get model.pkl --rev experiment-branch
   
   # Add to model registry
   dvc studio model create --name customer-model --version v1.0.0
   ```

2. **Promote models through stages**:
   ```bash
   # Assign to staging
   dvc studio model assign customer-model --stage staging --version v1.0.0
   
   # Promote to production
   dvc studio model assign customer-model --stage production --version v1.0.0
   ```

## DVC vs DataChain Workflows

### When to Use DVC Experiments
- **ML Model Development**: Training and tuning machine learning models
- **Hyperparameter Optimization**: Tracking parameter sweeps and optimization
- **Model Comparison**: Comparing different model architectures and approaches
- **Reproducible Research**: Ensuring reproducible ML experiments
- **Git-based Versioning**: When you want everything versioned in Git

### When to Use DataChain Jobs
- **Data Processing**: Large-scale unstructured data processing
- **ETL Workflows**: Extract, transform, load operations
- **Data Quality**: Data validation and quality checks
- **Feature Engineering**: Processing raw data into features
- **Batch Processing**: Scheduled or event-driven data processing

### Hybrid Workflows
Many teams use both approaches:
1. **DataChain jobs** for data preprocessing and feature engineering
2. **DVC experiments** for model training and evaluation
3. **DataChain Studio** as the unified interface for both workflows

## Integration with DataChain Workflows

### Data Pipeline Integration

```python
# DataChain processing feeds into DVC experiments
from datachain import DataChain
import dvclive

# Process data with DataChain
processed_data = (
    DataChain.from_storage("s3://raw-data/")
    .map(preprocess_function)
    .save("processed_features")
)

# Export for ML training
processed_data.export("features.parquet")

# Train model with DVC tracking
with dvclive.Live() as live:
    model = train_model("features.parquet")
    accuracy = evaluate_model(model)
    
    live.log_metric("accuracy", accuracy)
    live.log_artifact("model.pkl", type="model")
```

### Unified Monitoring

Monitor both DataChain jobs and DVC experiments in one place:
- **Job Status**: Track data processing job progress
- **Experiment Progress**: Monitor ML training progress
- **Resource Usage**: View resource consumption across all workflows
- **Performance Metrics**: Compare both data processing and ML metrics

## Best Practices

### Experiment Organization
1. **Consistent Naming**: Use consistent naming conventions for experiments
2. **Parameter Tracking**: Always track relevant hyperparameters
3. **Metric Selection**: Choose meaningful metrics for comparison
4. **Artifact Management**: Store important model artifacts and plots

### Model Registry
1. **Semantic Versioning**: Use semantic versioning for models
2. **Stage Management**: Clearly define model stages and promotion criteria
3. **Documentation**: Document model purpose, performance, and usage
4. **Access Control**: Manage who can promote models to production

### Integration
1. **Pipeline Design**: Design clear boundaries between data processing and ML training
2. **Data Validation**: Validate data quality before training
3. **Monitoring**: Monitor both data pipelines and model performance
4. **Automation**: Automate model retraining when data changes

## Migration from Standalone DVC

If you're currently using DVC without Studio:

1. **Connect your repository** to DataChain Studio
2. **Push existing experiments**:
   ```bash
   dvc studio start
   dvc exp push origin --all
   ```
3. **Register existing models** in the model registry
4. **Set up automated workflows** for continuous training

## Next Steps

- Learn about [dataset management](../datasets/index.md) for data preprocessing
- Explore [job management](../jobs/index.md) for data processing workflows  
- Set up [team collaboration](../team-collaboration.md) for shared experiments
- Configure [API integration](../../api/index.md) for automated workflows