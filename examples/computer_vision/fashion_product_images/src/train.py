import numpy as np
import torch
from torch import nn, optim
from torchvision.transforms import v2
from tqdm import tqdm

# Data Transformations
transform = v2.Compose(
    [
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Resize((64, 64)),
        v2.RGB(),
    ]
)


# Define torch model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


#### Train  #####
def train_model(train_loader, num_classes, num_epochs=20, lr=0.001):
    """
    Trains a convolutional neural network on a given dataset.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        num_classes (int): Number of classes in the dataset.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 20.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the trained model and the optimizer used for training.

    """

    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_loss = []

    for epoch in range(num_epochs):
        for i, data in tqdm(enumerate(train_loader)):  # noqa: B007
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        epoch_mean_loss = np.mean(epoch_loss)
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, epoch_mean_loss))
        epoch_loss = []

    print("Finished Training")
    return model, optimizer


#### Evaluate  #####
def evaluate_model(model, test_loader, label_names) -> tuple[list[dict], list[int]]:
    # Initialize lists to store predictions and labels
    predictions = []
    true_labels = []

    # Set model to evaluation mode
    model.eval()

    print("Run inference")
    for i, data in tqdm(enumerate(test_loader)):
        inputs, labels = data
        true_labels.extend(labels.tolist())

        # Forward pass
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_labels = torch.max(probs, 1)

        for i in range(len(predicted_labels)):
            label = int(predicted_labels[i].item())
            confidence_value = float(confidence[i].item())
            label_name = label_names[label]
            predictions.append(
                {
                    "pred_class": label,
                    "predc_proba": confidence_value,
                    "pred_label": label_name,
                }
            )

    return predictions, true_labels


#### Evaluation Reports  #####
def build_evaluation_report(pred_labels, true_labels) -> tuple:
    """
    Builds an evaluation report for classification results, including a confusion matrix
    and a classification report.

    Args:
        pred_labels (array-like): Predicted labels from the model.
        true_labels (array-like): True labels from the dataset.

    Returns:
        tuple: A tuple containing the confusion matrix and classification report.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    print("Evaluation reports:\n")

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate classification report with zero_division parameter set to 0
    class_report = classification_report(true_labels, pred_labels, zero_division=0)

    # Print classification report
    print("Classification Report:")
    print(class_report)

    return conf_matrix, class_report
