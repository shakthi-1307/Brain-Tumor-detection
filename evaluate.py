import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "brain_tumor_model.pth"
TEST_DIR = "Testing"

IMAGE_SIZE = 256
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor",
]


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Input: 256x256
        # After 4 pooling layers: 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BRAIN TUMOR CNN - TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(TEST_DIR):

        raise FileNotFoundError(
            f"Testing directory not found: {TEST_DIR}"
        )

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ])

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform
    )

    print(
        f"\nTest images: "
        f"{len(test_dataset):,}"
    )

    print(
        f"Classes: "
        f"{test_dataset.classes}"
    )

    print(
        f"Class mapping: "
        f"{test_dataset.class_to_idx}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = BrainTumorCNN(
        num_classes=4
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            # Sometimes the checkpoint itself
            # is a state_dict.
            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(device)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    print("\nRunning inference...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Samples : {total:,}"
    )

    print(
        f"Correct      : {correct:,}"
    )

    print(
        f"Accuracy     : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision : {precision * 100:.2f}%"
    )

    print(
        f"Macro Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Macro F1        : {f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n")
    print(cm)

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

    for i, class_name in enumerate(
        test_dataset.classes
    ):

        class_total = cm[i].sum()

        class_correct = cm[i, i]

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0
        )

        print(
            f"{class_name:15s} "
            f"{class_correct:4d}/"
            f"{class_total:4d} "
            f"({class_accuracy * 100:.2f}%)"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":

    main()