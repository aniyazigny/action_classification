import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from PoseActionDataset import PoseActionDataset
from PoseActionClassifier import PoseActionClassifierLSTM
from torch.utils.tensorboard import SummaryWriter
import argparse

def train_and_evaluate(train_dir, val_dir, test_dir=None, num_epochs=50, save_path='runs/train/weights', tensorboard_path='runs/train/tensorboard'):
    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Create datasets
    train_dataset = PoseActionDataset(os.path.join(train_dir, 'poses'))
    val_dataset = PoseActionDataset(os.path.join(val_dir, 'poses'))

    # Save the class labels to a text file
    classes_file = os.path.join(save_path, 'classes.txt')
    with open(classes_file, 'w') as f:
        for class_name in train_dataset.class_labels:
            f.write(f"{class_name}\n")
    print(f"Class labels saved to {classes_file}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.class_labels)
    model = PoseActionClassifierLSTM(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set up TensorBoard
    writer = SummaryWriter(tensorboard_path)

    best_val_accuracy = 0.0
    best_model_path = os.path.join(save_path, 'best.pt')
    last_model_path = os.path.join(save_path, 'last.pt')
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        writer.add_scalar('Validation Loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        # Save last model
        torch.save(model.state_dict(), last_model_path)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with accuracy: {best_val_accuracy:.2f}%')

    print('Training finished.')

    # Test the model if test_dir is provided
    if test_dir:
        model.load_state_dict(torch.load(best_model_path))
        test_dataset = PoseActionDataset(os.path.join(test_dir, 'poses'))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        writer.add_scalar('Test Accuracy', test_accuracy)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate PoseActionClassifier with LSTM.')

    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory')
    parser.add_argument('--test_dir', type=str, default=None, help='Path to the test data directory (optional)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--save_path', type=str, default='runs/train/weights', help='Directory to save models')
    parser.add_argument('--tensorboard_path', type=str, default='runs/train/tensorboard', help='Directory to save TensorBoard logs')

    args = parser.parse_args()

    train_and_evaluate(args.train_dir, args.val_dir, args.test_dir, args.num_epochs, args.save_path, args.tensorboard_path)

#sample run command
    
# python train.py --train_dir /home/aniyazi/masters_degree/action_recognition/data/train/ --val_dir /home/aniyazi/masters_degree/action_recognition/data/val/ --test_dir /home/aniyazi/masters_degree/action_recognition/data/test/