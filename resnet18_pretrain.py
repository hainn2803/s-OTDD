import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.datasets import load_imagenet
import argparse
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_data(task_num, parent_dir, batch_size):
    train_path = f'{parent_dir}/data/trainset_{task_num}.pt'
    test_path = f'{parent_dir}/data/test_{task_num}.pt'
    
    train_data, train_labels = torch.load(train_path)
    test_data, test_labels = torch.load(test_path)
    
    dataset_train = TensorDataset(train_data, train_labels)
    dataset_test = TensorDataset(train_data, train_labels)

    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    return dataloader_train, dataloader_test


def get_model(num_classes=40):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE).train()


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")


def train_and_evaluate(task_num, parent_dir, batch_size, num_epochs, learning_rate, 
                      checkpoint_freq=1, resume_from=None):
    print(f"\n{'='*50}\nTraining Task {task_num}\n{'='*50}")
    
    # Initialize paths and directories
    checkpoint_dir = os.path.join(parent_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    train_loader, test_loader = load_data(task_num, parent_dir, batch_size)
    
    # Model setup
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training state variables
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = 0
    best_model_wts = None
    train_history = []
    best_model_path = None

    # Resume training if specified
    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_model_wts = checkpoint['best_model_wts']
        train_history = checkpoint['train_history']
        print(f"Resumed training from epoch {start_epoch} with best loss {best_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_history.append(epoch_loss)
        
        print(f"Task {task_num} | Epoch {epoch+1:02d}/{num_epochs} | Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if checkpoint_freq > 0:
            if (epoch + 1) % checkpoint_freq == 0:
                print("Evaluating...")
                accuracy = evaluate_model(model, test_loader)
                print(f"Task {task_num} | Epoch {epoch+1:02d}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_loss = epoch_loss
                    checkpoint_task_dir = os.path.join(checkpoint_dir, f"task_{task_num}")
                    best_model_path = os.path.join(checkpoint_task_dir, f'task_{task_num}_best.pt')
                    os.makedirs(checkpoint_task_dir, exist_ok=True)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'loss': epoch_loss,
                    }, best_model_path)

    # Save final models
    checkpoint_task_dir = os.path.join(checkpoint_dir, f"task_{task_num}")
    os.makedirs(checkpoint_task_dir, exist_ok=True)
    final_model_path = os.path.join(checkpoint_task_dir, f'task_{task_num}_last.pt')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'loss': train_history[-1],
    }, final_model_path)
    
    # Evaluate both models
    print("\nEvaluating models...")
    # Load final model

    print(f"\nTask {task_num} Evaluation Results:")
    final_model = get_model()
    final_model.load_state_dict(torch.load(final_model_path)["state_dict"])
    final_model = final_model.to(DEVICE).eval()
    final_accuracy = evaluate_model(final_model, test_loader)
    print(f"Final Model Accuracy: {final_accuracy:.2f}%")
    with open(f'{checkpoint_task_dir}/results_task_{task_num}.txt', 'a') as f:
        f.write(f"Final Model Accuracy: {final_accuracy:.2f}%\n")
        f.write(f"Final Training Loss: {train_history[-1]:.4f}\n")

    # Load best model
    if best_model_path is not None:
        best_model = get_model()
        best_model.load_state_dict(torch.load(best_model_path)["state_dict"])
        best_model = best_model.to(DEVICE).eval()
        best_accuracy = evaluate_model(best_model, test_loader)
        print(f"Best Loss Model Accuracy: {best_accuracy:.2f}%")
        with open(f'{checkpoint_task_dir}/results_task_{task_num}.txt', 'a') as f:
            f.write(f"Best Loss Model Accuracy: {best_accuracy:.2f}%\n")
            f.write(f"Best Training Loss: {best_loss:.4f}\n")


def evaluate_model(model, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description='ImageNet Training with Dual Model Saving')
    parser.add_argument('--tasks', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--parent_dir', type=str, default="saved_split_task")
    parser.add_argument('--checkpoint_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    for task_num in args.tasks:
        train_and_evaluate(
            task_num=task_num,
            parent_dir=args.parent_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume
        )

if __name__ == "__main__":
    main()