import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train_model(model, train_loader, val_loader, device, epochs, initial_lr):
    print("Training started!")

    # Computing class weights for weighted loss
    class_counts = torch.tensor([28675, 1730], dtype=torch.float)  # [Non-lens, Lens]
    class_weights = class_counts.sum() / (2 * class_counts)
    class_weights = class_weights.to(device)

    # weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)


    # Lists to track losses,accuracies and AUC scores
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_aucs = []

    # mixed precusion training to ease trianing compute cost
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)

    best_val_auc = 0.0  # Track best AUC score
    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs} started")

        # Training Loop below : 
        model.train()
        train_running_loss, correct_train, total_train = 0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed precision for efficient training
                predictions = model(inputs)
                loss = criterion(predictions, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_running_loss += loss.item() * inputs.shape[0]
            correct_train += (predictions.argmax(dim=1) == labels).sum().item()
            total_train += labels.size(0)

        train_loss = train_running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation Phase
        model.eval()
        val_running_loss, correct_val, total_val = 0, 0, 0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, labels)

                val_running_loss += loss.item() * inputs.shape[0]
                correct_val += (predictions.argmax(dim=1) == labels).sum().item()
                total_val += labels.size(0)

                # Collect labels and probabilities for AUC
                probs = torch.softmax(predictions, dim=1)[:, 1]  # Probabilities for class 1 (lenses)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_running_loss / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Compute AUC-ROC for current epoch
        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)

        # SCheduler step
        scheduler.step(metrics=val_loss)

        # Epochwise loss and acuracy log
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.3f} | Val AUC: {val_auc:.4f}")

        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "results/best_model_auc.pth")
            print(f"Saved new best model with Val AUC: {val_auc:.4f}")
        # Save best model based on Val Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "results/best_model_acc.pth")
            print(f"Saved new best model with Val ACC: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs, val_aucs
