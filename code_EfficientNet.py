####
'''
M13A030007
07/15/2025
'''
###
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from efficientnet_pytorch import EfficientNet
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
    import numpy as np
    import seaborn as sns
    import os
    import time
    start_time = time.time()

    # 1. Load EfficientNet-B0 model
    model = EfficientNet.from_pretrained('efficientnet-b7')

    # 2. Freeze model parameters
    for name, param in model.named_parameters():
        if "fc" not in name and "blocks" in name:  # Only unfreeze specific layers
            param.requires_grad = True

    # 3. Modify classification head for 2 classes
    num_classes = 2
    
    # Adjust classification head
    model._fc = nn.Sequential(
        nn.Linear(model._fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, num_classes),
    )
    # 4. Define transformations
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 6. Load datasets
    train_dataset = datasets.ImageFolder(root=r"", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    val_dataset = datasets.ImageFolder(root=r"", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8)

    test_dataset = datasets.ImageFolder(root=r"", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

    # 7. Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.5], device=device, dtype=torch.float))
    optimizer = torch.optim.AdamW([
        {"params": model._fc.parameters(), "lr": 0.001},
        {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": 0.0001}
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 8. Training parameters
    num_epochs = 10000000
    patience = 500
    best_val_acc = 0.0
    early_stop_counter = 0
    train_acc_list = []
    val_acc_list = []
    stop = 0
    
    # Mixed precision training
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

        val_acc = 100 * correct_val / total_val
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:  # When validation accuracy exceeds the current best
            stop = 1  # Reset stable improvement counter
            best_val_acc = val_acc
            early_stop_counter = 0  # Reset early stopping counter
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            print(f"New best model saved with Val Acc: {val_acc:.2f}%")
        elif val_acc == best_val_acc:  # When validation accuracy equals the best
            stop += 1  # Increase stable improvement counter
            if stop == patience:  # Stop if stabilized for enough epochs
                torch.save(model.state_dict(), 'best_model_steady.pth')  # Save the steady best model
                print("Validation accuracy stabilized after consecutive epochs. Stopping training.")
                break
        else:
            stop = 0  # Reset stable counter if no improvement
            early_stop_counter += 1  # Increase early stopping counter
            if early_stop_counter >= patience:  # Stop if no improvement for many epochs
                print("Early stopping triggered due to no improvement.")
                break

        # Step scheduler
        scheduler.step()

    print("\n================= TEST INFERENCE =================\n")
    # Test Inference
    model.eval()
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Inference"):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_probs.extend(probs[:, 1].cpu().numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")
    mins, secs = divmod(elapsed_time, 60)
    print(f"\nTotal elapsed time: {int(mins)} minutes {secs:.2f} seconds")

    # Directory to save plots and results
    save_dir = r"D:\M13A030007\P__EfficientNet__\b7\0720"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "elapsed_time.txt"), "w") as f:
        f.write(f"Total elapsed time: {int(mins)} minutes {secs:.2f} seconds\n")

    # Classification Report
    report_text = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    print(report_text)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)
     
    # Plot accuracy curves
    plt.figure()
    plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(val_acc_list) + 1), val_acc_list, label='Validation Accuracy', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "Val_Training and Validation Accuracy.png"))
    #plt.show()

    # Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_confusion_matrix.png"))
    #plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_roc_curve.png"))
    #plt.show()

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation Precision-Recall Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_pr_curve.png"))
    #plt.show()

    # F1 vs Threshold
    thresholds = np.linspace(0,1,101)
    f1_scores = [f1_score(all_labels, (np.array(all_probs) >= t).astype(int)) for t in thresholds]
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score vs Threshold")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_f1_vs_threshold.png"))
    #plt.show()

    # Confidence Distribution
    plt.figure()
    plt.hist(all_probs, bins=50, color='orange', alpha=0.7)
    plt.xlabel("Predicted Probability for Class 1")
    plt.ylabel("Count")
    plt.title("Validation Confidence Score Distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_confidence_distribution.png"))
    #plt.show()

    # Test Classification Report
    report_text = classification_report(all_test_labels, all_test_preds, target_names=test_dataset.classes)
    print(report_text)

    with open(os.path.join(save_dir, "test_classification_report.txt"), "w") as f:
        f.write(report_text)

    # Test Confusion Matrix
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(all_test_labels, all_test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    #plt.show()

    # Test ROC Curve
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_roc_curve.png"))
    #plt.show()

    # Test PR Curve
    precision, recall, _ = precision_recall_curve(all_test_labels, all_test_probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_pr_curve.png"))
    #plt.show()

    # Test F1 vs Threshold
    thresholds = np.linspace(0,1,101)
    f1_scores = [f1_score(all_test_labels, (np.array(all_test_probs) >= t).astype(int)) for t in thresholds]
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Test F1 Score vs Threshold")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_f1_vs_threshold.png"))
    #plt.show()

    # Test Confidence Distribution
    plt.figure()
    plt.hist(all_test_probs, bins=50, color='green', alpha=0.7)
    plt.xlabel("Predicted Probability for Class 1")
    plt.ylabel("Count")
    plt.title("Test Confidence Score Distribution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_confidence_distribution.png"))
    #plt.show()
