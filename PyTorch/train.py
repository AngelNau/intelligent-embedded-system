import torch
import time
from copy import deepcopy


def train_validate_test(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    test_loader=None,
    num_epochs=25,
    scheduler=None,
    early_stop_patience=None,
    min_delta=0.0,
):
    since = time.time()
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_since_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # -------- TRAINING --------
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs = model(inputs)               # shape: [batch, num_classes], probabilities
            # log_probs = torch.log(outputs + 1e-8) # avoid log(0)
            # loss = criterion(log_probs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_train_loss = running_loss / total
        epoch_train_acc = running_correct / total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        # -------- VALIDATION --------
        model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = running_loss / total
        epoch_val_acc = running_correct / total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if scheduler:
            try:
                scheduler.step(epoch_val_loss)
            except TypeError:
                scheduler.step()

        improved = (epoch_val_acc - best_val_acc) > min_delta
        if improved:
            best_val_acc = epoch_val_acc
            best_state = deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        print(
            f" train loss: {epoch_train_loss:.4f} train_acc: {epoch_train_acc:.4f} "
            f"| val loss: {epoch_val_loss:.4f} val_acc: {epoch_val_acc:.4f}"
            + (" [improved]" if improved else "")
        )

        # Early stopping check
        if early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
            print(
                f"Early stopping triggered: no val_acc improvement > {min_delta:.4f} for {early_stop_patience} epochs."
            )
            break

    elapsed = time.time() - since
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # -------- TESTING --------
    if test_loader:
        model.load_state_dict(best_state)  # use the best model
        model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_loss = running_loss / total
        test_acc = running_correct / total
        print(f"Test loss: {test_loss:.4f} accuracy: {test_acc:.4f}")
        return history, best_state, (test_loss, test_acc)

    return history, best_state, (None, None)
