import torch

def getSubsets(train_dataset, test_dataset, val_dataset):
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_subset_base, val_subset_base, test_subset_base = torch.utils.data.random_split(
        train_dataset, [0.7, 0.15, 0.15], generator=generator
    )

    # Extract indices from the splits
    train_indices = train_subset_base.indices
    val_indices = val_subset_base.indices
    test_indices = test_subset_base.indices

    # Apply the same indices to all datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    return train_subset, test_subset, val_dataset
