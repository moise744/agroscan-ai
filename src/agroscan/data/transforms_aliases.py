

# ---------------------------------------------------------------------------
# Compatibility aliases used by notebooks
# ---------------------------------------------------------------------------
def get_train_transforms(image_size: int = 224):
    """Alias → get_agri_transforms(image_size, is_training=True)."""
    return get_agri_transforms(image_size, is_training=True)


def get_val_transforms(image_size: int = 224):
    """Alias → get_agri_transforms(image_size, is_training=False)."""
    return get_agri_transforms(image_size, is_training=False)
