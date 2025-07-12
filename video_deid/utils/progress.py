"""
Progress bar utilities
"""
from tqdm import tqdm


def create_progress_bar(total, desc, show_progress):
    """
    Creates a progress bar using the tqdm library.

    Parameters:
    - total (int): Total number of items for the progress bar
    - desc (str): Description for the progress bar
    - show_progress (bool): Whether to show the progress bar

    Returns:
    - tqdm or None: A tqdm progress bar instance if show_progress is True, None otherwise
    """
    return tqdm(total=total, desc=desc, ncols=100) if show_progress else None
