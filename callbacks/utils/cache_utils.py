import time
from pathlib import Path

def clear_old_cache_files(cache_dir, max_age_minutes=60, verbose=True):
    """
    Delete files older than `max_age_minutes` from a cache directory.
    
    Parameters:
        cache_dir (str or Path): Path to the cache directory.
        max_age_minutes (int): Age in minutes above which files are deleted.
        verbose (bool): Whether to print info about deleted files.
    """
    now = time.time()
    cache_dir = Path(cache_dir)
    for file in cache_dir.glob("*"):
        if file.is_file():
            file_age_minutes = (now - file.stat().st_mtime) / 60
            if file_age_minutes > max_age_minutes:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"⚠️ Could not delete {file}: {e}")
    for parquet_file in cache_dir.glob("*.parquet"):
        if parquet_file.is_file():
            file_age_minutes = (now - parquet_file.stat().st_mtime) / 60
            if file_age_minutes > max_age_minutes:
                try:
                    parquet_file.unlink()  # Delete the file
                except Exception as e:
                    print(f"⚠️ Could not clean or delete {parquet_file}: {e}")