import time
import shutil
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
	for item in cache_dir.glob("*"):
		try:
			item_age_minutes = (now - item.stat().st_mtime) / 60
			if item_age_minutes > max_age_minutes:
				if item.is_file():
					item.unlink()
					if verbose:
						print(f"🗑️ Deleted file: {item}")
				elif item.is_dir():
					shutil.rmtree(item)
					if verbose:
						print(f"🗑️ Deleted directory: {item}")
		except Exception as e:
			print(f"⚠️ Could not delete {item}: {e}")