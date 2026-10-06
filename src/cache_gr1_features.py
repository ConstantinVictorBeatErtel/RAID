"""Removed legacy cache entry point; it never loads or downloads model assets."""
def main():
    raise SystemExit("This legacy cache command is permanently disabled; use src/cache_mae_features.py")
if __name__ == "__main__":
    main()
