import shutil
from pathlib import Path


def remove_specific_pycache(project_root: Path) -> None:
    targets = [
        project_root / "app" / "__pycache__",
        project_root / "app" / "core" / "__pycache__",
        project_root / "app" / "models" / "__pycache__",
        project_root / "app" / "routes" / "__pycache__",
        project_root / "app" / "services" / "__pycache__",
    ]

    for path in targets:
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
        else:
            print(f"Skip (not found): {path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    print("Cleaning selected __pycache__ folders...")
    remove_specific_pycache(project_root)
    print("Done.")
