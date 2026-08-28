from .synthetic import generate_dataset

if __name__ == "__main__":
    files = generate_dataset()
    print("SYNTHETIC data generated:")
    for name, path in files.items(): print(f"  {name}: {path}")
