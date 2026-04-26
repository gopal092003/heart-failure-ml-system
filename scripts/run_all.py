import subprocess


def run_command(cmd):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"❌ Failed: {cmd}")


if __name__ == "__main__":
    run_command("python scripts/train.py")
    run_command("python scripts/evaluate.py")
    run_command("python scripts/explain.py")

    print("\n✅ Full pipeline executed successfully!")