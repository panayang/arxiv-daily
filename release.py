import os
import shutil
import subprocess
from pathlib import Path

def build_and_gather():
    # Configuration
    project_dir = "web_app"
    release_dir = Path("release_assets")
    
    # Files/Dirs to copy (Source -> Destination name)
    to_copy = [
        ("assets", "assets"),
        ("data", "data"),
        ("target/site", "site"),
        (f"target/release/{project_dir}", project_dir),
        ("config.toml", "config.toml")
    ]

    print(f"🚀 Entering {project_dir} and starting build...")
    
    try:
        # 1. Run cargo leptos build --release
        subprocess.run(
            ["cargo", "leptos", "build", "--release"], 
            cwd=project_dir, 
            check=True
        )
        print("✅ Build successful.")

        # 2. Create release_assets folder
        if release_dir.exists():
            print(f"🧹 Cleaning existing {release_dir}...")
            shutil.rmtree(release_dir)
        release_dir.mkdir(parents=True, exist_ok=True)

        # 3. Copy files
        print("📦 Gathering assets...")
        for src_path, dest_name in to_copy:
            full_src = Path(project_dir) / src_path
            full_dest = release_dir / dest_name

            if full_src.exists():
                if full_src.is_dir():
                    shutil.copytree(full_src, full_dest)
                else:
                    shutil.copy2(full_src, full_dest)
                print(f"  - Copied: {src_path}")
            else:
                print(f"  - ⚠️ Warning: {src_path} not found. Skipping.")

        print(f"\n🎉 Release ready in: {release_dir.absolute()}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Build failed with exit code {e.returncode}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    build_and_gather()
