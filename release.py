import os
import shutil
import subprocess
from pathlib import Path

def build_and_gather():
    # 1. Configuration
    # Running from root, but cargo command runs inside web_app
    root_dir = Path.cwd()
    project_dir = root_dir / "web_app"
    release_dir = root_dir / "release_assets"
    
    env = os.environ.copy()
    env["LEPTOS_WASM_OPT_VERSION"] = "version_125"
    
    # List of items to find. We will search project_dir and root_dir.
    to_gather = [
        "assets",
        "data",
        "target/site",
        "target/release/web_app",
        "config.toml"
    ]

    print(f"🚀 Entering {project_dir.name} and starting build...")
    
    try:
        # 2. Run Build
        subprocess.run(
            ["cargo", "leptos", "build", "--release"], 
            cwd=project_dir, 
            env=env,
            check=True
        )
        print("✅ Build successful.")

        # 3. Fresh Start for Release Folder
        if release_dir.exists():
            shutil.rmtree(release_dir)
        release_dir.mkdir(parents=True, exist_ok=True)

        # 4. Smart Copy Logic
        print("📦 Gathering assets...")
        for item in to_gather:
            # Check project subfolder first, then root
            possible_paths = [
                project_dir / item,
                root_dir / item
            ]
            
            found = False
            for path in possible_paths:
                if path.exists():
                    dest_path = release_dir / path.name
                    if path.is_dir():
                        shutil.copytree(path, dest_path)
                    else:
                        shutil.copy2(path, dest_path)
                    print(f"  - [FOUND] {item} at {path.relative_to(root_dir)}")
                    found = True
                    break # Move to next item once found
            
            if not found:
                print(f"  - ⚠️  [MISSING] {item} (checked subfolder and root)")

        print(f"\n🎉 Release ready in: {release_dir}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Build failed.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    build_and_gather()