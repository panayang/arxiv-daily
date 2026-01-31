import os
import shutil
import subprocess
from pathlib import Path

def build_and_gather():
    root_dir = Path.cwd()
    project_dir = root_dir / "web_app"
    release_dir = root_dir / "release_assets"
    
    env = os.environ.copy()
    env["LEPTOS_WASM_OPT_VERSION"] = "version_125"
    
    # List of items to find.
    to_gather = [
        "assets",
        "data",
        "target/site",
        "target/release/web_app",
        "config.toml"
    ]

    print(f"🚀 Entering {project_dir.name} and starting build...")
    
    try:
        subprocess.run(
            ["cargo", "leptos", "build", "--release"], 
            cwd=project_dir, 
            env=env,
            check=True
        )
        print("✅ Build successful.")

        if release_dir.exists():
            shutil.rmtree(release_dir)
        release_dir.mkdir(parents=True, exist_ok=True)

        print("📦 Gathering assets...")
        for item in to_gather:
            possible_paths = [project_dir / item, root_dir / item]
            
            found_path = next((p for p in possible_paths if p.exists()), None)
            
            if found_path:
                dest_path = release_dir / found_path.name
                
                # SPECIAL HANDLING: Only copy .bin files if the item is "assets"
                if found_path.name == "assets" and found_path.is_dir():
                    dest_path.mkdir(exist_ok=True)
                    for bin_file in found_path.glob("*.bin"):
                        shutil.copy2(bin_file, dest_path / bin_file.name)
                    print(f"  - [FILTERED] Copied only .bin files from {item}")
                
                # DEFAULT HANDLING: Copy everything else normally
                elif found_path.is_dir():
                    shutil.copytree(found_path, dest_path)
                    print(f"  - [FOUND] {item} (directory)")
                else:
                    shutil.copy2(found_path, dest_path)
                    print(f"  - [FOUND] {item} (file)")
            else:
                print(f"  - ⚠️  [MISSING] {item}")

        print(f"\n🎉 Release ready in: {release_dir}")

    except subprocess.CalledProcessError:
        print(f"❌ Error: Build failed.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    build_and_gather()