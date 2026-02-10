import os
import shutil


def rename_sampling_folders(root_dir=r".", name='knn_api'):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if dir_name.startswith(f"{name}-"):
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, f"{name}")
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            elif dir_name.startswith(f"{name}."):
                old_path = os.path.join(root, dir_name)
                shutil.rmtree(old_path)
            elif dir_name.startswith("build"):
                old_path = os.path.join(root, dir_name)
                shutil.rmtree(old_path)
            elif dir_name.startswith("dist"):
                old_path = os.path.join(root, dir_name)
                shutil.rmtree(old_path)


if __name__ == "__main__":
    rename_sampling_folders()
