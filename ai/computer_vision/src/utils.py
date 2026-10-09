from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_image_file(file_path):
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def create_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_image_files(directory):
    directory = Path(directory)

    if not directory.exists():
        return []

    return [
        file
        for file in directory.iterdir()
        if file.is_file() and is_image_file(file)
    ]