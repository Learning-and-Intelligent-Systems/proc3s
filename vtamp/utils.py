import hashlib
import os
import pathlib
import threading

import hydra
import pybullet as p
import pystache


def get_log_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg["runtime"]["output_dir"]


def save_log(path, text):
    with open(os.path.join(get_log_dir(), path), "w") as f:
        f.write(text)


def get_prompt_element_map():

    folder_path = os.path.join(
        pathlib.Path(__file__).parent, "policies/prompt_elements"
    )
    file_text_map = {}

    # Iterate over all the files in the specified folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Ensure it's a file (not a directory)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                file_text_map[file_name.replace(".txt", "")] = text

    return file_text_map


def parse_text_prompt(text_prompt_path):
    entries = []
    current_role = None
    current_content = []
    prompt_elements = get_prompt_element_map()

    with open(text_prompt_path, "r") as file:
        template = file.read()
        rendered = pystache.render(template, prompt_elements)
        for line in rendered.split("\n"):
            if line.startswith("#define"):
                if current_role is not None:
                    # Join all content lines and strip to clean up whitespace
                    content = "".join(current_content)
                    entries.append({"role": current_role, "content": content})
                    current_content = []
                current_role = line.split()[-1]  # Get the last word, which is the role
            else:
                current_content.append(line + "\n")

        # Don't forget to add the last entry if there is one
        if current_role is not None and current_content:
            content = "".join(current_content)
            entries.append({"role": current_role, "content": content})

    return entries


def write_prompt(path, entries):
    with open(os.path.join(get_log_dir(), path), "w") as file:
        for entry in entries:
            # Write the role definition
            file.write(f"#define {entry['role']}\n")
            # Write the content, each line is separated
            content_lines = entry["content"]

            file.write(content_lines + "\n")


def threaded_input(*args, **kwargs):
    data = []
    thread = threading.Thread(
        target=lambda: data.append(input(*args, **kwargs)), args=[]
    )
    thread.start()
    try:
        while thread.is_alive():
            p.getMouseEvents()
    finally:
        thread.join()
    return data[-1]


def get_previous_log_folder(base_dir):
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    subdirs = [
        s for s in subdirs if os.path.realpath(s) != os.path.realpath(get_log_dir())
    ]
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    most_recent_dir = subdirs[0] if subdirs else None
    return most_recent_dir


def file_hash(filename):
    hash_algo = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()


def read_file(filename):
    with open(filename, "rb") as f:
        content = f.read().decode("utf-8")
    return content


def are_files_identical(file1, file2):
    return file_hash(file1) == file_hash(file2)
