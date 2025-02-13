import os
import re
import json
import shutil
import string
import pandas as pd
from datetime import datetime
import logging
import zipfile


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def sanitize_filename(filename):
    """Sanitizes a filename by keeping only valid characters."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return "".join(c for c in filename if c in valid_chars)


def find_latest_json(dir_path):
    """Finds the latest '_final_' JSON file in a directory."""
    final_jsons = [
        f for f in os.listdir(dir_path) if "_final_" in f and f.endswith(".json")
    ]
    latest_final_json = None
    latest_final_time = None

    for f in final_jsons:
        try:
            dt = datetime.strptime(
                f.split("_")[-2] + " " + f.split("_")[-1].split(".json")[0],
                "%Y%m%d %H%M%S",
            )
            if latest_final_time is None or dt > latest_final_time:
                latest_final_time = dt
                latest_final_json = f
        except ValueError as e:
            logging.error(f"Error parsing filename {f}: {e}")
    return latest_final_json

def process_article(output_dir, final_dir, command_dir, topic, category):
    """Processes a single article directory."""
    sanitized_topic = re.sub(r"[^\w\s]", "", topic).replace(" ", "_")
    dir_path = os.path.join(output_dir, sanitized_topic)
    if not os.path.exists(dir_path):
        logging.info(f"Skipping. Does not exist: {dir_path}")
        return None

    latest_final_json = find_latest_json(dir_path)
    if not latest_final_json:
        logging.info(f"No valid JSON found for {topic}")
        return None

    try:
        source_path = os.path.join(dir_path, latest_final_json)
        # Validate JSON by loading it
        with open(source_path, "r") as f:
            json_data = json.load(f)

        # Sanitize the filename
        safe_filename = sanitize_filename(latest_final_json)
        destination_path = os.path.join(final_dir, safe_filename) 
        shutil.copy(source_path, destination_path)

        command_path = os.path.join(command_dir, safe_filename)
        command = f'python manage.py import_articles --category-name "{category}" --json-path "{command_path}"'
        return command
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in {latest_final_json}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error on {latest_final_json}: {e}")
        return None

def zip_directory(folder_path, zip_path):
    """Creates a zip archive of a directory, preserving the folder structure."""
    base_dir = os.path.basename(folder_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(
                    base_dir, os.path.relpath(file_path, folder_path)
                )
                zipf.write(file_path, arcname=arcname)
    logging.info(f"Successfully zipped '{folder_path}' to '{zip_path}'")

def main():
    """Main processing function."""
    output_dir = "/Users/vince/Salk/PaperGeneration/data/output/gemini-2.0-flash-exp"
    data_dir = "/Users/vince/Salk/PaperGeneration/data/"
    final_dir = os.path.join(data_dir, "final_json")
    command_dir = "/home/vrothenberg_salk_edu/wagtail/data/final_json"
    zip_file_path = os.path.join(data_dir, "final_json.zip")
    os.makedirs(final_dir, exist_ok=True)
    commands = []

    df = pd.read_csv("/Users/vince/Salk/PaperGeneration/data/condition_revised.csv").fillna("")

    for i, row in df.iterrows():
        condition_name = row["Condition"]
        alternative_name = row["Alternative Name"]
        category = row["Category"]
        topic = condition_name
        if alternative_name:
            topic = f"{topic} ({alternative_name})"
        logging.info(f"[{i}] {condition_name} - {category}")
        command = process_article(
            output_dir, final_dir, command_dir, topic, category
        )
        if command:
            commands.append(command)

    for command in commands:
        logging.info(command)

    command_file_path = os.path.join(data_dir, "import_commands.sh")

    with open(command_file_path, "w") as f:
        for command in commands:
            f.write(command + "\n")

    zip_directory(final_dir, zip_file_path)


if __name__ == "__main__":
    main()