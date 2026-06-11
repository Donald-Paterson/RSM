import os

from validator import validate
from config import TEST_FOLDER
from report_generator import generate_report


def main():

    all_results = []

    images = os.listdir(TEST_FOLDER)

    for img in images:

        path = os.path.join(TEST_FOLDER, img)

        print("\nProcessing:", img)

        results = validate(path)

        all_results.extend(results)

    generate_report(
        all_results,
        output_folder=r"C:\Users\Donald Paterson\Documents\Donald_Paterson-AI_Engineer\RSM\outputs\Reports"
    )


if __name__ == "__main__":

    main()