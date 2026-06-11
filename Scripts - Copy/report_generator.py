import pandas as pd
from datetime import datetime
import os
from openpyxl.drawing.image import Image


def generate_report(results, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = os.path.join(
        output_folder,
        f"RSM_Inspection_Report_{timestamp}.xlsx"
    )

    df = pd.DataFrame(results)

    df.to_excel(file_path, index=False)

    from openpyxl import load_workbook

    wb = load_workbook(file_path)
    ws = wb.active

    for i, row in df.iterrows():

        r = i + 2

        try:

            g_img = Image(row["Golden ROI"])
            t_img = Image(row["Test ROI"])
            d_img = Image(row["Diff Map"])

            g_img.width = 60
            g_img.height = 60

            t_img.width = 60
            t_img.height = 60

            d_img.width = 60
            d_img.height = 60

            ws.add_image(g_img, f"H{r}")
            ws.add_image(t_img, f"I{r}")
            ws.add_image(d_img, f"J{r}")

        except:
            pass

    wb.save(file_path)

    print(f"\nInspection report saved:\n{file_path}")