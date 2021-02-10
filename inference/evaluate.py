import re
import subprocess
from pathlib import Path

base_path = Path(
    r"data/experiment")

true = "prediction_FID_original"
compare = [
    "prediction_FID_original",
    "prediction_FID_output_No_M1",
    "prediction_FID_output_No_M2",
    "prediction_FID_output_No_Q",
    "prediction_FID_output_full_0.001",
    "prediction_FID_output_full_0.01",
    "prediction_FID_output_full_0.1",
    "prediction_FID_output_full_1",
]

FID_scores = {}
accuracies = {}
kappas = {}

for folder in compare:
    path_1 = str(base_path / true)
    path_2 = str(base_path / folder)

    r = re.compile(r'FID:.*?(\d+\.\d+)')
    response = subprocess.Popen(["python", "-m", "pytorch_fid", path_1, path_2,
                                 "--gpu=0"], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    fid = r.search(response)
    fid = fid.group(1) if fid else fid
    FID_scores[folder] = float(fid)

    response = subprocess.Popen([
        "python",
        r"PATH_TO_classifier.py",
        path_2
    ], stdout=subprocess.PIPE).communicate()[0].decode("utf-8")
    r = re.compile(r'Accuracy:.*?(\d+\.\d+)')
    acc = r.search(response)
    acc = acc.group(1) if acc else acc
    accuracies[folder] = 100*float(acc)

    r = re.compile(r'Kappa:.*?(\d+\.\d+)')
    ck = r.search(response)
    ck = ck.group(1) if ck else ck
    kappas[folder] = float(ck)

print({
    "FID": FID_scores,
    "accuracy": accuracies,
    "kappa": kappas
})
