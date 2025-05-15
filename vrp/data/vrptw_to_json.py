import json
import os
from benchmarks import getMatrix


""" VRPTW .TXT TO .JSON FORMATTER """

def vrp_To_Json(folder_Path, input_file):

    with open(f"{folder_Path}{input_file}", "r") as f:
        lines = f.readlines()

    line_num = 0
    fleet_Composition = {}
    jobs = []
    vehicles = []
    coords = []

    while len(lines) > 0:
        line = lines.pop(0).strip()
        line_num += 1
        if line_num == 5:
            fleet_Composition[f"{line.split()[1]}"] = f"{line.split()[0]}"
            continue
        if line_num < 10:
            continue
        else:
            job = line.split()
            coords.append([float(job[1]), float(job[2])])
            jobs.append(
                {
                    "job_id": int(job[0]),
                    "location": [float(job[1]), float(job[2])],
                    "load": int(job[3]),
                    "time_windows": [[int(job[4]), int(job[5])]],
                    "service": int(job[6]),
                }
            )

    capacities = list(fleet_Composition.keys())

    for i in capacities:
        n_vehicles = int(fleet_Composition[i])
        for j in range(n_vehicles):
            vehicles.append(
                {
                    "id": j,
                    "start": coords[0],
                    "end": coords[0],
                    "capacity": [int(i)],
                    "time_window": jobs[0]["time_windows"][0],
                }
            )

    matrix = getMatrix(coords)
    json_file = {
        "fleet_Comp": fleet_Composition,
        "vehicles": vehicles,
        "jobs": jobs,
        "matrix": matrix,
    }

    output_file_name = input_file[: input_file.rfind(".txt")] + ".json"
    print(f"Writing {input_file} to {output_file_name}")

    with open(f"{folder_Path}{output_file_name}", "w") as out:
        out.write(json.dumps(json_file, indent=4))


def main():
    cwd = os.getcwd()
    folder_Path = cwd + "\\Benchmarks\\VRPTW\\solomon\\"
    for file in os.listdir(folder_Path):
        vrp_To_Json(folder_Path, file)


if __name__ == "__main__":
    main()
