import json, random, os

from src.utils.benchmarks import getMatrix


""" THIS FILE IS FOR COMPLETELY RANDOM INSTANCE GENERATION """

def genInstance(seed_Num, n_Jobs):
    random.seed(seed_Num)

    # n_Jobs = 300
    depot_Coords = [40, 40]
    depot_TW = [[28800, 64800]]  # 8:00, 18:00
    fleet_Comp = {"1000": 25, "2000": 25, "5000": 25}
    capacities = list(fleet_Comp.keys())

    coords = [
        [random.randint(0, 100), random.randint(0, 100)] if i > 0 else depot_Coords
        for i in range(n_Jobs + 1)
    ]

    demands = [random.randint(1, 10) * 10 if i > 0 else 0 for i in range(n_Jobs + 1)]

    jobs_Skills = [
        [random.randint(0, len(capacities) - 1)] if i > 0 else []
        for i in range(n_Jobs + 1)
    ]  # We assign a random skill for each job, with range 1 to number of capacities

    service_Time = 300  # 300 [s] = 5 [min]

    tw_start1_options = [21600, 28800, 36000]  # 6:00, 8:00, 10:00
    tw_end1_options = [43200, 46800, 50400]  # 12:00, 13:00, 14:00
    tw_start2_options = [46800, 50400, 54000]  # 13:00, 14:00, 15:00
    tw_end2_options = [64800, 68400, 72000]  # 18:00, 17:00, 20:00
    time_Windows = [
        [
            [
                tw_start1_options[random.randint(0, 2)],
                tw_end1_options[random.randint(0, 2)],
            ],
            [
                tw_start2_options[random.randint(0, 2)],
                tw_end2_options[random.randint(0, 2)],
            ],
        ]
        if i > 0
        else depot_TW
        for i in range(n_Jobs + 1)
    ]

    matrix = getMatrix(coords)

    jobs = []
    for i in range(n_Jobs + 1):
        jobs.append(
            {
                "job_id": int(i),
                "location": coords[i],
                "load": int(demands[i]),
                "time_windows": time_Windows[i],
                "skills": jobs_Skills[i],
                "service": int(service_Time),
            }
        )

    json_Data = {"fleet_Comp": fleet_Comp, "jobs": jobs, "matrix": matrix}

    return json_Data


def main():
    for n in range(1):
        n_Jobs = 300
        seed_Num = n
        json_Data = genInstance(seed_Num, n_Jobs)

        cwd = os.getcwd()
        folder_Path = cwd + "\\Benchmarks\\Generated_Instances\\"  # , VRPTW\\solomon
        try:
            os.mkdir(f"{folder_Path}")
        except:
            pass

        name_Number = str(n_Jobs + seed_Num)

        output_file_name = name_Number + ".json"
        print(f"Writing {output_file_name}")

        with open(f"{folder_Path}{output_file_name}", "w") as out:
            out.write(json.dumps(json_Data, indent=4))
    return None


if __name__ == "__main__":
    main()
