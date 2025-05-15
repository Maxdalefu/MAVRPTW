import json
from math import ceil
from ..utils.benchmarks import getMatrix

folder_Path = "C:\\Users\\user\\Dev\\Fleet_Opt\\Benchmarks\\CVRP\\A\\"
input_file = "A-n32-k5.vrp"

""" CVRP .VRP TO .JSON FORMATTER """

def cvrp_to_json(input_file):
    with open(f"{folder_Path}{input_file}", "r") as f:
        lines = f.readlines()

    line_num = 0
    demand = False
    coord = False
    demands = []
    vehicles = []
    coords = []

    while len(lines) > 0:
        line = lines.pop(0).strip()
        line_num += 1
        if "EOF" in line:
            break
        if "CAPACITY" in line.split():
            capacity = int(line.split()[2])
            continue
        if "NODE_COORD_SECTION" in line:
            coord = True
            continue
        if "DEMAND_SECTION" in line:
            demand = True
            coord = False
            continue
        if "DEPOT_SECTION" in line:
            break
        if coord:
            coord_str = line.split()[1:]
            coords.append([float(coord_str[0]), float(coord_str[1])])
        elif demand:
            demands.append(int(line.split()[1]))

    N = ceil(sum(demands) / capacity) + 1
    depot_pos = [coords[0]]

    for i in range(N):
        vehicles.append(
            {
                "id": i,
                "start": depot_pos,
                "end": depot_pos,
                "capacity": capacity,
            }
        )

    matrix = getMatrix(coords)
    json_file = {
        "fleet_Size": N,
        "vehicles": vehicles,
        "demands": demands,
        "matrix": matrix,
    }

    output_file_name = input_file[: input_file.rfind(".vrp")] + ".json"
    print(f"Writing {input_file} to {output_file_name}")

    with open(f"{folder_Path}{output_file_name}", "w") as out:
        out.write(json.dumps(json_file, indent=4))


if __name__ == "__main__":
    cvrp_to_json(input_file)
