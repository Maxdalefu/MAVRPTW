import json
import sys
from ..utils.benchmarks import getMatrix

""" HVRP .TXT TO .JSON FORMATTER """

def hvrp_to_json(file):
    folder_Path = (
        "C:\\Users\\user\\Dev\\Fleet_Opt\\Benchmarks\\HFVRP\\"  # Your files folder
    )

    with open(f"{folder_Path}{file}", "r") as f:
        lines = f.readlines()

    start_Line = 5
    info = lines[start_Line].split()
    n_customers = int(info[0])
    n_vehicle_Types = int(info[1])

    # Vehicles: Number, capacity, fixed cost, cost per distance unit
    fleet = []
    for i in range(start_Line + 1, start_Line + n_vehicle_Types + 1):
        fleet.append(lines[i].split())
    vehicles = []
    vehicle_ID = 0
    for i in range(n_vehicle_Types):
        for j in range(int(fleet[i][0])):
            vehicle = {}
            vehicle_ID += 1
            vehicle["ID"] = vehicle_ID
            vehicle["Capacity"] = int(fleet[i][1])
            vehicle["var_Cost"] = float(fleet[i][3])
            vehicles.append(vehicle)

    # Deport Coordinates
    depot_Coordinates = lines[start_Line + n_vehicle_Types + 1].split()
    depot_Coordinates = [float(depot_Coordinates[0]), float(depot_Coordinates[1])]

    # Customers: Coordinate x, coordinate y, demand
    customers = []
    for i in range(
        start_Line + n_vehicle_Types + 2, start_Line + n_vehicle_Types + n_customers + 2
    ):
        customers.append(lines[i].split())

    coords = [[float(customer[0]), float(customer[1])] for customer in customers]
    coords.insert(0, depot_Coordinates)

    demands = [int(customers[i - 1][2]) if i > 0 else 0 for i in range(len(coords))]

    matrix = getMatrix(coords)

    json_data = {
        "demands": demands,
        "vehicles": vehicles,
        "matrix": matrix,
    }

    output_file_name = file[: file.rfind(".txt")] + ".json"
    print(f"Writing {file} to {output_file_name}")

    with open(f"{folder_Path}{output_file_name}", "w") as out:
        out.write(json.dumps(json_data, indent=4))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Missing input file")
    else:
        input_File = sys.argv[1]
        try:
            hvrp_to_json(input_File)
        except:
            print("Incorrect file")

# file = "HVRP13.txt"
# hvrp_to_json(file)
