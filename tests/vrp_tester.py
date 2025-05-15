import json, os, concurrent.futures, cProfile, pstats
from datetime import date
from pathlib import Path
from pprint import pprint
from time import time

from src.Ruin_and_Recreat import ruinAndRecreate
from src.CROSS_exchange import crossEx
from src.MACYW import genRoutes
from src.solution import Instance, Sol
from src.utils.folder import Folder


folder = Folder()
TODAY = date.today()


"""C: Clustered, R: Random, RC: Mixed, 1: short scheduling horizon, 2: long scheduling horizon"""
""" No esta actualizada """


def seeResults(problem: str) -> None:

    RESULTS_PATH = folder.benchmarks + f"{problem}\\Results\\"

    for index, file in enumerate(os.listdir(RESULTS_PATH)):

        file_Name, file_Extension = os.path.splitext(file)
        if file_Extension == ".json":

            with open(f"{RESULTS_PATH}{file}", "r") as f:
                data = json.load(f)

            print(
                f"File: {file_Name}, Unassigned Jobs: {data['Results']['Solution']['Unassigned_Jobs']}"
            )


def localSearch(solution: Sol, t1: float) -> Sol:

    for _ in range(10):
        initial_Cost = solution.total_Cost
        solution = crossEx(solution, L_Max=5)
        solution = ruinAndRecreate(solution, relax="shaw")
        if solution.total_Cost == initial_Cost:
            break
    return solution


""" Instance arguments:
data: dict = None,
file: str = None,
fleet: dict = None,
speed_Factor: float = 1,
max_Jobs_Per_Route: float = float("inf")
"""


def solveVRP(args: tuple) -> Sol:
    (
        file_Path,
        output_Path,
        fleet,
        speed_Factor,
        max_Jobs_Per_Route,
    ) = args

    file_Name = file_Path.stem
    # Se carga la informacion del archivo
    with file_Path.open() as f:
        data = json.load(f)
    t1 = time()

    # Generamos una solucion inicial
    inst = Instance(
        data=data,
        file=file_Name,
        fleet=fleet,
        speed_Factor=speed_Factor,
        max_Jobs_Per_Route=max_Jobs_Per_Route,
    )
    # print(f"Solving instance: {file_Name:10} | Fleet: {fleet} | N_jobs: {inst.n_Jobs}")
    # print("Instance created")
    solution = genRoutes(inst)
    # print("Time: " + str(round(time() - t1)) + "[s]")

    # Comenzamos con la busqueda local
    solution = crossEx(solution, L_Max=5)
    # print("Time: " + str(round(time() - t1)) + "[s]")

    solution = ruinAndRecreate(solution, relax="route")
    # print("Time: " + str(round(time() - t1)) + "[s]")

    # (Cross-X + RandR with shaw removal) x10
    solution = localSearch(solution, t1)
    # solution.showRoutes()
    solution.reduceVehicleCapacity()
    # if modified:
    # solution = localSearch(solution, t1)
    # solution.reduceVehicleCapacity()
    # print("#" * 40 + f"\nSearch completed")

    # Se revisa la solucion final
    solution.checkSolution()
    solution.postProcess()
    processing_Time = str(round(time() - t1))
    print(
        f"{file_Name:^15} | {solution.total_Cost:15.2f} | {sum(solution.unassigned_Jobs):15} | {processing_Time:10} | {solution.used_Fleet}"
    )

    # Se guardan los resultados
    json_Data = {
        "File": file_Name,
        "Summary": solution.summary(),
        "Routes": solution.routes_Dict_List,
        "P_Time": processing_Time,
    }
    output_Path.write_text(json.dumps(json_Data, indent=4))

    return solution


"""
Procesa todos los archivos dentro de la carpeta dada como input, si especifica archivos(file_Paths) solo se procesarán esos
"""


def multiProcessData(
    input_Folder_Path: Path,
    output_Folder_Path: Path,
    fleet: dict,
    speed_Factor: float = 1,
    max_Jobs_Per_Route: float = float("inf"),
    file_Paths: list = [],
) -> None:

    # Creamos la carpeta de output si no existe
    output_Folder_Path.mkdir(parents=True, exist_ok=True)
    # Procesamos todos los archivos
    print(
        f"{'Instance':^15} | {'Total Cost':^15} | {'Unassigned Jobs':^15} | {'P time [s]':^10} | {'Used Fleet':^15}"
    )
    # Se guardan todos los archivos a procesar en una lista
    args = []
    if file_Paths:
        for file_Path in file_Paths:
            if file_Path.is_file() and file_Path.suffix == ".json":
                output_Path = output_Folder_Path / file_Path.name
                args.append(
                    (
                        file_Path,
                        output_Path,
                        fleet,
                        speed_Factor,
                        max_Jobs_Per_Route,
                    )
                )
    else:
        for file_Path in input_Folder_Path.iterdir():
            if file_Path.is_file() and file_Path.suffix == ".json":
                output_Path = output_Folder_Path / file_Path.name
                args.append(
                    (
                        file_Path,
                        output_Path,
                        fleet,
                        speed_Factor,
                        max_Jobs_Per_Route,
                    )
                )
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(solveVRP, args)


def main():

    input_Folder_Path = folder.solomon_instances
    fleet = None  # {"1000": 25, "3000": 25, "5000": 25}
    output_Folder_Path = folder.solomon_results
    multiProcessData(input_Folder_Path, output_Folder_Path, fleet)

    # file = "2022-07-25"  # "2022-07-25"
    # fleet = {"1000": 25, "3000": 25, "5000": 25}
    # speed_Factor = 0.28  # 1
    # max_Jobs_Per_Route = 33  # float("inf")
    # file_Path = folder.empresa / "Original" / f"{file}.json"
    # output_Path = folder.empresa / "Original" / "Results" / f"{file}.json"
    # print(
    #     f"{'Instance':^15} | {'Total Cost':^15} | {'Unassigned Jobs':^15} | {'P time [s]':^10} | {'Used Fleet':^15}"
    # )
    # solution = solveVRP(
    #     args=(file_Path, output_Path, fleet, speed_Factor, max_Jobs_Per_Route)
    # )
    # solution.showRoutes()

    # with cProfile.Profile() as pr:
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10)


if __name__ == "__main__":
    main()
