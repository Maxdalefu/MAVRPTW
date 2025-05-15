import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from datetime import date
from pathlib import Path

from src.utils.folder import Folder

folder = Folder()
TODAY = date.today()


"""Generic function to plot routes"""
def plotRoutes(routes, jobs):
    """
    "job_id": int(i),
    "location": coords[i],
    "load": int(demands[i]),
    "time_windows": time_Windows[i],
    "skills": jobs_Skills[i],
    "service": int(service_Time),
    """
    # print(plt.style.available)
    plt.style.use("seaborn-whitegrid")
    colors = plt.cm.get_cmap("tab20", 20)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.grid(True)
    plt.title("Routes")
    plt.xlabel("x_coord")
    plt.ylabel("y_coord")
    plt.tight_layout(pad=0.5)
    # fig = plt.figure()
    # ax = plt.axes()

    # Plot Depot
    depot_Pos = jobs[0]["location"]
    plt.plot(depot_Pos[0], depot_Pos[1], color="black", marker="s")  #

    # Plot nodes
    nodes_Pos_X = [jobs[i]["location"][0] for i in range(1, len(jobs))]
    nodes_Pos_Y = [jobs[i]["location"][1] for i in range(1, len(jobs))]
    plt.scatter(nodes_Pos_X, nodes_Pos_Y, color="black", marker=".")

    # Plot Routes
    for i, route in enumerate(routes):
        route_Pos_X = [
            nodes_Pos_X[route[i - 1] - 1]
            if i > 0 and i < len(route) + 1
            else depot_Pos[0]
            for i in range(len(route) + 2)
        ]
        route_Pos_Y = [
            nodes_Pos_Y[route[i - 1] - 1]
            if i > 0 and i < len(route) + 1
            else depot_Pos[1]
            for i in range(len(route) + 2)
        ]
        plt.plot(
            route_Pos_X,
            route_Pos_Y,
            color=colors(i),
            marker=".",
            linewidth=1,
            label="Route " + str(i + 1),
        )  # linestyle='--',
        plt.pause(0.3)

    plt.legend()  # loc='upper right'
    plt.show()


def plotNodes(file, problem):

    plt.style.use("seaborn")
    plt.figure(figsize=(10, 9))
    title = f"Nodes"
    plt.title(title)
    plt.xlabel("x_coord")
    plt.ylabel("y_coord")
    plt.tight_layout(pad=0.5)
    file_Path = DATA_PATH + f"{problem}\\{file}.json"
    with open(f"{file_Path}", "r") as g:
        data = json.load(g)

    jobs = data["jobs"]
    # Plot Depot
    depot_Pos = jobs[0]["location"]
    plt.scatter(depot_Pos[0], depot_Pos[1], color="black", marker="s")  #

    # Plot nodes
    nodes_Pos_X = [jobs[i]["location"][0] for i in range(1, len(jobs))]
    nodes_Pos_Y = [jobs[i]["location"][1] for i in range(1, len(jobs))]
    plt.scatter(nodes_Pos_X, nodes_Pos_Y, color="b", marker="*")
    plt.show()


def plotProblemNodes(problem):

    plots_Path = folder.PLOTS_SOLOMON_DIR / "Solomon_Nodes"
    try:
        os.mkdir(plots_Path)
    except:
        pass

    # problem_Path = DATA_PATH + f"{problem}\\"
    for index, file in enumerate(os.listdir(problem_Path)):
        file_Name, file_Extension = os.path.splitext(file)

        if file_Extension == ".json":

            file_Path = problem_Path + f"{file}"
            with open(f"{file_Path}", "r") as g:
                data = json.load(g)

            # Make plot
            plt.style.use("seaborn")
            plt.figure(figsize=(10, 9))
            title = f"Nodes position in {file_Name} instance"
            plt.title(title)
            plt.xlabel("x_coord")
            plt.ylabel("y_coord")
            plt.tight_layout(pad=0.5)
            jobs = data["jobs"]
            # Plot Depot
            depot_Pos = jobs[0]["location"]
            plt.scatter(depot_Pos[0], depot_Pos[1], color="black", marker="s")  #

            # Plot nodes
            nodes_Pos_X = [jobs[i]["location"][0] for i in range(1, len(jobs))]
            nodes_Pos_Y = [jobs[i]["location"][1] for i in range(1, len(jobs))]
            plt.scatter(nodes_Pos_X, nodes_Pos_Y, color="b", marker="*")
            plt.savefig(
                f"{plots_Path}\\{file_Name}.png",
                bbox_inches="tight",
                dpi=500,
            )


def loadSolomonResults():

    solomon_Results_Path = folder.SOLOMON_RESULTS_DIR
    results = list()
    for file_Path in solomon_Results_Path.iterdir():

        with file_Path.open() as f:
            solution = json.load(f)
        results.append(
            {
                "File": solution["File"],
                "Total_Cost": solution["Summary"]["Total_Cost"],
                "Used_Fleet": solution["Summary"]["Used_Fleet"],
                "Unassigned_Jobs": solution["Summary"]["Unassigned_Jobs"],
                "P_Time": solution["P_Time"],
            }
        )

    return results


def plotPTimes(plots_Folder_Path, save):

    results = loadSolomonResults()
    x = list()
    Heuristic_Times = list()

    for result in results:
        x.append(str(result["File"]))
        Heuristic_Times.append(int(result["P_Time"]))

    mean_Time = round(sum(Heuristic_Times) / len(Heuristic_Times), 1)
    mean_Times = [mean_Time for i in x]

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))
    title = "Tiempo de procesamiento de la heurística"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(rotation=65, fontsize=7)
    # plt.yticks(list(range(0, int(max(Heuristic_Times)) + 2, 2)))
    plt.ylabel("Tiempo [s]")
    plt.tight_layout(pad=0.5)

    plt.plot(x, Heuristic_Times, color="b", label="Tiempo de cada instancia")
    plt.plot(
        x,
        mean_Times,
        color="g",
        label=f"Tiempo medio ({mean_Time})",
        linestyle="dashed",
    )
    plt.legend(fontsize="small")  # loc='upper right'
    if save:
        plot_Path = plots_Folder_Path / f"{title}.png"
        plt.savefig(
            plot_Path,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotCostGaps(plots_Folder_Path, save):

    with folder.BKS.open() as f:
        BKS_Data = json.load(f)

    results = loadSolomonResults()

    x = list()
    Heuristic_Gaps = list()

    for result in results:
        x.append(str(result["File"]))
        Heuristic_cost = result["Total_Cost"]
        BKS_cost = BKS_Data[str(result["File"]) + "_distance"]["best_known_cost"]
        Heuristic_Gaps.append((Heuristic_cost - BKS_cost) * 100 / BKS_cost)

    mean_Gap = round(sum(Heuristic_Gaps) / len(Heuristic_Gaps), 1)
    mean_Gaps = [mean_Gap for i in x]

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))
    title = "Diferencia porcentual costo total con BKS"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(rotation=65, fontsize=7)
    plt.yticks(list(range(0, int(max(Heuristic_Gaps)) + 2, 2)))
    plt.ylabel("Diferencia [%]")
    plt.tight_layout(pad=0.5)

    plt.plot(x, Heuristic_Gaps, color="b", label="Diferencia por instancia")
    plt.plot(
        x,
        mean_Gaps,
        color="g",
        label=f"Diferencia media ({mean_Gap})",
        linestyle="dashed",
    )
    plt.legend(fontsize="small")  # loc='upper right'
    if save:
        plot_Path = plots_Folder_Path / f"{title}.png"
        plt.savefig(
            plot_Path,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotBKSCostComparison(plots_Folder_Path, save):

    # Cargamos la informacion
    with folder.BKS.open() as f:
        BKS_Data = json.load(f)
    results = loadSolomonResults()

    x = list()
    Heuristic_Costs = list()
    BKS_Distance_Costs = list()
    for result in results:
        x.append(str(result["File"]))
        Heuristic_Costs.append(result["Total_Cost"])
        BKS_Distance_Costs.append(
            BKS_Data[str(result["File"]) + "_distance"]["best_known_cost"]
        )

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))

    title = f"Comparación con BKS en costo total"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(rotation=65, fontsize=7)
    plt.ylabel("Costo total")
    plt.tight_layout(pad=0.5)

    plt.bar(x, Heuristic_Costs, color="g", label="Heuristica")
    plt.bar(x, BKS_Distance_Costs, color="b", label="BKS")
    plt.legend(fontsize="small")  # loc='upper right'
    if save:
        plot_Path = plots_Folder_Path / f"{title}.png"
        plt.savefig(
            plot_Path,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotFleetDifference(plots_Folder_Path, save):

    with folder.BKS.open() as f:
        BKS_Data = json.load(f)

    results = loadSolomonResults()

    x = list()
    Fleets_Difference = list()

    for result in results:
        x.append(str(result["File"]))
        Heuristic_Vehicles = sum(result["Used_Fleet"].values())
        BKS_Vehicles_Fleet = int(BKS_Data[str(result["File"])]["solved_with_vehicles"])
        Fleets_Difference.append(Heuristic_Vehicles - BKS_Vehicles_Fleet)

    mean_Diff = round(sum(Fleets_Difference) / len(Fleets_Difference), 1)
    mean_Diffs = [mean_Diff for i in x]

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))
    title = "Diferencia de flota utilizada con BKS"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(rotation=65, fontsize=7)
    plt.yticks(list(range(0, max(Fleets_Difference) + 1)))
    plt.ylabel("Diferencia de flota urtilizada")
    plt.tight_layout(pad=0.5)
    plt.plot(x, Fleets_Difference, color="b", label="Diferencia")
    plt.plot(
        x,
        mean_Diffs,
        color="g",
        label=f"Diferencia media ({mean_Diff})",
        linestyle="dashed",
    )
    plt.legend(fontsize="small")  # loc='upper right'
    if save:
        plot_Path = plots_Folder_Path / f"{title}.png"
        plt.savefig(
            plot_Path,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotUsedVehiclesComparison(plots_Folder_Path, save=False):

    with folder.BKS.open() as f:
        BKS_Data = json.load(f)

    results = loadSolomonResults()
    x = list()
    Heuristic_Vehicles = list()
    BKS_Vehicles_Fleet = list()

    for result in results:
        x.append(str(result["File"]))
        Heuristic_Vehicles.append(sum(result["Used_Fleet"].values()))
        BKS_Vehicles_Fleet.append(
            int(BKS_Data[str(result["File"])]["solved_with_vehicles"])
        )

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))
    title = "Comparación de flota utilizada con BKS"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(rotation=65, fontsize=7)
    plt.yticks(list(range(1, max(Heuristic_Vehicles) + 1)))
    plt.ylabel("Flota utilizada")
    plt.tight_layout(pad=0.5)
    plt.bar(x, Heuristic_Vehicles, color="g", label="Heuristica")
    plt.bar(x, BKS_Vehicles_Fleet, color="b", label="BKS")
    plt.legend(fontsize="small")  # loc='upper right'
    if save:
        plot_Path = plots_Folder_Path / f"{title}.png"
        plt.savefig(
            plot_Path,
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


""" Plot heuristic results on solomon instances """
def plotSolomonResults(save=False):

    plots_Folder_Path = folder.PLOTS_SOLOMON_DIR / f"{TODAY}"
    plots_Folder_Path.mkdir(parents=True, exist_ok=True)

    plotBKSCostComparison(plots_Folder_Path, save)
    plotCostGaps(plots_Folder_Path, save=save)
    plotUsedVehiclesComparison(plots_Folder_Path, save)
    plotFleetDifference(plots_Folder_Path, save)
    plotPTimes(plots_Folder_Path, save)


def main():
    # Plots de resultados de instancias de solomon
    plotSolomonResults(save=False)


if __name__ == "__main__":
    main()