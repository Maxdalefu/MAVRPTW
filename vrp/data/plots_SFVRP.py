import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json, os
from datetime import date
from pathlib import Path
from utils.folder import Folder

folder = Folder()
TODAY = date.today()
# FOLDER_PATH = str(Path(__file__).resolve().parent.parent)
DATA_PATH = folder.DATA_DIR


""" RESULT PLOTS """
""" Calcula el costo total correspondiente a toda la flota utilizada en un período """
def getTotalFleetCost(
    fixed_Fleet: dict, total_Var_Fleet: dict, n_Days: int, dsct_Factor: float
) -> float:

    # fixed_Vehicle_Costs = {"1000": 1_500_000, "3000": 2_000_000, "5000": 3_000_000}
    variable_Vehicle_Costs = {"1000": 100_000, "3000": 200_000, "5000": 300_000}

    total_Cost = 0
    for key in fixed_Fleet.keys():
        total_Cost += (
            fixed_Fleet[f"{key}"]
            * variable_Vehicle_Costs[f"{key}"]
            * n_Days
            * (1 - dsct_Factor)
            + total_Var_Fleet[f"{key}"] * variable_Vehicle_Costs[f"{key}"]
        )

    return total_Cost

def fleetCostFromSummary(summary, dsct_Factor=0.15):
    fixed_Fleet = summary["Fixed_fleet"]
    total_Vat_Fleet = summary["Total_Var_fleet"]
    N_Days = summary["Days_N"]
    return getTotalFleetCost(fixed_Fleet, total_Vat_Fleet, N_Days, dsct_Factor)

""" UNUSED """
def plotMonthUsedVehicles(year, month):
    results_Path = folder.EMPRESA_RESULTS_DIR

    x = list()
    num_Vehicles = list()
    type_Vehicles = list()

    for index, file in enumerate(os.listdir(results_Path)):

        file_Name, file_Extension = os.path.splitext(file)
        if file_Name.split("-")[1] == month:
            with open(f"{results_Path}{file}", "r") as g:
                solution = json.load(g)

            fleet_Dict = solution["Summary"]["Used_Fleet"]
            x.extend([str(file_Name) for i in range(len(fleet_Dict.keys()))])
            num_Vehicles.extend(list(fleet_Dict.values()))
            type_Vehicles.extend(list(fleet_Dict.keys()))

    data = {"x": x, "y": num_Vehicles, "cat": type_Vehicles}

    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 7))
    title = f"Uso de vehículos por tipo al mes"
    plt.title(title)
    # plt.xlabel("Día")
    plt.xticks(rotation=65, fontsize=7)
    plt.ylabel("Número de vehículos usados")
    plot = sns.barplot(x="x", y="y", hue="cat", data=data)
    plt.legend(title="Tipo")  # loc='upper right'
    plt.savefig(
        f"{DATA_PATH}Plots\\empresa\\{TODAY}\\{title}.png",
        bbox_inches="tight",
        dpi=500,
    )
    plt.show()

""" UNUSED """
def plotMonthCosts(year, month):
    results_Path = folder.EMPRESA_RESULTS_DIR

    x = list()
    Heuristic_Costs = list()

    for index, file in enumerate(os.listdir(results_Path)):

        file_Name, file_Extension = os.path.splitext(file)
        if file_Name.split("-")[1] == month:
            with open(f"{results_Path}{file}", "r") as g:
                solution = json.load(g)

            x.append(str(file_Name))
            Heuristic_Costs.append(solution["Summary"]["Total_Cost"])

    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    title = "Distancia total diaria de la planificación de rutas en un mes"
    plt.title(title)
    # plt.xlabel("Día")
    plt.xticks(rotation=65, fontsize=7)
    plt.ylabel("Distancia total")
    plt.bar(x, Heuristic_Costs)
    plt.savefig(
        f"{DATA_PATH}Plots\\empresa\\{TODAY}\\{title}.png",
        bbox_inches="tight",
        dpi=500,
    )
    plt.show()

""" UNUSED """
def plotRes(fleet=None, cost=None, sub=False):

    results = loadSolomonResults()
    results_df = pd.json_normalize(results)
    results_df.columns = results_df.columns.str.replace("Summary.", "")
    results_df.columns = results_df.columns.str.replace("Used_Fleet.", "V_")
    # results_df["Day"] = results_df["Date"].apply(lambda x: pd.Timestamp(x).day_name())
    Fleet_df = results_df.melt(
        id_vars=["File", "Total_Cost"],
        value_vars=[col for col in list(results_df.columns) if not col.find("V_")],
        var_name="Vehicle_type",
        value_name="Used",
    )
    Fleet_df["Used"] = Fleet_df["Used"].fillna(0).astype("int32")
    # Fleet_df.astype({"Used": "int32"})
    # print(max(Fleet_df["Used"]))

    if fleet:
        plt.style.use("seaborn")
        title = "Uso de vehículos por tipo al mes"
        plt.figure(figsize=(15, 6))
        plt.title(title)
        plt.xticks(rotation=65, fontsize=7)
        plt.ylabel("Número de vehículos usados")
        sns.barplot(x="File", y="Used", hue="Vehicle_type", data=Fleet_df)
        plt.show()
    if cost:
        plt.style.use("seaborn")
        title = "Distancia total diaria de la planificación de rutas en un mes"
        plt.figure(figsize=(15, 6))
        plt.title(title)
        plt.xticks(rotation=65, fontsize=7)
        plt.ylabel("Distancia total")
        sns.barplot(x="File", y="Total_Cost", data=results_df, color="g")
        plt.show()
    if sub:
        plt.style.use("seaborn")
        # sns.color_palette("tab10")
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
        for ax in axes:
            plt.setp(ax.get_xticklabels(), fontsize=7, rotation=65)
        fig.suptitle("Resultados heurística")
        # axes[0].set_title("Flota utilizada")
        plt.yticks(list(range(1, max(Fleet_df["Used"]) + 1)))
        sns.barplot(ax=axes[0], x="File", y="Used", data=Fleet_df, color="blue")
        # axes[1].set_title("Costo total")
        sns.barplot(ax=axes[1], x="File", y="Total_Cost", data=results_df, color="blue")
        plt.show()


def getVarFleetUsed(fixed_Fleet: dict, used_Fleet: dict) -> list:

    return [max(0, v - fixed_Fleet[k]) for k, v in used_Fleet.items()]


""" Grafica la flota fija y variable utilizada en un período
flota: 'Flota_distinguida' o 'Flota_ilimitada' """
def plotFixedandVarFleet(
    input_Summary_Path: Path, flota: str, save: bool = False
) -> None:

    with input_Summary_Path.open() as s:
        summary = json.load(s)

    # files_To_Read = summary["Test_Files"]
    fixed_Fleet = summary[f"{flota}"]["Fixed_fleet"]
    fixed_Fleet_List = list(fixed_Fleet.values())
    capacities = sorted(list(fixed_Fleet.keys()))

    files = []
    used_Fleets = []
    for file_Results in summary[f"{flota}"]["Results"]:
        files.append(file_Results["File"])
        used_Fleets.append(file_Results["Summary"]["Used_Fleet"])

    var_Fleets = [
        getVarFleetUsed(fixed_Fleet, used_Fleet) for used_Fleet in used_Fleets
    ]
    x = files
    y = [fixed_Fleet_List for _ in range(len(x))]
    y_F, z_F, k_F = list(zip(*y))
    y_V, z_V, k_V = list(zip(*var_Fleets))

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 5))
    X_axis = np.arange(len(x))
    width = 0.3
    plt.bar(
        X_axis - width,
        y_F,
        width=width,
        color="cornflowerblue",
        align="center",
        label=f"{capacities[0]}_F",
    )
    plt.bar(
        X_axis,
        z_F,
        width=width,
        color="lightgreen",
        align="center",
        label=f"{capacities[1]}_F",
    )
    plt.bar(
        X_axis + width,
        k_F,
        width=width,
        color="y",
        align="center",
        label=f"{capacities[2]}_F",
    )
    plt.bar(
        X_axis - width,
        y_V,
        width=width,
        color="navy",
        align="center",
        label=f"{capacities[0]}_V",
        bottom=y_F,
    )
    plt.bar(
        X_axis,
        z_V,
        width=width,
        color="darkgreen",
        align="center",
        label=f"{capacities[1]}_V",
        bottom=z_F,
    )
    plt.bar(
        X_axis + width,
        k_V,
        width=width,
        color="orange",
        align="center",
        label=f"{capacities[2]}_V",
        bottom=k_F,
    )

    title = f"Vehículos utilizados en muestra generada siguiendo el plan distinguido"
    plt.title(title)
    plt.xlabel("Instancia")
    plt.xticks(X_axis, x, rotation=65, fontsize=7)
    plt.ylabel("Vehículos utilizados")
    plt.tight_layout(pad=0.5)
    plt.legend(title="Tipo", fontsize="small")  # loc='upper right'
    if save:
        plt.savefig(
            f"{DATA_PATH}plots\\empresa\\{title}.png",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


""" Grafica los costos asociados a la flota total utilizada en cada muestra """
def plotSamplesFleetCostDiff(samples_Path: Path, save: bool = False) -> None:

    cost_Diffs = []
    dsct_Factors = [0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    for dsct_Factor in dsct_Factors:
        costos_Flota_ilimitada = []
        costos_Flota_distinguida = []
        for sample_Path in samples_Path.iterdir():
            summary_Path = sample_Path / "Test_summary.json"
            with summary_Path.open() as s:
                summary = json.load(s)
            costos_Flota_distinguida.append(
                fleetCostFromSummary(
                    summary["Flota_distinguida"], dsct_Factor=dsct_Factor
                )
            )
            costos_Flota_ilimitada.append(
                fleetCostFromSummary(
                    summary["Flota_ilimitada"], dsct_Factor=dsct_Factor
                )
            )
        mean_FD = sum(costos_Flota_distinguida) / len(costos_Flota_distinguida)
        mean_FI = sum(costos_Flota_ilimitada) / len(costos_Flota_ilimitada)
        perc_Mean_Diff = (mean_FD - mean_FI) * 100 / mean_FI
        cost_Diffs.append(perc_Mean_Diff)

    # Plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    axes.plot(cost_Diffs, marker="o")
    # title = f"Costo total de cada muestra por flota utilizada"
    # plt.title(title)
    plt.xlabel("Porcentaje de descuento")
    plt.ylabel("Diferencia costo total de la flota (%)")
    axes.set_xticks(
        ticks=[i for i in range(7)],
        labels=list(map(lambda x: str(int(x * 100)) + "%", dsct_Factors)),
    )
    plt.tight_layout(pad=0.5)
    # plt.legend(title="Flota", fontsize="small")
    if save:
        plt.savefig(
            f"{DATA_PATH}\\plots\\empresa\\Cost_Diff_VS_Dsct.png",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotFleetCostPerLongVariation(save):
    var_Folder_Path = folder.long_Periodos
    variations = ["5", "10", "15", "25", "30", "35", "40"]
    fleet_Costs = []
    for variation in variations:
        # variation_Costs = []
        # for i in range(10):
        sample_Summary_Path = (
            var_Folder_Path / f"{variation}" / f"sample_0" / "Test_summary.json"
        )
        with sample_Summary_Path.open() as s:
            summary = json.load(s)
        variation_Cost = summary["Flota_distinguida"]["Total_Fleet_Cost"]
        fleet_Costs.append(variation_Cost)

    base_Summary_Path = folder.empirica / "sample_0" / "Test_summary.json"
    with base_Summary_Path.open() as s:
        summary_Base = json.load(s)
    caso_Base = summary_Base["Flota_distinguida"]["Total_Fleet_Cost"]
    diferencia_Costos = [(i - caso_Base) * 100 / caso_Base for i in fleet_Costs]
    diferencia_Costos.insert(3, 0)
    fleet_Costs.insert(3, caso_Base)
    # Plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    axes.plot(diferencia_Costos, markersize=7, marker="o", color="blue")
    axes2 = axes.twinx()
    axes2.plot(fleet_Costs, markersize=7, marker="o", color="green")
    # title = f"Costo total de 10 muestras para 4 variaciones del número de pedidos"  #
    # plt.title(title)
    axes.set_ylabel("Diferencia del costo total [%]", color="blue")
    axes2.set_ylabel(
        "Costo total de la flota en este período en CLP [MM]", color="green"
    )
    axes.set_yticks(
        ticks=axes.get_yticks(),
        labels=list(map(lambda x: str(int(x)) + "%", axes.get_yticks())),
        # labels=list(map(lambda x: int(x // 1000000), axes.get_yticks())),
    )
    axes2.set_yticks(
        ticks=axes2.get_yticks(),
        labels=list(map(lambda x: int(x // 1000000), axes2.get_yticks())),
    )
    x_Label = "Número de semanas en un período"
    plt.xlabel(x_Label)  #
    plt.xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6, 7],
        labels=["5", "10", "15", "20", "25", "30", "35", "40"],
    )
    plt.tight_layout(pad=0.5)
    # plt.legend(title="Flota", fontsize="small")
    if save:
        plt.savefig(
            f"{DATA_PATH}\\plots\\empresa\\{x_Label}.png",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotFleetCostPerVariation(var_Folder_Path, save):
    # variations = ["Mod_-30", "Mod_-20", "Mod_-10", "Mod_10", "Mod_20", "Mod_30"]
    variations = ["5", "10", "15", "25", "30", "35", "40"]
    variations_Mean_Costs = []
    variation_Mean_Daily_Cost = []
    for variation in variations:
        variation_Costs = []
        variation_Daily_Costs = []
        for i in range(10):
            sample_Summary_Path = (
                var_Folder_Path / f"{variation}" / f"sample_{i}" / "Test_summary.json"
            )
            with sample_Summary_Path.open() as s:
                summary = json.load(s)
            # variation_Costs.append(fleetCostFromSummary(summary["Flota_distinguida"]))
            variation_Daily_Costs.append(
                fleetCostFromSummary(summary["Flota_distinguida"])
                / summary["Flota_distinguida"]["Days_N"]
            )
        # variations_Mean_Costs.append(sum(variation_Costs) / len(variation_Costs))
        variation_Mean_Daily_Cost.append(
            sum(variation_Daily_Costs) / len(variation_Daily_Costs)
        )

    costos_Caso_Base = []
    for i in range(10):
        base_Summary_Path = folder.empirica / f"sample_{i}" / "Test_summary.json"
        with base_Summary_Path.open() as s:
            summary_Base = json.load(s)
        costos_Caso_Base.append(fleetCostFromSummary(summary_Base["Flota_distinguida"]))

    costo_Medio_Caso_Base = sum(costos_Caso_Base) / len(costos_Caso_Base)
    costo_Diario_Medio_Caso_Base = costo_Medio_Caso_Base / 20
    # diferencia_Costos = [
    #     (i - costo_Medio_Caso_Base) * 100 / costo_Medio_Caso_Base
    #     for i in variations_Mean_Costs
    # ]
    # diferencia_Costos.insert(3, 0)
    diferencia_Costos = [
        (i - costo_Diario_Medio_Caso_Base) * 100 / costo_Diario_Medio_Caso_Base
        for i in variation_Mean_Daily_Cost
    ]
    diferencia_Costos.insert(3, 0)

    # Plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    axes.plot(diferencia_Costos, markersize=7, marker="o")
    # x_Label = "Factor de escala accesibilidad de pedidos"
    # x_Label = "Porcentaje pedidos de difícil acceso"
    x_Label = "Duración horizonte de tiempo [semanas]"
    plt.xlabel(x_Label)
    plt.ylabel("Diferencia del costo diario [%]")
    plt.xticks(
        ticks=[0, 1, 2, 3, 4, 5, 6, 7],
        # labels=["0.7", "0.8", "0.9", "1", "1.1", "1.2", "1.3"],
        # labels=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"],
        labels=[i for i in range(1, 9)]
        # labels=list(
        #     map(lambda x: str(int(x * 100)) + "%", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # ),
    )
    plt.yticks(
        ticks=axes.get_yticks(),
        labels=list(map(lambda x: str(int(x)) + "%", axes.get_yticks())),
        # labels=list(map(lambda x: int(x // 1000000), axes.get_yticks())),
    )
    plt.tight_layout(pad=0.5)
    # plt.legend(title="Flota", fontsize="small")
    if save:
        plt.savefig(
            f"{DATA_PATH}\\plots\\empresa\\sensibilidad_Long_Periodos.png",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def plotDataHistorica(save=False):

    folder_Path = folder.HISTORIC_DATA_DIR
    dsct_Factor = 0.15  # [0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    meses = [
        "1_enero_Febrero",
        "2_febrero_Marzo",
        "3_marzo_Abril",
        "4_abril_Mayo",
        "5_mayo_Junio",
        "6_junio_Julio",
    ]

    diferencias_Costos = []
    # for dsct_Factor in dsct_Factors:
    diferencia_Costos = []
    for mes in meses:
        sample_Summary_Path = folder_Path / f"{mes}" / "Test_summary.json"
        with sample_Summary_Path.open() as s:
            summary = json.load(s)
        costo_Flota_Ilimitada = fleetCostFromSummary(
            summary["Flota_ilimitada"], dsct_Factor=dsct_Factor
        )
        costo_Flota_Distinguida = fleetCostFromSummary(
            summary["Flota_distinguida"], dsct_Factor=dsct_Factor
        )
        diferencia_Costos.append(
            (costo_Flota_Distinguida - costo_Flota_Ilimitada)
            * 100
            / costo_Flota_Ilimitada
        )
    diferencias_Costos.append(diferencia_Costos)

    # Plot
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    axes.plot(
        diferencias_Costos[0],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factor*100)}%",
    )
    """
    axes.plot(
        diferencias_Costos[1],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[1]*100)}%",
    )
    axes.plot(
        diferencias_Costos[2],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[2]*100)}%",
    )
    axes.plot(
        diferencias_Costos[3],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[3]*100)}%",
    )
    axes.plot(
        diferencias_Costos[4],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[4]*100)}%",
    )
    axes.plot(
        diferencias_Costos[5],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[5]*100)}%",
    )
    axes.plot(
        diferencias_Costos[6],
        markersize=7,
        marker="o",
        label=f"{int(dsct_Factors[6]*100)}%",
    )
    """
    x_Label = "Mes de prueba"
    plt.xlabel(x_Label)
    plt.ylabel("Diferencia de costos en el período [%]")
    plt.xticks(
        ticks=[0, 1, 2, 3, 4, 5],
        labels=[mes.split("_")[1] for mes in meses]
        # labels=list(
        #     map(lambda x: str(int(x * 100)) + "%", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # ),
    )
    plt.yticks(
        ticks=axes.get_yticks(),
        labels=list(map(lambda x: str(int(x)) + "%", axes.get_yticks())),
    )
    # plt.tight_layout(pad=0.5)
    axes.legend(
        title="Descuento",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=4,
        fancybox=True,
    )
    # plt.legend(
    #     title="Descuento", fontsize="small", loc="upper right"
    # )  # loc='upper right'
    if save:
        plt.savefig(
            f"{DATA_PATH}\\plots\\empresa\\resultados_Data_Historica.png",
            bbox_inches="tight",
            dpi=500,
        )
    plt.show()


def exPlots():

    # Plots data historica junio-Julio
    # plotFixedandVarFleet(sample_Summary_Path, flota, save=True)
    plotDataHistorica(save=False)

    # Plots meustras generadas
    # Empirica
    # sample_Summary_Path = folder.empirica / "sample_0" / "Test_summary.json"
    # flota = "Flota_distinguida"  # Flota_distinguida
    # plotFixedandVarFleet(sample_Summary_Path, flota, save=True)

    # samples_Path = folder.empirica
    # plotSamplesFleetCostDiff(samples_Path, save=True)


def main():

    exPlots()
    # var_Folder_Path = folder.long_Periodos
    # plotFleetCostPerVariation(var_Folder_Path, save=True)
    # plotFleetCostPerLongVariation(save=False)
    # flota = "Flota_distinguida"
    # sample_Summary_Path = folder.empirica / "sample_4" / "Test_summary.json"
    # samples_Summary_Path = folder.empirica / "Samples_summary.json"
    # plotJustVarFleet(sample_Summary_Path, save=False)
    # plotFixedandVarFleet(sample_Summary_Path, flota, save=False)
    # plotVariationsFleetCost(input_Folder_Path, "distinguished_Fleet")



if __name__ == "__main__":
    main()
