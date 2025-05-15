import json, os, concurrent.futures
from random import seed, shuffle
from pathlib import Path
from pprint import pprint

from tests.vrp_tester import solveVRP, multiProcessData
from src.utils.folder import Folder

folder = Folder()


"""
La idea de este script es tomar un conjunto de instancias (varios días)
resolverlo con flota ilimitada, simulando la operación normal con flota completamente variable
en base a estos resultados, escoger una flota y testear su desempeño en otro período similar.

Comparación de metodologías:
1.- Comparar la función de consenso con una flota completamente variable, económicamente y con análisis estadístico

Análisis de sensibilidad:
Ver como afectan distintas variaciones en las muestras generadas sobre nuestra metodología, 3 variaciones:
1.- Número de pedidos por día
2.- Volúmen de pedio
3.- Accesibilidad geográfica; mayor/menor proporción de clientes de difícil acceso
*4.- Tamaño del período
"""


def loadResultsSample(
    size: int = 100, input_Folder_Path: str = str(folder.empresa) + "Original"
):

    all_Results = []
    for file in os.listdir(str(input_Folder_Path) + "Results\\"):
        # print(file)
        file_Name, file_Extension = os.path.splitext(file)
        if file_Extension == ".json":
            with open(f"{input_Folder_Path}Results\\{file}", "r") as g:
                solution = json.load(g)
            try:
                all_Results.append(
                    {
                        k: v
                        for k, v in solution.items()
                        if (k == "File") or (k == "Summary")
                    }
                )

            except:
                pass
    shuffle(all_Results)

    train_Files = [result["File"] for result in all_Results[:75]]
    train_Summaries = [result["Summary"] for result in all_Results[:75]]
    test_Files = [result["File"] for result in all_Results[75:size]]

    return train_Files, train_Summaries, test_Files


""" Prueba la meotodología tomando instancias de data historica y separandola en conjunto train test de manera aleatoria """


def testVariationSample(sample: int = 0, variation: str = "Original"):

    print(f"Variation = {variation}, sample = {sample}")
    sample_Size = 100
    seed(sample)
    input_Folder_Path = f"{folder.empresa}{variation}\\"
    output_Folder_Path = f"{input_Folder_Path}sample{sample}\\"

    train_Files, train_Summaries, test_Files = loadResultsSample(
        sample_Size, input_Folder_Path
    )

    print("Training files:", len(train_Files))
    print("Testing files:", len(test_Files))

    distinguishedPlan = getDistinguishedFleet(train_Summaries)
    # distinguishedPlan = distinguishedPlans
    print("Distinguished plan:", distinguishedPlan)
    try:
        os.mkdir(output_Folder_Path)
        print(f"Directory sample{sample} created")
    except:
        pass

    with open(f"{output_Folder_Path}sample_Summary.json", "w") as out:
        json_Data = {
            "Sample": sample,
            "Train_Files": train_Files,
            "Train_Summaries": train_Summaries,
            "Test_Files": test_Files,
            "Sample_Size": sample_Size,
            "Distinguished_Plan": distinguishedPlan,
        }
        out.write(json.dumps(json_Data, indent=4))

    args = []
    for file in test_Files:
        args.append((input_Folder_Path, output_Folder_Path, file, distinguishedPlan))
    # file = "2020-10-06"
    # solution = solveVRP(
    #     args=(input_Folder_Path, output_Folder_Path, "empresa", file, distinguishedPlan)
    # )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(solveVRP, args)


def genVariationSummary(input_Folder_Path):

    variations = ["Loadx0.5", "Loadx0.8", "Original", "Loadx1.4"]  # , "Loadx1.8"

    for variation in variations:

        variation_Path = f"{input_Folder_Path}{variation}\\"
        sample_Summaries = []

        for i in range(10):

            sample_Path = variation_Path + f"sample{i}\\"
            with open(sample_Path + "sample_Summary.json", "r") as s:
                sample_Summary = json.load(s)

            distinguished_Var_Fleet, d_num_Files = getSampleVarFleet(
                sample_Path, "distinguished_Fleet"
            )
            reduced_Var_Fleet, r_num_Files = getSampleVarFleet(
                sample_Path, "smallest_Fleet"
            )
            test_Files = sample_Summary["Test_Files"]
            just_Var_Fleets = []
            for file in test_Files:
                with open(f"{variation_Path}\\Results\\{file}.json", "r") as r:
                    just_Var_Fleets.append(
                        list(json.load(r)["Summary"]["Used_Fleet"].values())
                    )
            x, y, z = list(zip(*just_Var_Fleets))
            var_Fleet_Without_Fixed_Fleet = [sum(x), sum(y), sum(z)]

            sample_Dict = {
                "Sample": "sample" + str(i),
                "Sample_Size": sample_Summary["Sample_Size"],
                "Distinguished_Fleet": sample_Summary["Distinguished_Plan"],
                "D_Successful_Results": d_num_Files,
                "Var_Fleet_From_Distinguished_Fleet": distinguished_Var_Fleet,
                "R_Successful_Results": r_num_Files,
                "Var_Fleet_From_Reduced_Fleet": reduced_Var_Fleet,
                "Var_Fleet_Without_Fixed_Fleet": var_Fleet_Without_Fixed_Fleet,
            }
            sample_Summaries.append(sample_Dict)
        # print(samples_Summaries)

        with open(f"{variation_Path}variation_Summary.json", "w") as out:
            variation_Summary = {
                "variation": variation,
                "sample_Summaries": sample_Summaries,
            }
            out.write(json.dumps(variation_Summary, indent=4))


def testSampleFleet(variation_Path: str, sample: str) -> None:

    sample_Folder_Path = variation_Path + f"sample{sample}\\"
    output_Folder_Path = sample_Folder_Path + "smallest_Fleet\\"
    try:
        os.mkdir(output_Folder_Path)
    except:
        pass

    with open(f"{sample_Folder_Path}sample_Summary.json", "r") as d:
        sample_Summary = json.load(d)

    files = sample_Summary["Test_Files"]
    distinguishedPlan = sample_Summary["Distinguished_Plan"]

    capacities = list(distinguishedPlan.keys())
    smallest_Capacity = min(capacities)
    smallest_Fleet = dict(
        [
            (k, 0)
            if k != smallest_Capacity
            else (k, sum(list(distinguishedPlan.values())))
            for k in capacities
        ]
    )

    args = []
    for file in files:
        args.append(
            (variation_Path, output_Folder_Path, "empresa", file, smallest_Fleet)
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(solveVRP, args)


def getSmallestFleet(results: list):
    daily_Fleets = [
        list(result["Summary"]["Used_Fleet"].values()) for result in results
    ]
    smallest_Num = sum(list(daily_Fleets[0]))
    for daily_Fleet in daily_Fleets:
        this_Fleet_Num = sum(list(daily_Fleet))
        if this_Fleet_Num < smallest_Num:
            smallest_Num = this_Fleet_Num
            smallest_Fleet = daily_Fleet
    # print("Smallest cant:", smallest_Num)
    # print("Smallest fleet:", smallest_Fleet)
    return dict(zip(list(results[0]["Summary"]["Used_Fleet"].keys()), smallest_Fleet))


def moveFiles(input_Folder_Path, output_Folder_Path):

    try:
        os.mkdir(output_Folder_Path)
    except:
        pass

    with open(input_Folder_Path + "sample_Summary.json", "r") as s:
        summary = json.load(s)

    files = summary["Test_Files"]
    for file in files:
        try:
            os.rename(
                f"{input_Folder_Path}{file}.json", f"{output_Folder_Path}{file}.json"
            )
        except:
            pass


""" Genera el resumen de todas las muestras; flota fija , flota variable total utilizada y costo total 
ELIMINAR, NO AGREGA NADA """


# def genSamplesSummary(samples_Folder_Path: Path) -> None:

#     samples_Summaries = {}
#     flotas = ["Flota_distinguida", "Flota_ilimitada"]
#     output_Path = samples_Folder_Path / "Samples_summary.json"

#     for sample_Folder_Path in samples_Folder_Path.iterdir():
#         if sample_Folder_Path.is_dir():
#             sample_Summary = {}
#             sample = sample_Folder_Path.stem
#             test_Summary_Path = sample_Folder_Path / "Test_summary.json"

#             with test_Summary_Path.open() as g:
#                 summary = json.load(g)
#             for flota in flotas:

#                 fixed_Fleet = summary[f"{flota}"]["Fixed_fleet"]
#                 total_Var_fleet = summary[f"{flota}"]["Total_Var_fleet"]
#                 total_Cost = getTotalFleetCost(fixed_Fleet, total_Var_fleet)

#                 sample_Summary[f"{flota}"] = {
#                     "Fixed_fleet": fixed_Fleet,
#                     "Total_Var_fleet": total_Var_fleet,
#                     "Total_cost": total_Cost,
#                 }
#             samples_Summaries[f"{sample}"] = sample_Summary

#     output_Path.write_text(json.dumps(samples_Summaries, indent=4))


""" Calcula la flota distinguida de una lista de resultados """


def getDistinguishedFleet(results: list) -> dict:

    daily_Fleets = [
        list(result["Summary"]["Used_Fleet"].values()) for result in results
    ]
    plan_Differences = []

    for plan1 in daily_Fleets:
        difference = 0

        for plan2 in daily_Fleets:
            difference += sum([abs(plan1[i] - plan2[i]) for i in range(len(plan1))])

        plan_Differences.append(difference)

    least_Difference = min(plan_Differences)
    best_Plans = set(
        [
            tuple(daily_Fleets[i])
            for i, value in enumerate(plan_Differences)
            if value == least_Difference
        ]
    )

    return dict(
        zip(list(results[0]["Summary"]["Used_Fleet"].keys()), list(best_Plans)[0])
    )


""" Extrae la flota variable total utilizada a lo largo de un período """


def getSampleVarFleet(used_Fleets: list, fixed_Fleet: dict) -> dict:

    total_Var_Fleet = dict([(key, 0) for key in fixed_Fleet.keys()])

    for used_Fleet in used_Fleets:
        for k, v in used_Fleet.items():
            if v > fixed_Fleet[f"{k}"]:
                total_Var_Fleet[f"{k}"] += v - fixed_Fleet[f"{k}"]
    return total_Var_Fleet


""" Genera el resumen de entrenamiento de una muestra
    Interesa guardar las flotas utilizadas y la flota distinguida
    Falta especificar lo que interesa de Summary """


def genTrainingSummary(sample_Folder_Path: Path) -> None:

    train_Results_Folder_Path = sample_Folder_Path / "Train" / "Flota_ilimitada"
    output_Path = sample_Folder_Path / "Training_summary.json"
    all_Results = []
    for file_Path in train_Results_Folder_Path.iterdir():
        if file_Path.is_file():
            with file_Path.open() as g:
                solution = json.load(g)
            all_Results.append(
                {k: v for k, v in solution.items() if (k == "File") or (k == "Summary")}
            )
    distinguished_Fleet = getDistinguishedFleet(all_Results)
    print("Distinguished fleet:", distinguished_Fleet)
    summary = {
        "Training_results": all_Results,
        "Distinguished_fleet": distinguished_Fleet,
    }
    output_Path.write_text(json.dumps(summary, indent=4))


""" Genera el resumen de entrenamiento de todas las muestras"""


def genAllSamplesTrainSummary(samples_Folder_Path: Path) -> None:
    for sample_Folder_Path in samples_Folder_Path.iterdir():
        if sample_Folder_Path.is_dir():
            genTrainingSummary(sample_Folder_Path)


""" Genera el resumen de Test para una muestra, incluyendo los resultados para cada flota testeada
    Acá interesa gardar las flotas utilizadas, tanto fija como variabel y el costo total asociado a esta """


def genTestSummary(sample_Folder_Path: Path) -> None:

    test_Folder_Path = sample_Folder_Path / "Test"
    output_Path = sample_Folder_Path / "Test_summary.json"
    test_Summary = {}
    # Vemos las distintas flotas testeadas
    for results_Folder_Path in test_Folder_Path.iterdir():
        if results_Folder_Path.is_dir():
            fleet_Summary = {}
            flota = results_Folder_Path.stem

            # Guardamos los resultados de todas las instancias de la muestra
            all_Results = []
            used_Fleets = []
            for file_Path in results_Folder_Path.iterdir():
                if file_Path.is_file():
                    with file_Path.open() as g:
                        solution = json.load(g)
                    all_Results.append(
                        {
                            k: v
                            for k, v in solution.items()
                            if (k == "File") or (k == "Summary")
                        }
                    )
                    used_Fleets.append(solution["Summary"]["Used_Fleet"])
            fleet_Summary["Results"] = all_Results

            # Guardamos la flota fija utilizada
            if flota == "Flota_ilimitada":
                fixed_Fleet = {"1000": 0, "3000": 0, "5000": 0}
            else:
                train_Summary_Path = sample_Folder_Path / "Training_summary.json"
                with train_Summary_Path.open() as g:
                    train_Summary = json.load(g)
                fixed_Fleet = train_Summary[f"Distinguished_fleet"]
            fleet_Summary["Fixed_fleet"] = fixed_Fleet

            # Guardamos la flota variable total utilizada
            total_Var_Fleet = getSampleVarFleet(used_Fleets, fixed_Fleet)
            fleet_Summary["Total_Var_fleet"] = total_Var_Fleet

            # Guardamos el total de días del período
            fleet_Summary["Days_N"] = len(all_Results)

            # Y por último el costo total asociado a la flota del período
            # total_Cost = getTotalFleetCost(fixed_Fleet, total_Var_Fleet)
            # fleet_Summary["Total_Fleet_Cost"] = total_Cost

            # Guardamos el summary de cada flota
            test_Summary[f"{flota}"] = fleet_Summary

    output_Path.write_text(json.dumps(test_Summary, indent=4))


""" Genera el resumen de test de todas las muestras """


def genAllSamplesTestSummary(samples_Folder_Path: Path) -> None:
    for sample_Folder_Path in samples_Folder_Path.iterdir():
        if sample_Folder_Path.is_dir():
            genTestSummary(sample_Folder_Path)


""" Resuelve el conjunto de entrenamiento de una muestra con flota ilimitada """


def trainSample(sample_Folder_Path: Path) -> None:

    # for sample_Folder_Path in samples_Folder_Path.iterdir():
    #     if sample_Folder_Path.is_dir():
    #         for conjunto in conjuntos:
    input_Folder_Path = sample_Folder_Path / "Train"
    output_Folder_Path = input_Folder_Path / "Flota_ilimitada"
    fleet = {"1000": 50, "3000": 50, "5000": 50}
    multiProcessData(
        input_Folder_Path,
        output_Folder_Path,
        fleet,
        speed_Factor=0.28,
        max_Jobs_Per_Route=33,
    )


""" Resuelve el conjunto Test de todas las muestras con la flota distinguida obtenida """


def testSample(sample_Folder_Path: Path, flota_Distinguida: bool = True) -> None:

    input_Folder_Path = sample_Folder_Path / "Test"
    if flota_Distinguida:
        summary_Path = sample_Folder_Path / "Training_summary.json"
        with summary_Path.open() as g:
            summary = json.load(g)
        fleet = summary["Distinguished_fleet"]
        output_Folder_Path = input_Folder_Path / "Flota_distinguida"
    else:
        fleet = {"1000": 50, "3000": 50, "5000": 50}
        output_Folder_Path = input_Folder_Path / "Flota_ilimitada"

    multiProcessData(
        input_Folder_Path,
        output_Folder_Path,
        fleet,
        speed_Factor=0.28,
        max_Jobs_Per_Route=33,
    )


"""
Experimentos:
- Utilizando data histórica o muestras generadas
- Entrenar con el conjunto Trainy generar su resumen
- Testear la flota distinguida con el conjunto Test y generar resumen

Data histórica:
- Resolver Junio-Julio

Muestras generadas:
Validación de metodollogía

Experimento N°1:
- Resolver el conjunto de entrenamiento y prueba (por separada) con flota ilimitada 

Análisis de sensibilidad

Experimento N°2 (n° pedidos):
- Generar muestras con mayor número de pedidos por día 10-30%

Experimento N°3 (vol pedidos):
- Generar muestras con mayor volumen medio 10-30%

Experimento N°4 (accesibilidad pedidos):
- Generar muestras con mayor número de pedidos con restricción de acceso 30% -> 40-50%

Experimento N°5 (tamaño período):
- Generar muestras con mayor/menor número de días que tiene un período
"""


def solveSample(sample_Folder_Path: Path, test_Flota_Ilimitada: bool = False):
    # Entrenamiento
    print(f"\nTraining sample in folder: {sample_Folder_Path}\n")
    trainSample(sample_Folder_Path)
    # Generacion del resumen de entrenamiento
    genTrainingSummary(sample_Folder_Path)
    # Testeo
    print(f"\nTesting distinguished fleet in folder: {sample_Folder_Path}\n")
    testSample(sample_Folder_Path)
    # Si se quiere comparar con la flota ilimitada, tambien lo testeamos
    if test_Flota_Ilimitada:
        print(f"\nTesting unlimited fleet in folder: {sample_Folder_Path}\n")
        testSample(sample_Folder_Path, flota_Distinguida=False)
    # Y finalmente se genera el resumen de Test
    genTestSummary(sample_Folder_Path)


def solveExperiments(exp_N):

    # VALIDACION

    # Resolver data historica junio-julio, mes_mes
    if exp_N == 0:
        sample_Foder_Path = folder.empresa / "enero_Febrero"
        solveSample(sample_Foder_Path, test_Flota_Ilimitada=True)

    # Resolver muestra de data generada de la distribucion empirica
    if exp_N == 1:
        for i in range(10):
            print("Solving experiment sample:", i)
            sample_Foder_Path1 = folder.empirica / f"sample_{i}"
            solveSample(sample_Foder_Path1, test_Flota_Ilimitada=True)

    # ANALISIS DE SENSIBILIDAD

    # Numero de pedidos
    if exp_N == 2:
        modds = ["-30", "-10", "10", "30"]
        for mod in modds:
            for i in range(10):

                print(f"\n\nSOLVING SAMPLE {i} FROM MOD {mod}\n\n")

                # Numero de pedidos
                sample_Foder_Path2 = folder.num_Pedidos / f"Mod_{mod}" / f"sample_{i}"
                solveSample(sample_Foder_Path2)

                # Tamaño de pedidos
                sample_Foder_Path3 = folder.tam_Pedidos / f"Mod_{mod}" / f"sample_{i}"
                solveSample(sample_Foder_Path3)

                # Accesibilidad pedidos
                sample_Foder_Path4 = folder.acc_Pedidos / f"Mod_{mod}" / f"sample_{i}"
                solveSample(sample_Foder_Path4)

    # Duracion del periodo
    if exp_N == 3:
        longitudes = ["5", "10", "15", "25", "30", "35", "40"]
        for longitud in longitudes:
            for i in range(10):

                print(f"\n\nSOLVING SAMPLE {i} FROM MOD {longitud}\n\n")
                sample_Foder_Path5 = (
                    folder.long_Periodos / f"{longitud}" / f"sample_{i}"
                )
                solveSample(sample_Foder_Path5)


def main():

    # samples_Folder_Path = folder.long_Periodos
    # for var_Path in samples_Folder_Path.iterdir():
    #     genAllSamplesTestSummary(var_Path)
    solveExperiments(exp_N=0)


if __name__ == "__main__":
    main()
