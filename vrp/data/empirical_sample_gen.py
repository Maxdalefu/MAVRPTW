from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from random import choice, seed
from typing import Callable
from pprint import pprint
from pathlib import Path
import pandas as pd
import numpy as np
import os, json
from osrm import table

from utils.folder import Folder
from other.simpli.simpli_to_json import getNodesSantiago


folder = Folder()


""" THIS FILE DEFINES THE SAMPLE GENERATOR WHICH GENERATES FICTIONAL SAMPLES BASED ON EMPIRICAL DATA """


def loadDailySummary() -> list:
    summary = []
    for file in os.listdir(folder.ORIGINAL_DATA_DIR):
        df = pd.read_csv(folder.ORIGINAL_DATA_DIR + str(file), sep="\t")
        n_Nodes = len(df)
        total_Load = int(sum(df["load"]))
        load_0 = len(df[df["load"] == 0])
        nans = df["load"].isna().sum()
        summary.append([file, n_Nodes, total_Load, load_0, nans])
    return summary


def is_outlier(points, thresh=3.5) -> bool:

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# def f(row):
#     a,b,c,d=row
#     return [[a,b],[c,d]]
# df["result"] = df["window_start", "window_end", "window_start_2", "window_end_2"].apply(f,axis=1,raw=True)


class sampleGen:
    def __init__(self) -> None:

        self.data = pd.read_csv(folder.EMPRESA_DATA_DIR / "data.csv", parse_dates=["fecha"])
        self.vol_Mean = self.data["load"].mean()

        # Nodes info: lat_lon, time windows, duration
        self.nodos_DF = self.getFilteredNodes()
        self.nodos_Stgo_List = self.nodos_DF.to_dict("records")
        self.nodos_Dificil_Acceso = self.nodos_DF[self.nodos_DF["skills"] == 1].to_dict(
            "records"
        )
        self.nodos_Facil_Acceso = self.nodos_DF[self.nodos_DF["skills"] == 0].to_dict(
            "records"
        )

        # Empirica inversa para el número de pedidos al dia
        self.ei_Num_Pedidos = self.getInvNumPedidos()

        # Empirica inversa para el volmen de demanda por pedido
        self.ei_Vol_Pedido = self.getInvVolPedido()

    """ Extrae la informacion de cada nodo, coordenadas, ventanas de tiempo y duracion """

    def getFilteredNodes(self):
        nodes_Stgo, polygon_Stgo = getNodesSantiago()
        df2 = pd.DataFrame(
            nodes_Stgo.drop(
                [
                    "latitude",
                    "longitude",
                    "geometry",
                ],
                axis=1,
            )
        )
        return df2
        # else:
        #     return df2.to_dict("records")

    """ Define la funcion inversa aproximada de la ECDF del numero de pedidos por dia """

    def getInvNumPedidos(self) -> Callable:
        n_Pedidos_Por_Dia = self.data.groupby("fecha")["load"].count()
        filtered_n = n_Pedidos_Por_Dia[~is_outlier(np.array(n_Pedidos_Por_Dia))]
        ecdf_n = ECDF(x=filtered_n)
        e_x_n = sorted(set(filtered_n))
        e_y_n = ecdf_n(e_x_n)
        return interp1d(e_y_n, e_x_n, kind="previous")

    """ Define la funcion inversa aproximada de la ECDF del volumen de los pedidos """

    def getInvVolPedido(self) -> Callable:
        volumen_Por_Pedido = self.data[self.data["load"] > 0]["load"]
        vol_ped_final = volumen_Por_Pedido[~is_outlier(np.array(volumen_Por_Pedido))]
        ecdf_t = ECDF(x=vol_ped_final)
        e_x_t = sorted(set(vol_ped_final))
        e_y_t = ecdf_t(e_x_t)
        return interp1d(e_y_t, e_x_t, kind="previous")

    """ Genera una muestra de numero de pedidos para el numero de dias dado """

    def sampleNumPedidos(self, dias: int = 1) -> np.array:
        uni_Sample = np.random.uniform(min(self.ei_Num_Pedidos.x), 1, dias)
        return self.ei_Num_Pedidos(uni_Sample).astype(int)

    """ Genera una muestra de volumenes para el numero de pedidos de un dia dado """

    def sampleVolPedidos(self, pedidos: int) -> np.array:
        uni_sample = np.random.uniform(min(self.ei_Vol_Pedido.x), 1, pedidos)
        return self.ei_Vol_Pedido(uni_sample)

    """ Genera una muestra de los volumenes de los pedidos condicionada por una media deseada """

    def sampleVolPedidosCondicional(self, pedidos: int, des_Mean: float) -> list:
        uni_sample = np.random.uniform(min(self.ei_Vol_Pedido.x), 1, 1)
        vol_Sample = self.ei_Vol_Pedido(uni_sample)[0]
        vol_Pedidos = [vol_Sample]
        while len(vol_Pedidos) < pedidos:
            flag = False
            uni_sample = np.random.uniform(min(self.ei_Vol_Pedido.x), 1, 1)
            vol_Sample = self.ei_Vol_Pedido(uni_sample)[0]
            if np.mean(vol_Pedidos) < des_Mean:
                flag = True
            if flag:
                if vol_Sample < des_Mean:
                    # print(f"Pedido {vol_Sample} rechazado")
                    continue
            vol_Pedidos.append(vol_Sample)
            # print(f"Pedido {vol_Sample} aceptado, media = {np.mean(vol_Pedidos)}")
        # print("Muestra generada:", vol_Pedidos)
        # print("Media de la muestra:", np.mean(vol_Pedidos))
        return vol_Pedidos

    """ Genera una muestra de de la distribución empírica de los datos
    Parametros:
    dias(int): tamaño de la muestra a generar
    sid(int): semilla para la selección "aleatoria"
    path(Path): ubicación donde se desea escribir la muestra
    num_Modifier: if the number of requests wants to be modified
    vol_Modifier: if the requests size wants to be modified
    acc_Modifier: if the requests distribution wants to be modified """

    def genSample(
        self,
        dias: int,
        sid: int,
        path: Path,
        num_Modifier: float = 0,
        vol_Modifier: float = 0,
        acc_Modifier: float = 0,
    ) -> None:
        seed(sid)
        depot = {
            "id": 0,
            "location": [-70.6629, -33.5044],
            "load": 0,
            "time_windows": [[0, 36000]],
            "skills": 0,
            "service": 0,
        }
        # Sampleamos el numero de pedidos por dia, devuelve una lista de enteros de tamaño dias
        num_Ped = self.sampleNumPedidos(dias)
        # Modificar la muestra con num_Modifier
        num_Pedidos = list(map(lambda x: int(x + x * num_Modifier), num_Ped))
        for dia in range(dias):

            jobs = []
            file_Name = f"dia_{dia}"
            if vol_Modifier:
                media_Deseada = self.vol_Mean * (1 + vol_Modifier)
                vol_Pedidos = self.sampleVolPedidosCondicional(
                    num_Pedidos[dia], media_Deseada
                )
            else:
                vol_Pedidos = self.sampleVolPedidos(num_Pedidos[dia])

            for pedido in range(num_Pedidos[dia]):

                # Si hay modificacion de accesibilidad, tomar nodos de dificil acceso primero
                if acc_Modifier:
                    if len(jobs) < (acc_Modifier + 0.3) * num_Pedidos[dia]:
                        job = choice(self.nodos_Dificil_Acceso)
                    else:
                        job = choice(self.nodos_Facil_Acceso)
                else:
                    job = choice(self.nodos_Stgo_List)
                job.update({"id": pedido + 1, "load": vol_Pedidos[pedido]})
                jobs.append(job)

            jobs.insert(0, depot)
            coords = list(pd.DataFrame(jobs)["location"])
            # return jobs
            matrix = table(coords)

            json_Data = {
                "file": file_Name,
                "jobs": jobs,
                "matrix": matrix,
            }
            # pprint(json_Data)
            if not path:
                pprint(json_Data)
            else:
                print(f"Writing {file_Name}")
                with open(f"{str(path)}\\{file_Name}.json", "w") as out:
                    out.write(json.dumps(json_Data, indent=4))


""" Análisis de sensibilidad 
    - Numero de pedidos diarios
    - Volumen medio de pedido
    - Cantidad de clientes con restriccion de acceso """


def genSamples(samples_N: int = 1):

    gener = sampleGen()
    sub_Groups = ["Train", "Test"]
    modds = ["-30", "-20", "-10", "10", "20", "30"]
    scales = [-0.3, -0.1, 0.1, 0.3]

    # for mod, var in zip(modds, scales):
    #     for i in range(samples_N):
    #         for group in sub_Groups:

    # Empirica normal
    for i in range(10, samples_N):
        for group in sub_Groups:
            path = folder.empirica / f"sample_{i}" / f"{group}"
            path.mkdir(parents=True, exist_ok=True)
            gener.genSample(dias=20, sid=i, path=path)

    # Num pedidos
    # path1 = folder.num_Pedidos / f"Mod_{mod}" / f"sample_{i}" / f"{group}"
    # path1.mkdir(parents=True, exist_ok=True)
    # gener.genSample(dias=20, sid=i, path=path1, num_Modifier=var)

    # Vol pedidos
    # path2 = folder.tam_Pedidos / f"Mod_{mod}" / f"sample_{i}" / f"{group}"
    # path2.mkdir(parents=True, exist_ok=True)
    # gener.genSample(dias=20, sid=i + 10, path=path2, vol_Modifier=var)

    # Acc pedidos
    # path3 = folder.acc_Pedidos / f"Mod_{mod}" / f"sample_{i}" / f"{group}"
    # path3.mkdir(parents=True, exist_ok=True)
    # gener.genSample(dias=20, sid=i + 20, path=path3, acc_Modifier=var)

    # Tamaño periodos
    # tamaños = [30, 35, 40]  # de 1 a 7 semanas 5, 10, 15, 25,
    # args = []
    # for tam in tamaños:
    #     for i in range(samples_N):
    #         for group in sub_Groups:
    #             # Tam periodos
    #             path4 = folder.long_Periodos / f"{tam}" / f"sample_{i}" / f"{group}"
    #             path4.mkdir(parents=True, exist_ok=True)
    #             gener.genSample(dias=tam, sid=i + 30, path=path4)
    # args.append((tam, i + 30, path4))
    # print(args)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(sampleGen.genSample, args)


def main():

    # gen = sampleGen()
    # nodes = gen.getFilteredNodes(True)
    # gen.genSample(dias=1)
    genSamples(samples_N=100)

    # nodes, polygon = getNodesSantiago()
    # print(gen.getNodesInfo())


if __name__ == "__main__":
    main()
