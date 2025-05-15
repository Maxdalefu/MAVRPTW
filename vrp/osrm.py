import pandas
import requests
from pathlib import Path

DEFAULT_IP = "127.0.0.1"  # "0.0.0.0"
DEFAULT_PORT = "5000"
FOLDER_PATH = str(Path(__file__).resolve().parent.parent)
DATA_PATH = str(Path(__file__).resolve().parent.parent.parent) + "\\data\\"


def format_Table_Request(locs):
    req = "http://" + DEFAULT_IP + ":" + DEFAULT_PORT + "/table/v1/car/"
    for loc in locs:
        req += str(loc[0]) + "," + str(loc[1]) + ";"

    return req[:-1]


def table(locs):
    req = format_Table_Request(locs)
    try:
        return requests.get(req).json()["durations"]
    except:
        print(requests.get(req).json())


def main():
    df = pandas.read_csv(DATA_PATH + "2020-10-02.csv", sep="\t")
    # coords = lon, lat
    # Cambiar estas coordenadas por las reales del depot
    coords = [(-70.6629, -33.5044)]
    coords.extend(list(zip(df["longitude"], df["latitude"])))
    matrix = table(coords)


if __name__ == "__main__":
    main()
