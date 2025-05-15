from pathlib import Path


class Folder:
    def __init__(self) -> None:

        self.BASE_DIR = Path(__file__).resolve().parents[2]

        """Main folder"""
        self.DATA_DIR = self.BASE_DIR / "data"
        self.SRC_DIR = self.BASE_DIR / "src"
        self.TESTS = self.BASE_DIR / "tests"

        """data"""

        self.EMPRESA_DATA_DIR = self.DATA_DIR / "empresa"
        """data/empresa"""
        self.ORIGINAL_DATA_DIR = self.EMPRESA_DATA_DIR / "Original"
        self.HISTORIC_DATA_DIR = self.EMPRESA_DATA_DIR / "Data_Historica"
        self.GENERATED_DATA_DIR = self.EMPRESA_DATA_DIR / "Muestras_Generadas"

        self.INSTANCES_DIR = self.DATA_DIR / "instances"
        """data/instances"""
        self.SOLOMON_INSTANCES_DIR = self.INSTANCES_DIR / "solomon_instances"
        self.BKS = self.INSTANCES_DIR / "BKS.json"

        self.PLOTS_DIR = self.DATA_DIR / "plots"
        """data/plots"""
        self.PLOTS_EMPRESA_DIR = self.PLOTS_DIR / "empresa"
        self.PLOTS_SOLOMON_DIR = self.PLOTS_DIR / "solomon"

        self.RESULTS_DIR = self.DATA_DIR / "results"
        """data/results"""
        self.EMPRESA_RESULTS_DIR = self.RESULTS_DIR / "empresa_results"
        self.SOLOMON_RESULTS_DIR = self.RESULTS_DIR / "solomon_results"





def main():

    folder = Folder()
    # Checkeamos si todas lasc arpetas existen
    for k, v in vars(folder).items():
        if not v.exists():
            print(f"{v} not exists")


if __name__ == "__main__":
    main()
    # print(Path(__file__).resolve().parents[2])
