import json, timeit, cProfile, pstats
from pprint import pprint

from utils.folder import Folder


folder = Folder()


def _copyList(l: list) -> list:
    ret = l.copy()
    for idx, item in enumerate(ret):
        ret[idx] = item
    return ret


"""CLASSES"""


""" data: {"matrix": [[]], "fleet_comp": {"capacity": number}, 
"jobs": [{"skills": [], "load": int, "time_windows": list, "service": int}, ...]} """
class Instance:
    def __init__(
        self,
        data: dict, # move data preparation to tester
        file: str = None, # move to tester
        fleet: dict = None,
        speed_Factor: float = 1,
        max_Jobs_Per_Route: float = float("inf"),
    ) -> None:

        # Parameters
        self.file = file # Instance should not have file
        self.speed_Factor = speed_Factor
        self.max_Jobs_Per_Route = max_Jobs_Per_Route

        # This statement should not exist
        if data:
            self.data = data
        else:
            raise "No data given"

        # Informacion general de la instancia
        # Move to data preparation
        self.matrix = self.data["matrix"]
        self.scaled_Matrix = self.getScaledMatrix()
        if fleet:
            self.fleet_Comp = fleet
        else:
            self.fleet_Comp = self.data["fleet_Comp"]
        for key, value in self.fleet_Comp.items():
            self.fleet_Comp[key] = int(value)
        self.jobs = self.data["jobs"]
        self.n_Jobs = len(self.jobs) - 1

        # Extraemos la informacion de cada job, en el indice 0 esta el depot
        # Move to data preparation

        if "skills" in self.jobs[1].keys():
            self.job_Skills = [self.jobs[i + 1]["skills"] for i in range(self.n_Jobs)]
        else:
            self.job_Skills = [0 for _ in range(self.n_Jobs)]
        self.demands = [self.jobs[i]["load"] for i in range(self.n_Jobs + 1)]
        self.overall_Demand = sum(self.demands)
        self.scaled_Demands = self.getScaledDemands()
        self.time_Windows = [
            self.jobs[i]["time_windows"] for i in range(self.n_Jobs + 1)
        ]
        self.service_Times = [self.jobs[i]["service"] for i in range(self.n_Jobs + 1)]

        # Definimos las capacidades y skills de la flota
        # Move to data preparation

        self.capacities = sorted([int(i) for i in self.fleet_Comp.keys()])
        self.skills_Per_VehicleType = dict.fromkeys(self.capacities, 0)
        self.addVehicleSkills()

        self.initial_Cost = sum(
            [2 * self.matrix[0][i] for i in range(1, len(self.matrix))]
        )
        self.checkRestrictions()

    def __str__(self) -> str:
        return f"\nNumber of jobs: {self.n_Jobs}\nFleet: {self.fleet_Comp}\n" + "#" * 40

    """ Agrega las skills a los vehiculos que se desee """
    """ Check this function used? """

    def addVehicleSkills(self) -> None:
        if 1000 in self.capacities:
            self.skills_Per_VehicleType[1000] = 1

    """ Return the skills of a vehicle """

    def getSkills(self, capacity: int) -> list[int]:
        return self.skills_Per_VehicleType[capacity]

    """ Return the vehicles with skills to service a job (used in RandR) """

    def getSkilledVehicles(self, job: int) -> list[int]:
        # Si el job no tiene skills, todos los vehiculos son aptos
        if not self.job_Skills[job - 1]:
            return self.capacities
        skilled_Vehicles = _copyList(self.capacities)
        for capacity in self.capacities:
            capacity_Skills = self.getSkills(capacity)
            if self.job_Skills[job - 1] > capacity_Skills:
                skilled_Vehicles.remove(capacity)
                break
        return skilled_Vehicles

    """ Return the number of skilled vehicles for a job (used in RandR) """

    def vehiclesInTypeList(self, skilledVehicles: list[str]) -> int:
        return sum([self.fleet_Comp[str(capacity)] for capacity in skilledVehicles])

    """ Return normalized distances matrix """

    def getScaledMatrix(self) -> list:
        filtered_Matrix = [
            self.matrix[i][j]
            for i in range(len(self.matrix))
            for j in range(len(self.matrix))
        ]
        matrix_Min_Max = (
            min(filtered_Matrix),
            max(filtered_Matrix),
        )
        return [
            [
                Instance.normalize(cost, matrix_Min_Max) if i != j else 0
                for j, cost in enumerate(row)
            ]
            for i, row in enumerate(self.matrix)
        ]

    """ Return normalized demands list """

    def getScaledDemands(self) -> list:
        filtered_Demands = [demand for demand in self.demands if demand > 0]
        demands_Min_Max = (
            min(filtered_Demands),
            max(filtered_Demands),
        )
        return [
            Instance.normalize(demand, demands_Min_Max) if i != 0 else 0
            for i, demand in enumerate(self.demands)
        ]

    """ Validation of instance  """

    def checkRestrictions(self) -> None:
        for demand in self.demands:
            if demand < 0:
                print("Negative demand")
                raise "Invalid instance"
        for vehicles in self.fleet_Comp.values():
            if int(vehicles) < 0:
                print("Negative number of vehicles")
                raise "Invalid instance"
        for job_Time_Windows in self.time_Windows:
            for time_Window in job_Time_Windows:
                if time_Window[1] < time_Window[0]:
                    print("End time lower than start time")
                    raise "Invalid instance"
        for service_Time in self.service_Times:
            if service_Time < 0:
                print("Negative service time")
                raise "Invalid instance"
        for capacity in self.capacities:
            if capacity < 0:
                print("Negative capacity")
                raise "Invalid instance"
        # print("Instance is ok")

    @staticmethod
    def normalize(value: float, min_Max: tuple) -> float:
        return (value - min_Max[0]) / (min_Max[1] - min_Max[0])


class Sol:

    # inst: an instance object, route_VehicleCapacity: a list with all the routes and the capacity of its vehicles [[capacity,[route]], ...]
    def __init__(self, inst: object, route_VehicleCapacity: list) -> None:
        self.inst = inst
        self.genRoutesDict(route_VehicleCapacity)

    def __str__(self) -> str:
        text = f"\nTotal Cost = {round(self.total_Cost)}\nUnassigned Jobs = {self.unassigned_Jobs}\nUsed Fleet = {self.used_Fleet}\n"
        return text

    def summary(self) -> dict:
        return {
            "Total_Cost": round(self.total_Cost, 1),
            "Unassigned_Jobs": self.unassigned_Jobs,
            "Used_Fleet": self.used_Fleet,
            "N_Jobs": self.inst.n_Jobs,
            "Overall_Demand": self.inst.overall_Demand,
        }

    def showRoutes(self) -> None:
        print("Routes description:\n")
        for route_Dict in self.routes_Dict_List:
            pprint(route_Dict)

    def genRoutesDict(self, route_VehicleCapacity_List: list) -> None:

        routes_Dict_List = []
        job_RouteID = [-1 for i in range(len(self.inst.jobs) - 1)]
        unassigned_Jobs = [i for i in range(1, len(self.inst.jobs))]
        used_Fleet = dict((k, 0) for k in self.inst.fleet_Comp.keys())
        total_Cost = 0

        for index, route_VehicleCapacity in enumerate(route_VehicleCapacity_List):
            match, route_Times, unsupportedNode = Sol.getRouteTimes(
                self.inst, route_VehicleCapacity[1]
            )
            route_Dict = {
                "ID": index,
                "Route": route_VehicleCapacity[1],
                "Cost": Sol.getRouteCost(self.inst.matrix, route_VehicleCapacity[1]),
                "Length": len(route_VehicleCapacity[1]),
                "Demand": Sol.routeDemand(self.inst.demands, route_VehicleCapacity[1]),
                "Times": route_Times,
                "Vehicle_Capacity": int(route_VehicleCapacity[0]),
            }
            routes_Dict_List.append(route_Dict)
            total_Cost += route_Dict["Cost"]
            used_Fleet[str(route_VehicleCapacity[0])] += 1

            # jobs are defined as assigned, and the id of their route is saved
            for job in route_VehicleCapacity[1]:
                unassigned_Jobs.remove(job)
                job_RouteID[job - 1] = index

        self.total_Cost = total_Cost
        self.unassigned_Jobs = unassigned_Jobs
        self.job_RouteID = job_RouteID
        self.used_Fleet = used_Fleet
        self.routes_Dict_List = routes_Dict_List
        self.updateRoutesID()
        # print("Initial solution generated:", self)

    """ Update values """

    def updateRoutesID(self) -> None:
        self.routes_Dict_List.sort(key=lambda d: d["Length"], reverse=True)
        for index, route_Dict in enumerate(self.routes_Dict_List):
            route_Dict["ID"] = index
            self.redefineJobRouteID(route_Dict["Route"], index)

    def updateTotalCost(self) -> None:
        total_Cost = 0
        for route_Dict in self.routes_Dict_List:
            total_Cost += route_Dict["Cost"]
        total_Cost += self.unassignedJobsCost()
        self.total_Cost = total_Cost

    def redefineJobRouteID(self, route_List: list, ID: int) -> None:
        for job in route_List:
            self.job_RouteID[job - 1] = ID

    def reduceVehicleCapacity(self) -> object:
        # Check fleet availability
        remaining_Fleet = {}
        for key, value in self.inst.fleet_Comp.items():
            if value > self.used_Fleet[f"{key}"]:
                remaining_Fleet[f"{key}"] = value - self.used_Fleet[f"{key}"]
            else:
                remaining_Fleet[f"{key}"] = 0
        if sum(remaining_Fleet.values()) == 0:
            return False
        modified = 0
        # Search unjustified capacity vehicle
        for route_ID, route_Dict in enumerate(self.routes_Dict_List):
            capacity = int(route_Dict["Vehicle_Capacity"])
            capacity_Index = self.inst.capacities.index(capacity)
            if capacity_Index == 0:
                continue
            for index in range(capacity_Index + 1):
                # La capacidad esta injustificada
                smaller_Capacity = self.inst.capacities[index]
                if route_Dict["Demand"] < smaller_Capacity:
                    # Si hay vehículos de esta capacidad disponibles, reducir
                    if remaining_Fleet[str(smaller_Capacity)] > 0:
                        # print(self.used_Fleet)
                        self.used_Fleet[str(smaller_Capacity)] += 1
                        self.used_Fleet[str(capacity)] -= 1
                        self.routes_Dict_List[route_ID][
                            "Vehicle_Capacity"
                        ] = smaller_Capacity
                        modified = 1
                        # print(self.used_Fleet)
                        break
        # if modified:
        #     print("*" * 60 + f"\nFleet reduced\n\nUsed Fleet= {self.used_Fleet}\n")
        return modified

    """ Solution modification """

    """Just replace a route dict for another and redefine his jobs job_RouteID"""

    def updateRoute(self, new_Route_Dict: dict) -> None:
        self.routes_Dict_List[new_Route_Dict["ID"]] = new_Route_Dict
        for job in new_Route_Dict["Route"]:
            self.job_RouteID[job - 1] = new_Route_Dict["ID"]

    """Remove an entire route from solution, all his jobs are unassigned and update route IDs"""

    def removeRoute(self, route_To_Remove_ID: int) -> None:
        removed_Route_Dict = self.routes_Dict_List.pop(route_To_Remove_ID)
        self.used_Fleet[f'{removed_Route_Dict["Vehicle_Capacity"]}'] -= 1
        self.unassigned_Jobs.extend(removed_Route_Dict["Route"])
        self.redefineJobRouteID(removed_Route_Dict["Route"], -1)
        self.updateRoutesID()

    """This functions must be refactored"""

    def removeJob(self, job: int) -> None:
        job_Route_Dict = self.routes_Dict_List[self.job_RouteID[job - 1]]
        route_To_Remove = [i for i in job_Route_Dict["Route"]]
        route_To_Remove.remove(job)
        if not route_To_Remove:
            if self.isRemovable(job_Route_Dict["Vehicle_Capacity"]):
                self.removeRoute(job_Route_Dict["ID"])
                return True
            else:
                return False

        feasible, new_Route_Dict = Sol.routeIsFeasible(
            self.inst,
            route_To_Remove,
            job_Route_Dict["Vehicle_Capacity"],
        )
        if not feasible:
            print(f"reason ={new_Route_Dict}")
            print("old route dict =", self.routes_Dict_List[job_Route_Dict["ID"]])
            raise "Relaxation error, route unfeasible"

        new_Route_Dict["ID"] = job_Route_Dict["ID"]
        self.unassigned_Jobs.append(job)
        self.job_RouteID[job - 1] = -1
        self.updateRoute(new_Route_Dict)
        return True

    """ The old route is not considered in this function, just replaced with the new one
    This function is mainly for swaping route segments and insert unassigned jobs """

    def swapRoute(self, new_Route_Dict: dict) -> None:
        self.routes_Dict_List[new_Route_Dict["ID"]] = new_Route_Dict
        for job in new_Route_Dict["Route"]:
            if job in self.unassigned_Jobs:
                self.unassigned_Jobs.remove(job)
            self.job_RouteID[job - 1] = new_Route_Dict["ID"]

    """ Request values """

    def unassignedJobsCost(self) -> float:
        return sum(self.inst.matrix[0][job] * 2 for job in self.unassigned_Jobs)

    def isRemovable(self, capacity: int) -> bool:
        if not self.varFleet():
            return True
        if self.used_Fleet[str(capacity)] > self.inst.fleet_Comp[str(capacity)]:
            return True
        return False

    def varFleet(self) -> bool:
        if sum(list(self.used_Fleet.values())) > sum(
            list(self.inst.fleet_Comp.values())
        ):
            return True
        return False

    def unjustifiedVarFleet(self) -> bool:
        if not self.varFleet():
            return False
        for v_Type in list(self.used_Fleet.keys()):
            if int(self.used_Fleet[str(v_Type)]) < int(
                self.inst.fleet_Comp[str(v_Type)]
            ):
                print("Unjustified var fleet")
                return True
        return False

    """ Feasible solution check """

    def checkSolution(self) -> None:
        error = False
        total_Cost = 0
        unassigned_Jobs = [i for i in range(1, len(self.inst.jobs))]
        used_Fleet = dict((k, 0) for k in self.inst.fleet_Comp.keys())
        try:
            for route_Dict in self.routes_Dict_List:
                route = route_Dict["Route"]
                capacity = route_Dict["Vehicle_Capacity"]
                route_Demand = Sol.routeDemand(self.inst.demands, route)
                if round(route_Demand) != round(route_Dict["Demand"]):
                    print(f"Unfeasible solution for instance {self.inst.file}")
                    print(
                        f"route_Demand is not correct, {route_Demand} != ",
                        route_Dict["Demand"],
                    )
                    error = True

                feasible, equivalent_Route_Dict = Sol.routeIsFeasible(
                    self.inst, route, capacity
                )

                if feasible:
                    used_Fleet[f"{capacity}"] += 1
                    total_Cost += equivalent_Route_Dict["Cost"]
                    for job in route:
                        unassigned_Jobs.remove(job)
                        if self.job_RouteID[job - 1] != route_Dict["ID"]:
                            print(f"Unfeasible solution for instance {self.inst.file}")
                            print(
                                f"Job route id from job {job} not clear: {self.job_RouteID[job - 1]} != {route_Dict['ID']}: Job-RouteID not corresponded"
                            )
                            error = True
                else:
                    print(
                        "\n" * 5
                        + f"ROUTE {route_Dict['ID']} FROM FILE {self.inst.file} IS NOT FEASIBLE, REASON: {equivalent_Route_Dict}"
                        + "\n" * 5
                    )
                    if equivalent_Route_Dict == 1:
                        print(
                            f"Unfeasible solution for instance {self.inst.file}: Unsupported capacity"
                        )
                    elif equivalent_Route_Dict == 2:
                        print(
                            f"Unfeasible solution for instance {self.inst.file}: Unsupported skills"
                        )
                    elif equivalent_Route_Dict == 3:
                        print(
                            f"Unfeasible solution for instance {self.inst.file}: Unmet time windows"
                        )
                    error = True

            if used_Fleet != self.used_Fleet:
                print(
                    f"Used fleet {used_Fleet} != {self.used_Fleet}: Incorrect used fleet"
                )
                error = True
            if len(unassigned_Jobs) != 0:
                print(
                    f"Unfeasible solution for instance {self.inst.file}: Unassigned jobs"
                )
                error = True
            if round(total_Cost) != round(self.total_Cost):
                print(
                    f"{round(total_Cost, 2)} != {round(self.total_Cost, 2)}: Incorrect total cost"
                )
            if error:
                raise ValueError
        except:
            # self.showRoutes()
            # print(self)
            raise ValueError

    def postProcess(self) -> None:
        for index, route_Dict in enumerate(self.routes_Dict_List):
            self.routes_Dict_List[index]["Times"] = [
                list(map(int, tw)) for tw in route_Dict["Times"]
            ]
            self.routes_Dict_List[index]["Cost"] = int(round(route_Dict["Cost"], 1))

    """ Unused """

    def removeEmptyRoutes(self) -> None:
        for route_Dict in self.routes_Dict_List:
            if route_Dict["Length"] == 0:
                self.removeRoute(route_Dict["ID"])

    def getRoutesAsList(self) -> list:
        routes_List = list()
        for route_Dict in self.routes_Dict_List:
            routes_List.append(route_Dict["Route"])
        return routes_List

    def reinsertRoute(self, route_To_Reinsert_Dict: dict) -> None:
        self.routes_Dict_List.append(route_To_Reinsert_Dict)
        self.used_Fleet[f'{route_To_Reinsert_Dict["Vehicle_Capacity"]}'] += 1
        self.total_Cost += route_To_Reinsert_Dict["Cost"]
        self.updateRoutesID()

    def replaceRoute(self, route_To_Replace_ID: int) -> None:
        print("Route removed, ID =", route_To_Replace_ID)
        route_To_Replace_Dict = self.routes_Dict_List[route_To_Replace_ID]
        self.used_Fleet[f'{route_To_Replace_Dict["Vehicle_Capacity"]}'] -= 1
        self.total_Cost -= route_To_Replace_Dict["Cost"]
        moved_Route_Dict = self.routes_Dict_List.pop()
        moved_Route_Dict["ID"] = route_To_Replace_ID
        self.routes_Dict_List[route_To_Replace_ID] = moved_Route_Dict
        self.redefineJobRouteID(moved_Route_Dict["Route"], route_To_Replace_ID)

    """ Static methods """

    @staticmethod
    def getRouteCost(matrix: list, route: list) -> float:
        if not route:
            return 0
        cost = matrix[0][route[0]] + matrix[0][route[-1]]
        for i in range(1, len(route)):
            cost += matrix[route[i - 1]][route[i]]
        return cost

    @staticmethod
    def routeDemand(demands: list, route: list) -> float:
        return round(sum([demands[i] for i in route]), 1)

    @staticmethod
    def satisfiedSkills(inst, route: list, capacity: int) -> bool:
        vehicle_Skills = inst.getSkills(capacity)
        if vehicle_Skills:
            return True
        for job in route:
            if inst.job_Skills[job - 1]:
                return False
        return True

    @staticmethod
    def routeIsFeasible(
        inst, route: list, capacity: int, verbose: bool = False
    ) -> tuple[bool, dict]:

        if not route:
            return True, {
                "Cost": 0,
                "Length": 0,
                "Route": route,
                "Vehicle_Capacity": capacity,
                "Demand": 0,
            }

        route_Demand = Sol.routeDemand(inst.demands, route)
        len_Route = len(route)

        if route_Demand > capacity or len_Route > inst.max_Jobs_Per_Route:
            return False, 1

        elif not Sol.satisfiedSkills(inst, route, capacity):
            return False, 2

        match, routeTimes, UnsupportedNode = Sol.getRouteTimes(inst, route)
        if not match:
            if verbose:
                print(
                    f"Unmet TW, Route: {route}, time windows: {routeTimes}\nJob: {UnsupportedNode}, TW: {inst.time_Windows[UnsupportedNode]}"
                )
            return False, 3

        route_Dict = {
            "Route": route,
            "Length": len_Route,
            "Demand": route_Demand,
            "Times": routeTimes,
            "Vehicle_Capacity": capacity,
            "Cost": Sol.getRouteCost(inst.matrix, route),
        }

        return True, route_Dict

    """ Append is faster for short routes, over 30 is faster creating first the entire list """

    @staticmethod
    def getRouteTimes(inst, route) -> tuple[bool, list[int], int]:

        unsupportedJob = None
        depot_Departure_Time = inst.time_Windows[0][0][0]
        route_Times = [[0, depot_Departure_Time]]

        for index, job in enumerate(route):

            match = False
            # If the node is the first one
            if index == 0:
                travel_Time = inst.matrix[0][job] / inst.speed_Factor
            else:
                previous_Job = route[index - 1]
                travel_Time = inst.matrix[previous_Job][job] / inst.speed_Factor
            job_Arrival_Time = route_Times[index][1] + travel_Time

            for index2, time_Window in enumerate(inst.time_Windows[job]):
                # Every time window is checked
                if job_Arrival_Time <= time_Window[1]:
                    window = index2
                    match = True
                    break

            if not match:
                unsupportedJob = job
                return match, route_Times, unsupportedJob

            if job_Arrival_Time < inst.time_Windows[job][window][0]:
                job_Departure_Time = (
                    inst.time_Windows[job][window][0] + inst.service_Times[job]
                )
            else:
                job_Departure_Time = job_Arrival_Time + inst.service_Times[job]

            route_Times.append([job_Arrival_Time, job_Departure_Time])

        depot_Arrival_Time = (
            route_Times[-1][1] + inst.matrix[route[-1]][0] / inst.speed_Factor
        )
        route_Times.append([depot_Arrival_Time])

        return match, route_Times, unsupportedJob


def main():

    file = "2022-07-20"  # "2021-05-25"
    fleet = {"1000": 25, "3000": 25, "5000": 25}  # {"1000": 25, "3000": 25, "5000": 25}
    # speed_Factor = 0.28
    # max_Jobs_Per_Route=max_Jobs_Per_Route

    file_Path = folder.original / f"{file}.json"
    with file_Path.open() as f:
        data = json.load(f)

    inst = Instance(data=data, file=file, fleet=fleet)
    print(inst.jobs[176])


if __name__ == "__main__":
    main()

    # result = timeit.repeat(partial(Sol.getRouteTimes, ruta=ruta), number=1000)
    # print(f"Min: {min(result)}, Max: {max(result)}, Median: {np.mean(result)}")

    # print("Route times 1:", timeit.timeit(lambda: Sol.getRouteTimes(inst, ruta)))
    # with cProfile.Profile() as pr:
    #     Sol.getRouteTimes2(inst=inst, route=ruta)
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
