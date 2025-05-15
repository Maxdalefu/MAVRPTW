from random import random, randint, choice, seed
import json, cProfile, pstats
from copy import deepcopy

from MACYW import genRoutes
from solution import Instance, Sol
from utils.folder import Folder


RANDOM_SEED = 18

folder = Folder()
seed(RANDOM_SEED)


"""General functions"""


def _copyList(l):
    ret = l.copy()
    for idx, item in enumerate(ret):
        ret[idx] = item
    return ret


"""RELAXATION PROCEDURE"""


"""Function used to calculate relatedness needed for the shaw removal"""


def getStartServiceTime(solution, job: int) -> float:

    job_RouteID = solution.job_RouteID[job - 1]
    if job_RouteID == -1:
        print(f"Job {job} unassigned")
        raise "Job is not assigned to a route"

    route_Dict = solution.routes_Dict_List[job_RouteID]
    job_Index = route_Dict["Route"].index(job)
    return route_Dict["Times"][job_Index + 1][1] - solution.inst.service_Times[job]


"""Calculate relatedness needed to rank jobs for relaxation"""
"""Lower value for more related jobs"""


def getRelatedness(
    solution, job_i: int, job_j: int, scaled_Start_Service_Times: list
) -> float:

    φ = 9
    χ = 3
    ψ = 2
    ω = 5

    job_i_Fleet = solution.inst.getSkilledVehicles(job_i)
    job_j_Fleet = solution.inst.getSkilledVehicles(job_j)

    relatedness = (
        φ * solution.inst.scaled_Matrix[job_i][job_j]
        + χ
        * abs(
            scaled_Start_Service_Times[job_i - 1]
            - scaled_Start_Service_Times[job_j - 1]
        )
        + ψ
        * abs(solution.inst.scaled_Demands[job_i] - solution.inst.scaled_Demands[job_j])
        + ω
        * (
            1
            - (
                solution.inst.vehiclesInTypeList(
                    [capacity for capacity in job_i_Fleet if capacity in job_j_Fleet]
                )
            )
            / (
                min(
                    solution.inst.vehiclesInTypeList(job_i_Fleet),
                    solution.inst.vehiclesInTypeList(job_j_Fleet),
                )
            )
        )
    )

    return relatedness


"""Rank in decreasing relatedness all the insertion points"""
""" Relatedness is higher for lower values """


def getRelatednessRanking(solution: object) -> list:

    N_Jobs = solution.inst.n_Jobs
    start_Service_Times = [
        getStartServiceTime(solution, i) for i in range(1, N_Jobs + 1)
    ]
    start_Service_Time_Min_Max = [min(start_Service_Times), max(start_Service_Times)]
    scaled_Start_Service_Times = [
        Instance.normalize(start_Service_Time, start_Service_Time_Min_Max)
        for start_Service_Time in start_Service_Times
    ]

    relatednessRanking = [
        [
            (getRelatedness(solution, job_i, job_j, scaled_Start_Service_Times), job_j)
            for job_j in range(1, N_Jobs + 1)
        ]
        for job_i in range(1, N_Jobs + 1)
    ]
    sortedRelatednessRanking = [
        list(list(zip(*sorted(relatednessRanking[i])))[1]) for i in range(N_Jobs)
    ]

    return sortedRelatednessRanking


"""Remove jobs from the current solution"""


def randomAssignedJob(solution: object) -> int:
    while True:
        job = randint(1, len(solution.inst.jobs) - 1)
        if job not in solution.unassigned_Jobs:
            return job


def shawRemoval(solution: object) -> None:

    # Parameters
    p, E = 6, 0.4

    # Select a random number of nodes to remove
    q = randint(4, (min(100, int(E * solution.inst.n_Jobs))))

    # Get the relatedness for all jobs
    sorted_Relatedness_Ranking = getRelatednessRanking(solution)

    # Select a random job for an initial removal
    while True:
        random_Job = randomAssignedJob(solution)
        if solution.removeJob(random_Job):
            break

    # Remove the rest of the q jobs
    for _ in range(q):
        # Removal must be successful
        while True:
            # Select randomly a relaxed job
            random_Relaxed_Job = choice(solution.unassigned_Jobs)

            # Get all jobs sorted by relatedness for the recently selected job
            random_Job_Sorted_Relatedness_Ranking = _copyList(
                sorted_Relatedness_Ranking[random_Relaxed_Job - 1]
            )

            # An unassigned job can not be selected so they are removed
            for unassigned_Job in solution.unassigned_Jobs:
                random_Job_Sorted_Relatedness_Ranking.remove(unassigned_Job)

            # p: Parameter to control diversification, p = 1 to ignore relatedness
            # The job in the r**p position is selected for removal
            r = random() ** p
            job_To_Relax = random_Job_Sorted_Relatedness_Ranking[
                int(r * len(random_Job_Sorted_Relatedness_Ranking))
            ]

            if solution.removeJob(job_To_Relax):
                break


""" Greedy insertion """


def getRouteCheapestInsertion(solution: object, job: int, route_ID: int) -> tuple:

    # n_Insertion_Points = 0
    route_Dict = solution.routes_Dict_List[route_ID]
    cheapest_Route_Insertion = [float("inf")]
    route = route_Dict["Route"]
    initial_Route_Cost = route_Dict["Cost"]
    feasible_Insertion = False

    for index in range(route_Dict["Length"] + 1):

        route_Copy = _copyList(route)
        route_Copy.insert(index, job)
        feasible, new_Route_Dict = Sol.routeIsFeasible(
            solution.inst, route_Copy, route_Dict["Vehicle_Capacity"]
        )

        # If this insertion is feasible
        if feasible:

            # The index and route ID of insertion are saved
            insertion_Cost = new_Route_Dict["Cost"] - initial_Route_Cost
            if cheapest_Route_Insertion[0] > insertion_Cost:
                new_Route_Dict["ID"] = route_ID
                cheapest_Route_Insertion = [insertion_Cost, index, new_Route_Dict]
                feasible_Insertion = True
            # n_Insertion_Points += 1

    return feasible_Insertion, cheapest_Route_Insertion


""" Calculate how constrained a job is """
""" Also saves the cheapest insertion point in every route """


def getRestrictionDegree(solution: object, job: int) -> tuple:

    cheapest_Insertion_Per_Route = []
    feasible_Insertion = False

    for route_Dict in solution.routes_Dict_List:

        route_Insertion, cheapest_Route_Insertion = getRouteCheapestInsertion(
            solution, job, route_Dict["ID"]
        )

        if route_Insertion:
            cheapest_Insertion_Per_Route.append(cheapest_Route_Insertion)
            feasible_Insertion = True

    if not feasible_Insertion:
        return feasible_Insertion, {}

    cheapest_Insertion_Per_Route.sort(key=lambda x: x[0])

    restriction_Degree = cheapest_Insertion_Per_Route[0][0]

    return feasible_Insertion, {
        "Restriction_degree": restriction_Degree,
        "Cheapest_Insertion_Per_Route": cheapest_Insertion_Per_Route,
        "Job": job,
    }


""" Sort in descending order the unassigned jobs depending on: cheapest insertion / number of insertion points
    More constrained jobs first """


def rankJobsToInsert(solution: object) -> list:

    jobs_To_Insert = []

    for job in solution.unassigned_Jobs:

        feasible_Insertion, job_Dict = getRestrictionDegree(solution, job)

        if not feasible_Insertion:
            return feasible_Insertion

        jobs_To_Insert.append(job_Dict)

    sorted_Jobs = sorted(
        jobs_To_Insert, key=lambda d: d["Restriction_degree"], reverse=True
    )
    return sorted_Jobs


"""Instantiate the jobs with Maximized cv / pv (min_insertion_increase/num_insertion_points)
sorted_Jobs: sorted list of job dictionaries {'Job': 38, 'Cheapest_Route_Insertion': [cost, index, route_ID], 'Restriction_degree': 42.5}
Here we also get the cheapest insertion point of every job, so if its route is not modified, is just inserted there"""


def greedyInsertion(solution: object) -> tuple:

    iterations, insertion_Found, modified_Routes = 0, False, []
    sorted_Jobs = rankJobsToInsert(solution)

    # Job with no insertion points
    if not sorted_Jobs:
        return

    for job_Dict in sorted_Jobs:

        iterations += 1
        job = job_Dict["Job"]
        insertion_Found = False
        current_Cheapest_Insertion = [float("inf")]
        cheapest_Insertions_Per_Route = job_Dict["Cheapest_Insertion_Per_Route"]

        # Loop through saved feasible insertion point, if a point was not feasible to insert before, wont be now
        for index, saved_Cheapest_Insertion in enumerate(cheapest_Insertions_Per_Route):

            route_ID = saved_Cheapest_Insertion[2]["ID"]
            # If the cheapest insertion route is not modified, insert the job there
            if route_ID not in modified_Routes:

                if saved_Cheapest_Insertion[0] < current_Cheapest_Insertion[0]:
                    current_Cheapest_Insertion = saved_Cheapest_Insertion

                insertion_Found = True
                break

            # Otherwise recalculate cheapest insertion point for this job in this route
            (
                feasible_Insertion,
                new_Cheapest_Route_Insertion,
            ) = getRouteCheapestInsertion(solution, job, route_ID)

            # If this job has no feasible insertion points in this route
            if not feasible_Insertion:
                continue

            # If it is cheaper than the currrent saved insertion, keep the cheaper so far
            if new_Cheapest_Route_Insertion[0] < current_Cheapest_Insertion[0]:

                current_Cheapest_Insertion = new_Cheapest_Route_Insertion
                # If is cheaper than the next saved insertion, use the current cheapest
                if index < len(cheapest_Insertions_Per_Route) - 1:
                    if (
                        current_Cheapest_Insertion[0]
                        < cheapest_Insertions_Per_Route[index + 1][0]
                    ):
                        insertion_Found = True
                        break

        # Not insertion found for this job
        if not insertion_Found:
            return insertion_Found
        else:
            solution.swapRoute(current_Cheapest_Insertion[2])
            modified_Routes.append(saved_Cheapest_Insertion[2]["ID"])

    solution.updateTotalCost()

    return insertion_Found


def ruinAndRecreate(
    current_Solution: Sol, relax: str = "route", n: int = 10
) -> Sol:

    # print("*" * 50 + f"\nRuin and Recreate procedure with {relax} removal")

    # If the solution has unassigned jobs they must be reinserted
    if current_Solution.unassigned_Jobs:
        print("Reinserting initialy relaxed nodes")
        greedyInsertion(current_Solution)

    if relax == "shaw":

        for _ in range(n):

            solution_Copy = deepcopy(current_Solution)

            shawRemoval(solution_Copy)

            greedyInsertion(solution_Copy)

            if solution_Copy.unassigned_Jobs or solution_Copy.unjustifiedVarFleet():
                continue
            if (solution_Copy.total_Cost < current_Solution.total_Cost) or (
                sum(list(solution_Copy.used_Fleet.values()))
                < sum(list(current_Solution.used_Fleet.values()))
            ):
                # New solution found
                current_Solution = solution_Copy

    # Or just start relaxing entire routes by length (shortest firts)
    elif relax == "route":

        route_To_Remove = -1
        while -route_To_Remove < len(current_Solution.routes_Dict_List):

            # Check if the route can be removed
            route_To_Remove_Capacity = current_Solution.routes_Dict_List[
                route_To_Remove
            ]["Vehicle_Capacity"]
            if not current_Solution.isRemovable(route_To_Remove_Capacity):
                route_To_Remove -= 1
                continue

            # Remove route
            solution_Copy = deepcopy(current_Solution)
            solution_Copy.removeRoute(route_To_Remove)

            greedyInsertion(solution_Copy)

            # If all nodes were reinserted, save the solution
            if not solution_Copy.unassigned_Jobs:
                current_Solution = solution_Copy
            else:
                route_To_Remove -= 1

    # print("Results after RandR:", current_Solution)
    return current_Solution


def main() -> None:

    problem = "solomon"
    file = "R105"  # "2021-05-25"
    # fleet = {"1000": 25, "3000": 25, "5000": 25}
    # variation = "Loadx1.8"

    input_Folder_Path = folder.SOLOMON_INSTANCES_DIR

    with open(f"{input_Folder_Path}//{file}.json", "r") as f:
        data = json.load(f)

    inst = Instance(data=data, file=file)
    solution = genRoutes(inst)

    # solution = ruinAndRecreate(solution, relax="shaw")

    # solution.showRoutes()
    # solution.checkSolution()
    # print(solution)

    with cProfile.Profile() as pr:
        ruinAndRecreate(solution, relax="shaw")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)


if __name__ == "__main__":
    main()
