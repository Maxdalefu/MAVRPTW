from copy import deepcopy
import json

from solution import Instance, Sol
from utils.folder import Folder

"""
The fleet_Comp order should not afect
"""

folder = Folder()


def routeSavings(matrix, route):
    return sum(
        [
            calculateSavings(matrix, route[i], route[i + 1])
            for i in range(len(route) - 1)
        ]
    )


def isLimit(job, route):
    return route[0] == job or route[len(route) - 1] == job


def calculateSavings(matrix, actualJobID, candidateJobID):
    return (
        matrix[0][actualJobID]
        + matrix[0][candidateJobID]
        - matrix[actualJobID][candidateJobID]
    )


def getSavingsList(matrix, n_Jobs):
    return [
        [calculateSavings(matrix, i, j), i, j]
        for i in range(1, n_Jobs)
        for j in range(i + 1, n_Jobs + 1)
    ]


def mergeIsFeasible(job_x, job_y, route_x_ID, route_y_ID, route_x, route_y):
    if route_x_ID == route_y_ID:
        return False
    elif not (isLimit(job_x, route_x) and isLimit(job_y, route_y)):
        return False
    else:
        return True


def mergeJobs(
    inst,
    routes,
    job_x,
    job_y,
    job_Route_ID,
    modified_Routes_Bin,
    capacity,
    verbose=False,
):
    merge = False
    # Route ID of each job
    route_x_ID = job_Route_ID[job_x - 1]
    route_y_ID = job_Route_ID[job_y - 1]
    # Route position on routes list is his ID - 1
    route_x = routes[route_x_ID - 1]
    route_y = routes[route_y_ID - 1]

    # Check if nodes are not in the same route and both are start/end of route
    if mergeIsFeasible(job_x, job_y, route_x_ID, route_y_ID, route_x, route_y):
        route_w = deepcopy(route_x)
        route_z = deepcopy(route_y)
        # If job_x is at the beggining of the route
        if route_w.index(job_x) == 0:
            route_w.reverse()
        # And job_y at the end
        if route_z.index(job_y) == len(route_z) - 1:
            route_z.reverse()
        route_w.extend(route_z)

        # Check for problem restrictions
        feasible, new_Route_Dict = Sol.routeIsFeasible(inst, route_w, capacity)
        if feasible:
            merge = True
            route_x = route_w
            # if verbose:
            #     print(f"Route {route_x_ID} merged with route {route_y_ID} resulting in: {route_x}")
            #     print(f"Route {route_x_ID} demand:", Sol.routeDemand(demands, route_x))
            #     print(f"ID: {route_x_ID}  |  Route: {route_x}  |  Demand:  {Sol.routeDemand(demands, route_x)}  |  Jobs merged: {job_x}-{job_y}\n")

            # new route_x modified in routes list
            routes[route_x_ID - 1] = route_x
            # route ID modified in job_Route_ID list for every job originally in route_y
            for i in route_y:
                job_Route_ID[i - 1] = route_x_ID

            # Both routes are registered as modified
            modified_Routes_Bin[route_x_ID - 1] = 1
            modified_Routes_Bin[route_y_ID - 1] = 1

            return merge, routes, job_Route_ID, modified_Routes_Bin

        # If the previous route wasn´t feasible we try extendeing the route backwards
        route_w = deepcopy(route_x)
        route_z = deepcopy(route_y)
        # If job_x is at the end of the route
        if route_w.index(job_x) == len(route_w) - 1:
            route_w.reverse()
        # And job_y at the beggining
        if route_z.index(job_y) == 0:
            route_z.reverse()
        route_z.extend(route_w)

        # Check for problem restrictions
        feasible, new_route_Dict = Sol.routeIsFeasible(inst, route_z, capacity)

        if feasible:
            merge = True
            route_y = route_z
            # if verbose:
            #     print(f"Route {route_y_ID} merged with route {route_x_ID} resulting in: {route_z}")
            #     print(f"Route {route_y_ID} demand:", Sol.routeDemand(demands, route_y))
            #     print(f"ID: {route_y_ID}  |  Route: {route_y}  |  Demand:  {Sol.routeDemand(demands, route_y)}  |  Jobs merged: {job_y}-{job_x}\n")

            routes[route_y_ID - 1] = route_y
            for i in route_x:
                job_Route_ID[i - 1] = route_y_ID
            # Both routes are registered as modified
            modified_Routes_Bin[route_x_ID - 1] = 1
            modified_Routes_Bin[route_y_ID - 1] = 1

    return merge, routes, job_Route_ID, modified_Routes_Bin


def genRoutes(inst, verbose=False):

    # print("Generating routes...\n")

    """Generated data"""
    # n_Jobs = total number of jobs discounting the depot
    n_Jobs = len(inst.jobs) - 1
    # assigned_Jobs_Bin = [0, 0, 0, ..., 0] len n_Jobs and binary values
    assigned_Jobs_Bin = [0 for i in range(n_Jobs)]
    # assigned_Routes_ID = [], here we save the IDs of the routes with an assigned vehicle
    assigned_Routes_ID = []
    # Every job starts with an independent route
    routes = [[i] for i in range(1, n_Jobs + 1)]
    modified_Routes_Bin = assigned_Jobs_Bin.copy()
    # List of route IDs, position + 1 for the job and value for de route ID
    job_Route_ID = [i for i in range(1, n_Jobs + 1)]
    # If a job demand is bigger than the smallest vehicle, it is prioritized
    unassigned_Priority_Jobs = [
        job for job in range(n_Jobs + 1) if inst.demands[job] > inst.capacities[0]
    ]
    # if unassigned_Priority_Jobs:
    #     print("Priority job IDs:", unassigned_Priority_Jobs)
    # vehicle capacity for the value and route ID for the position
    routeID_VehicleCapacity = assigned_Jobs_Bin.copy()
    # List of savings and jobs: [savings, job_x, job_y]
    savings_List = getSavingsList(inst.matrix, n_Jobs)
    sorted_Savings_List = sorted(savings_List, reverse=True)
    used_Fleet_Comp = {capacity: 0 for capacity in inst.capacities}
    route_VehicleCapacity = []
    vehicles_Availability = [
        int(inst.fleet_Comp[str(capacity)]) for capacity in inst.capacities
    ]

    # First we assign the priority jobs
    for priority_Job in unassigned_Priority_Jobs:
        for capacity in inst.capacities[::-1]:
            route_ID = job_Route_ID[priority_Job - 1]
            route = routes[route_ID - 1]
            route_Demand = Sol.routeDemand(inst.demands, route)
            # print(
            #     f"Trying ot assign route: {route}, vehicle capacity: {capacity} job demand: {route_Demand}, job skills: {inst.jobs_Skills[priority_Job-1]}"
            # )
            feasible, new_Route_Dict = Sol.routeIsFeasible(inst, route, capacity)
            if feasible:
                for job in route:
                    assigned_Jobs_Bin[job - 1] = 1
                routeID_VehicleCapacity[route_ID - 1] = capacity
                route_VehicleCapacity.append([capacity, routes[route_ID - 1]])
                used_Fleet_Comp[capacity] += 1
                assigned_Routes_ID.append(route_ID)
                # unassigned_Priority_Jobs.remove(job)
                vehicles_Availability[-1] -= 1
                break
            # else:
            #     print("Unfeasible priority job assignment, reason:", new_Route_Dict)

    for capacity in inst.capacities[::-1]:

        capacity_Index = inst.capacities.index(capacity)
        # used_Fleet_Comp[capacity] = 0

        # All unassigned jobs take initial values again (and his routes)
        for i in range(n_Jobs):
            # If the job i+1 is not assigned
            if not assigned_Jobs_Bin[i]:
                # The route in the position i becomes the initial route again ([i+1])
                routes[i] = [i + 1]
                # And the route ID in the position i becomes i+1 as well
                job_Route_ID[i] = i + 1

        # Lets search a merge in decreasing order of savings
        for i in range(len(sorted_Savings_List)):
            # sorted_Savings_List[i] = [savings, job_x, job_y]
            job_x = sorted_Savings_List[i][1]
            job_y = sorted_Savings_List[i][2]
            if assigned_Jobs_Bin[job_x - 1] or assigned_Jobs_Bin[job_y - 1]:
                # if verbose:
                #       print(f"Job {job_x} or {job_y} alredy assigned")
                continue
            merged, routes, job_Route_ID, modified_Routes_Bin = mergeJobs(
                inst,
                routes,
                job_x,
                job_y,
                job_Route_ID,
                modified_Routes_Bin,
                capacity,
            )

        uniqueRouteIDs = sorted(list(set(job_Route_ID)))

        if capacity_Index == 0:

            # If there are unassigned priority jobs, the solution is unfeasible
            # if unassigned_Priority_Jobs:
            #     print("Unassigned priority job")
            #     raise ValueError
            # All routes are checked
            for route_ID in uniqueRouteIDs:

                if route_ID in assigned_Routes_ID:
                    continue

                # And all are assigned
                else:
                    for job in routes[route_ID - 1]:
                        assigned_Jobs_Bin[job - 1] = 1
                    routeID_VehicleCapacity[route_ID - 1] = capacity
                    route_VehicleCapacity.append([capacity, routes[route_ID - 1]])
                    used_Fleet_Comp[capacity] += 1
                    assigned_Routes_ID.append(route_ID)
                    vehicles_Availability[capacity_Index] -= 1

        # For heterogeneous fleet, we priorize routes per savings
        else:

            unique_Modified_Routes = [
                routes[i - 1] for i in uniqueRouteIDs if modified_Routes_Bin[i - 1]
            ]
            unique_Modified_Routes_ID = [
                i for i in uniqueRouteIDs if modified_Routes_Bin[i - 1]
            ]

            total_Savings_Per_Route = [
                round(routeSavings(inst.matrix, route), 1)
                for route in unique_Modified_Routes
            ]
            sorted_Savings_Per_Route = sorted(total_Savings_Per_Route, reverse=True)
            sorted_Routes_Per_Savings = [
                unique_Modified_Routes_ID[total_Savings_Per_Route.index(savings)]
                for savings in sorted_Savings_Per_Route
            ]

            for route_ID in sorted_Routes_Per_Savings:

                if vehicles_Availability[capacity_Index] <= 0:
                    break

                if route_ID in assigned_Routes_ID:
                    continue

                # Vehicle capacity unused
                # if (
                #     Sol.routeDemand(inst.demands, routes[route_ID - 1])
                #     < inst.capacities[capacity_Index - 1]
                # ):
                #     continue

                else:

                    for job in routes[route_ID - 1]:
                        assigned_Jobs_Bin[job - 1] = 1
                    routeID_VehicleCapacity[route_ID - 1] = capacity
                    route_VehicleCapacity.append([capacity, routes[route_ID - 1]])
                    used_Fleet_Comp[capacity] += 1
                    assigned_Routes_ID.append(route_ID)
                    vehicles_Availability[capacity_Index] -= 1

            # print("Assigned routes ID:", assigned_Routes_ID)
            # print("Assgined jobs:", assigned_Jobs_Bin)
            # print("Total assigned:", sum(assigned_Jobs_Bin))
            # print("\n" + "*" * 50)

    solution = Sol(inst, route_VehicleCapacity)
    # solution.checkSolution()
    return solution


def main():

    problem = "solomon"
    file = "R105"  # "2021-05-25"
    fleet = {"1000": 25, "3000": 25, "5000": 25}
    variation = "Original"
    input_Folder_Path = folder.SOLOMON_INSTANCES_DIR  # \\{variation}

    with open(f"{input_Folder_Path}//{file}.json", "r") as f:
        data = json.load(f)

    inst = Instance(data=data, file=file)
    # used_Fleet_Comp = {str(capacity): 0 for capacity in inst.capacities}
    # print(used_Fleet_Comp)
    solution = genRoutes(inst)
    # solution.showRoutes()
    solution.checkSolution()
    print(solution)


if __name__ == "__main__":
    main()
