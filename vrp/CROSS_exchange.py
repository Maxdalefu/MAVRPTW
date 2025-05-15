from pathlib import Path
import json
import cProfile, pstats

from MACYW import genRoutes
from solution import Instance, Sol
from utils.folder import Folder


folder = Folder()

"""For strict var fleet reduction a verification is missing in the improvement condition"""


# FOLDER_PATH = str(Path(__file__).resolve().parent.parent)
# DATA_PATH = FOLDER_PATH + f"\\Benchmarks\\"


"""Nested lists and dicts copier"""

# def _copy_list(l, dispatch):
#     ret = l.copy()
#     for idx, item in enumerate(ret):
#         cp = dispatch.get(type(item))
#         if cp is not None:
#             ret[idx] = cp(item, dispatch)
#     return ret


# def _copy_dict(d, dispatch):
#     ret = d.copy()
#     for key, value in ret.items():
#         cp = dispatch.get(type(value))
#         if cp is not None:
#             ret[key] = cp(value, dispatch)
#     return ret


# def dipcopy(sth):
#     _dispatcher = {}
#     _dispatcher[list] = _copy_list
#     _dispatcher[dict] = _copy_dict
#     cp = _dispatcher.get(type(sth))
#     if cp is None:
#         return sth
#     else:
#         return cp(sth, _dispatcher)


"""Segments generator for the cross X with one for loop"""

# def genSegCombs(len1, L_Max):

#     nodes1 = [i for i in range(len1)]
#     combs_Seg = combinations(nodes1, 4)
#     return filterfalse(
#         lambda x: x[1] - x[0] > L_Max or x[3] - x[2] > L_Max,
#         combs_Seg,
#     )

# All combinations for segments are generated
# segs_Combs = genSegCombs(len(route_0), L_Max)
# for segs_Comb in segs_Combs:
# a, b, c, d = segs_Comb[0], segs_Comb[1], segs_Comb[2], segs_Comb[3]

# def genSegsCombs(len1, len2, L_Max):

#     nodes1, nodes2 = [i for i in range(len1 + 1)], [i for i in range(len2 + 1)]
#     combs_Seg1, combs_Seg2 = combinations_with_replacement(
#         nodes1, 2
#     ), combinations_with_replacement(nodes2, 2)
#     filtered_Combs_Seg1, filtered_Combs_Seg2 = filterfalse(
#         lambda x: x[1] - x[0] > L_Max, combs_Seg1
#     ), filterfalse(lambda x: x[1] - x[0] > L_Max, combs_Seg2)
#     return product(filtered_Combs_Seg1, filtered_Combs_Seg2)

# All combinations for segments are generated
# segs_Combs = genSegsCombs(len(route_1) + 1, len(route_2) + 1, L_Max)

# for segs_Comb in segs_Combs:
#     a, b, c, d = segs_Comb[0][0], segs_Comb[0][1], segs_Comb[1][0], segs_Comb[1][1]


def _copyList(l: list) -> list:
    ret = l.copy()
    for idx, item in enumerate(ret):
        ret[idx] = item
    return ret


def _copyDict(d: dict) -> dict:
    ret = d.copy()
    for key, value in ret.items():
        ret[key] = value
    return ret


def swap(list1: list, list2: list, a: int, b: int, c: int, d: int):
    #                  a        b                       c               d
    # list1 = [x1, x2, x3..., xn-1, xn],   list2 = [y1, y2, ..., yn-3, yn-2, ...]
    new_List1 = list1[:a]
    swap_Segment1 = list1[a:b]
    end_Segment1 = list1[b:]
    new_List2 = list2[:c]
    swap_Segment2 = list2[c:d]
    end_Segment2 = list2[d:]

    new_List1.extend(swap_Segment2)
    new_List1.extend(end_Segment1)
    new_List2.extend(swap_Segment1)
    new_List2.extend(end_Segment2)

    return new_List1, new_List2


def interRouteSwap(list0: list, a: int, b: int, c: int, d: int):

    new_List0 = list0[:a]
    swap_Segment1 = list0[a:b]
    middle_Segment = list0[b:c]
    swap_Segment2 = list0[c:d]
    end_Segment = list0[d:]

    new_List0.extend(swap_Segment2)
    new_List0.extend(middle_Segment)
    new_List0.extend(swap_Segment1)
    new_List0.extend(end_Segment)

    return new_List0


def exploreInterRoute(solution, route_Index: int, L_Max: int) -> None:

    route_0_Dict = solution.routes_Dict_List[route_Index]
    route_0_Len = route_0_Dict["Length"]

    for a in range(route_0_Len - 1):

        for b in range(a, route_0_Len):

            if b - a > L_Max:
                break

            for c in range(b + 1, route_0_Len):

                for d in range(c + 1, route_0_Len + 1):

                    if d - c > L_Max:
                        break

                    new_Route_0 = _copyList(route_0_Dict["Route"])
                    new_Route_0 = interRouteSwap(new_Route_0, a, b, c, d)
                    new_Route_Cost = Sol.getRouteCost(solution.inst.matrix, new_Route_0)

                    if new_Route_Cost < route_0_Dict["Cost"]:

                        match, routeTimes, UnsupportedNode = Sol.getRouteTimes(
                            solution.inst, new_Route_0
                        )
                        if match:
                            # Better route found, keep it
                            new_Route_Dict = _copyDict(route_0_Dict)
                            new_Route_Dict["ID"] = route_Index
                            new_Route_Dict["Route"] = new_Route_0
                            new_Route_Dict["Times"] = routeTimes
                            new_Route_Dict["Cost"] = new_Route_Cost
                            solution.updateRoute(new_Route_Dict)


def exploreNeighborhood(
    solution, route_1_Index: int, route_2_Index: int, L_Max: int, verbose: bool = False
) -> None:

    route_To_Save_Dict = 0
    route_1_Dict = solution.routes_Dict_List[route_1_Index]
    route_1 = route_1_Dict["Route"]
    route_1_Capacity = route_1_Dict["Vehicle_Capacity"]
    route_1_Len = route_1_Dict["Length"]

    route_2_Dict = solution.routes_Dict_List[route_2_Index]
    route_2 = route_2_Dict["Route"]
    route_2_Capacity = route_2_Dict["Vehicle_Capacity"]
    route_2_Len = route_2_Dict["Length"]

    for a in range(route_1_Len + 1):

        next_A = False

        for b in range(a, route_1_Len + 1):

            if next_A or b - a > L_Max:
                break

            for c in range(route_2_Len + 1):

                if next_A:
                    break
                next_C = False

                for d in range(c, route_2_Len + 1):

                    if next_A or next_C or d - c > L_Max:
                        break
                    if a == b and c == d:
                        continue

                    new_Route_1, new_Route_2 = _copyList(route_1), _copyList(route_2)
                    new_Route_1, new_Route_2 = swap(
                        new_Route_1,
                        new_Route_2,
                        a,
                        b,
                        c,
                        d,
                    )
                    # A solution is better when the total cost is lower or a route removed
                    # If the first route is empty
                    if not new_Route_1:
                        # It can be removed
                        if not solution.isRemovable(route_1_Capacity):
                            continue
                        # And the new second is feasible
                        feasible, new_Route2_Dict = Sol.routeIsFeasible(
                            solution.inst,
                            new_Route_2,
                            route_2_Capacity,
                        )
                        if feasible:
                            # The removal is accepted
                            route_To_Save_Dict = new_Route2_Dict
                            route_To_Save_Dict["ID"] = route_2_Index
                            route_To_Remove_Index = route_1_Index
                        else:
                            continue
                    # Same for the second route if is empty
                    elif not new_Route_2:
                        if not solution.isRemovable(route_2_Capacity):
                            continue
                        feasible, new_Route1_Dict = Sol.routeIsFeasible(
                            solution.inst,
                            new_Route_1,
                            route_1_Capacity,
                        )
                        if feasible:
                            route_To_Save_Dict = new_Route1_Dict
                            route_To_Save_Dict["ID"] = route_1_Index
                            route_To_Remove_Index = route_2_Index
                    if route_To_Save_Dict:
                        # The lower index is saved to keep the list in order
                        # if route_1_Index < route_2_Index:
                        # route_To_Save_Dict["ID"] = route_1_Index
                        #     route_To_Remove_Index = route_2_Index
                        # else:
                        #     route_To_Save_Dict["ID"] = route_2_Index
                        #     route_To_Remove_Index = route_1_Index
                        # Is updated
                        solution.updateRoute(route_To_Save_Dict)
                        # The list is removed
                        removed_Route_Dict = solution.routes_Dict_List.pop(
                            route_To_Remove_Index
                        )
                        # Remove the vehicle from the route to remove
                        solution.used_Fleet[
                            str(removed_Route_Dict["Vehicle_Capacity"])
                        ] -= 1
                        # IDs are updated
                        solution.updateRoutesID()
                        return

                    # If none is empty then we check if the total cost decrease
                    cost_New_Route1 = Sol.getRouteCost(
                        solution.inst.matrix, new_Route_1
                    )
                    cost_New_Route2 = Sol.getRouteCost(
                        solution.inst.matrix, new_Route_2
                    )
                    if (
                        cost_New_Route1 + cost_New_Route2
                        < route_1_Dict["Cost"] + route_2_Dict["Cost"]
                    ):
                        # If both routes are feasible
                        feasible, new_Route1_Dict = Sol.routeIsFeasible(
                            solution.inst, new_Route_1, route_1_Capacity
                        )
                        if feasible:
                            feasible2, new_Route2_Dict = Sol.routeIsFeasible(
                                solution.inst,
                                new_Route_2,
                                route_2_Capacity,
                            )
                            if feasible2:
                                new_Route1_Dict["ID"] = route_1_Index
                                solution.updateRoute(new_Route1_Dict)
                                new_Route2_Dict["ID"] = route_2_Index
                                solution.updateRoute(new_Route2_Dict)

                                route_1_Len = len(route_1)
                                route_2_Len = len(route_2)

                            elif new_Route2_Dict == 2:
                                next_A = True
                        else:  # new_Route1_Dict == 1 or new_Route1_Dict == 2:
                            next_C = True


def crossEx(solution, L_Max: int) -> None:

    route_X_Index = 0
    while route_X_Index < len(solution.routes_Dict_List):
        route_Y_Index = route_X_Index
        while route_Y_Index < len(solution.routes_Dict_List):
            if route_X_Index == route_Y_Index:
                exploreInterRoute(solution, route_X_Index, L_Max)

            else:
                exploreNeighborhood(solution, route_X_Index, route_Y_Index, L_Max)
            route_Y_Index += 1
        route_X_Index += 1

    solution.updateTotalCost()
    # print("Results after cross_X:", solution)
    return solution


def main() -> None:
    file = "R105"
    problem = "solomon"
    fleet = {"1000": 25, "3000": 25, "5000": 25}
    input_Folder_Path = folder.SOLOMON_INSTANCES_DIR  # \\{variation}
    with open(f"{input_Folder_Path}//{file}.json", "r") as f:
        data = json.load(f)

    inst = Instance(data=data, file=file, fleet=fleet)
    solution = genRoutes(inst)
    solution.showRoutes()

    solution = crossEx(solution, L_Max=5)
    solution.showRoutes()
    solution.checkSolution()
    # print("Time: " + str(processing_Time) + "[s]")
    # print("Improvement: " + str(round(porcental_Improvement, 1)) + "%")

    # with cProfile.Profile() as pr:
    #     crossEx(solution, L_Max=5)
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(10)


if __name__ == "__main__":
    main()
