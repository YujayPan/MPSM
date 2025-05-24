import os, sys
import time
import argparse
import numpy as np
from datetime import timedelta
from functools import partial
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool
import math
import traceback

SPEED = 1.0
SCALE = 100000  # EAS uses 1000, while AM uses 100000
TIME_HORIZON = 3  # the tw_end for the depot node, all vehicles should return to depot before T


def create_data_model(depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None, grid_size=1, problem="CVRP"):
    """
        Stores the data for the problem. Includes scaling.
    """
    data = {}
    to_int = lambda x: int(x / grid_size * SCALE + 0.5) if isinstance(x, (int, float)) and not math.isinf(x) else int(99999 * SCALE)

    data['depot'] = 0
    locations = [depot] + loc
    data['locations'] = [(to_int(x), to_int(y)) for (x, y) in locations]
    data['real_locations'] = locations
    data['num_locations'] = len(data['locations'])
    data['demands'] = [0] + [int(d) for d in demand]
    data['num_vehicles'] = max(1, len(loc)) # Use max(1, ...) to ensure at least one vehicle
    data['vehicle_capacity'] = int(capacity)

    is_open = problem.startswith("O")
    has_tw = 'TW' in problem
    has_l = 'L' in problem

    data['dummy_depot'] = None
    if is_open:
        data['dummy_depot'] = data['num_locations']

    # Precompute Distance Matrix (Scaled) first, as it might be needed for TW
    _distances = {}
    for i in range(data['num_locations']):
        _distances[i] = {}
        for j in range(data['num_locations']):
            if i == j: _distances[i][j] = 0
            else:
                loc1 = data['locations'][i]
                loc2 = data['locations'][j]
                _distances[i][j] = int(math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2) + 0.5)
    data['distance_matrix'] = _distances

    # For TW
    if has_tw:
        if not isinstance(tw_start, (list, tuple)) or not isinstance(tw_end, (list, tuple)):
             raise TypeError("tw_start and tw_end must be lists or tuples for TW problems")
        if not isinstance(service_time, (list, tuple)):
             raise TypeError("service_time must be a list or tuple for TW problems")
        # Determine horizon carefully
        valid_tw_ends = [t for t in tw_end if not math.isinf(t) and isinstance(t, (int, float))]
        # Use a default large horizon if all ends are inf or list is empty
        depot_tw_end_unscaled = max(valid_tw_ends) * 1.5 if valid_tw_ends else (TIME_HORIZON * 10) # Adjusted default
        depot_tw_end_scaled = to_int(depot_tw_end_unscaled / grid_size)
        data['time_windows'] = [(0, depot_tw_end_scaled)] + \
                               [(to_int(e / grid_size), to_int(l / grid_size)) for e, l in zip(tw_start, tw_end)]
        # Correctly create the list of scaled service times including depot
        data['service_times'] = [0] + [to_int(st / grid_size) for st in service_time] # Scaled service times list
        if len(data['time_windows']) != data['num_locations'] or len(data['service_times']) != data['num_locations']:
             raise ValueError("Length mismatch: TW/Service times vs Locations")

        # <<< Add Time Matrix Calculation >>>
        _times = {}
        num_locs = data['num_locations']
        for i in range(num_locs):
            _times[i] = {}
            for j in range(num_locs):
                if i == j: _times[i][j] = 0
                else:
                    travel_time = data['distance_matrix'][i][j] # Scaled distance
                    service_time_at_i = data['service_times'][i] # Scaled service time at origin
                    _times[i][j] = service_time_at_i + travel_time
        data['time_matrix'] = _times # Store the computed time matrix

    # For duration limit
    if has_l:
        if route_limit is None or math.isinf(route_limit):
            data['distance_limit'] = int(999999 * SCALE) # Use a large default if None or inf
        else:
            data['distance_limit'] = to_int(route_limit / grid_size)

    return data


#######################
# Problem Constraints #
#######################
def Euc_distance(position_1, position_2):
    return int(np.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2))


def create_distance_evaluator(data):
    """
        Creates callback to return distance between points.
    """
    _distances = {}
    # precompute distance between location to have distance callback in O(1)
    for from_node in xrange(data['num_locations']):
        _distances[from_node] = {}
        for to_node in xrange(data['num_locations']):
            if from_node == to_node:
                _distances[from_node][to_node] = 0
            else:
                _distances[from_node][to_node] = (Euc_distance(data['locations'][from_node], data['locations'][to_node]))
    data['distance_matrix'] = _distances

    def distance_evaluator(manager, from_index, to_index):
        """
            Returns the manhattan distance between the two nodes.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == data['dummy_depot']:  # for open route
            return 0
        else:
            return _distances[from_node][to_node]

    return distance_evaluator


def add_distance_constraints(routing, data, distance_evaluator_index):
    """
        Adds duration limit constraint.
    """
    routing.AddDimension(
        distance_evaluator_index,
        0,  # null distance slack
        data['distance_limit'],
        True,  # start cumul to zero
        'Distance')


def create_demand_evaluator(data):
    """
        Creates callback to get demands at each location.
    """
    _demands = data['demands']

    def demand_evaluator(manager, from_index):
        """
            Returns the demand of the current node.
        """
        from_node = manager.IndexToNode(from_index)
        return _demands[from_node]

    return demand_evaluator


def add_capacity_constraints(routing, data, demand_evaluator_index, problem="CVRP"):
    """
        Adds capacity constraint.
    """
    if problem in ["VRPB", "OVRPB", "VRPBL", "VRPBTW", "VRPBLTW", "OVRPBL", "OVRPBTW", "OVRPBLTW"]:  # Optional for VRPB and VRPBL
        # Note (Only for the problems with backhauls): need to relax the capacity constraint, otherwise OR-Tools cannot find initial feasible solution;
        # However, it may be problematic since the vehicle could decide how many loads to carry from depot in this case.
        routing.AddDimension(
            demand_evaluator_index,
            0,  # null capacity slack
            data['vehicle_capacity'],
            False,  # don't force start cumul to zero
            'Capacity')
    else:
        routing.AddDimension(
            demand_evaluator_index,
            0,  # null capacity slack
            data['vehicle_capacity'],
            True,  # start cumul to zero
            'Capacity')


def create_time_evaluator(data):
    """
        Creates callback to get total times between locations.
    """

    def travel_time(data, from_node, to_node):
        """
            Gets the travel times between two locations.
        """
        return int(data['distance_matrix'][from_node][to_node] / SPEED)

    _total_time = {}
    # precompute total time to have time callback in O(1)
    for from_node in xrange(data['num_locations']):
        _total_time[from_node] = {}
        for to_node in xrange(data['num_locations']):
            if from_node == to_node:
                _total_time[from_node][to_node] = 0
            elif from_node == data['depot']:  # depot node -> customer node
                _total_time[from_node][to_node] = travel_time(data, from_node, to_node)
            else:
                _total_time[from_node][to_node] = int(data['service_times'][from_node] + travel_time(data, from_node, to_node))
    data['time_matrix'] = _total_time

    def time_evaluator(manager, from_index, to_index):
        """
            Returns the total time (service_time + travel_time) between the two nodes.
        """
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if to_node == data['dummy_depot']:
            return 0
        else:
            return _total_time[from_node][to_node]

    return time_evaluator


def add_time_window_constraints(routing, manager, data, time_evaluator_index, grid_size=1):
    """
        Add Global Span constraint.
    """
    time = 'Time'
    horizon = int(TIME_HORIZON / grid_size * SCALE + 0.5)
    routing.AddDimension(
        time_evaluator_index,
        horizon,  # allow waiting time
        horizon,  # maximum time per vehicle
        False,  # don't force start cumul to zero since we are giving TW to start nodes
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    # Add time window constraints for each location except depot
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot'] or location_idx == data['dummy_depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
        routing.AddToAssignment(time_dimension.SlackVar(index))

    # Add time window constraints for each vehicle start and end node
    # and 'copy' the slack var in the solution object (aka Assignment) to print it
    for vehicle_id in xrange(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        # Cumul(depot).SetRange(0, 0) -> vehicle must be at time 0 at depot
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0], data['time_windows'][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))
        # for open route
        if data['dummy_depot']:
            index = routing.End(vehicle_id)
            time_dimension.CumulVar(index).SetRange(0, horizon)
            # Warning: Slack var is not defined for vehicle's end node
            # routing.AddToAssignment(time_dimension.SlackVar(index))


###########
# Printer #
###########
def print_solution(data, manager, routing, assignment, problem="CVRP", log_file=None):
    """
        Only print route, and calculate cost (total distance).
    """

    def calc_vrp_cost(depot, loc, tour, problem):
        assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        if problem in ["CVRP", "VRPB", "VRPL", "VRPTW", "VRPBL", "VRPLTW", "VRPBTW", "VRPBLTW"]:
            return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
        elif problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:  # no need to return to depot
            full_tour = [0] + tour + [0]
            not_to_depot = np.array(full_tour)[1:] != 0
            return (np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1) * not_to_depot).sum()
        else:
            raise NotImplementedError

    route = []
    total_distance, total_load = 0, 0
    # distance_dimension = routing.GetDimensionOrDie('Distance')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for vehicle_id in xrange(data['num_vehicles']):
        if not routing.IsVehicleUsed(vehicle=vehicle_id, assignment=assignment):
            continue
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            plan_output += ' {0} Load({1}) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var))
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            # distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)  # Bugs: always output 0 if given variable index, don't know why
            from_node, to_node = manager.IndexToNode(previous_index), manager.IndexToNode(index)
            to_node = to_node if to_node != data['dummy_depot'] else data['depot']
            distance += data['distance_matrix'][from_node][to_node]  # use distance matrix instead

        load_var = capacity_dimension.CumulVar(index)
        # dist_var = distance_dimension.CumulVar(index)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var))
        # assert distance == assignment.Value(dist_var), ">> Distance not match!"
        plan_output += 'Distance of the route: {}\n'.format(distance)
        plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
        if log_file:
            print(plan_output, file=log_file)
        total_distance += distance
        total_load += assignment.Value(load_var)

    # double check
    cost = calc_vrp_cost(data['real_locations'][0], data['real_locations'][1:], route[1:], problem)
    if log_file:
        print('Route: {}'.format(route + [data['depot']]), file=log_file)
        print('Total Load of all routes: {}'.format(total_load), file=log_file)
        print('Total Distance of all routes: {} (Routing Error may exist)'.format(total_distance / SCALE), file=log_file)
        print('Final Result - Cost of the obtained solution: {}'.format(cost), file=log_file)

    return cost, route[1:]


def solve_or_tools_log(directory, name, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None,
                       timelimit=3600, grid_size=1, seed=1234, problem="CVRP"):
    """
    OR-Tools solver function. Aligned with ortools_solver.py logic.
    Returns (cost, flat_route, duration).
    """
    log_filename = os.path.join(directory, "{}.or_tools.log".format(name)) # Keep logs separate per run
    output_filename = os.path.join(directory, "{}.or_tools.pkl".format(name)) # Intermediate file

    try:
        # 1. Create Data Model (Handles Scaling)
        data = create_data_model(depot, loc, demand, capacity, route_limit=route_limit, service_time=service_time,
                                 tw_start=tw_start, tw_end=tw_end, grid_size=grid_size, problem=problem)

        # 2. Create Routing Manager
        num_locations = data['num_locations']
        num_vehicles = data['num_vehicles']
        depot_idx = data['depot']
        is_open = problem.startswith("O")
        dummy_depot_idx = data.get('dummy_depot')
        num_manager_nodes = num_locations + 1 if is_open else num_locations

        if is_open:
            starts = [depot_idx] * num_vehicles
            ends = [dummy_depot_idx] * num_vehicles
            manager = pywrapcp.RoutingIndexManager(num_manager_nodes, num_vehicles, starts, ends)
        else:
            starts = [depot_idx] * num_vehicles
            ends = [depot_idx] * num_vehicles
            manager = pywrapcp.RoutingIndexManager(num_manager_nodes, num_vehicles, starts, ends)

        # 3. Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # 4. Add Constraints based on problem type
        # Distance Cost/Callback
        distance_evaluator_index = routing.RegisterTransitCallback(partial(create_distance_evaluator(data), manager))
        routing.SetArcCostEvaluatorOfAllVehicles(distance_evaluator_index)

        # Capacity Constraint
        demand_evaluator_index = routing.RegisterUnaryTransitCallback(partial(create_demand_evaluator(data), manager))
        add_capacity_constraints(routing, data, demand_evaluator_index, problem=problem) # Pass problem type

        # Time Windows
        if 'TW' in problem:
            if 'time_matrix' not in data:
                 print(f"Error: Time matrix not precomputed for TW problem instance {name}")
                 raise ValueError("Time matrix required for TW problems")
            time_evaluator_index = routing.RegisterTransitCallback(partial(create_time_evaluator(data), manager))
            add_time_window_constraints(routing, manager, data, time_evaluator_index)

        # Distance Limit
        if 'L' in problem:
            add_distance_constraints(routing, data, distance_evaluator_index)

        # Backhaul (Currently only handled by capacity start_cumul_to_zero=False)
        if 'B' in problem and 'TW' in problem:
             linehaul_indices = [manager.NodeToIndex(i) for i, d in enumerate(data['demands']) if d > 0]
             backhaul_indices = [manager.NodeToIndex(i) for i, d in enumerate(data['demands']) if d < 0]
             if linehaul_indices and backhaul_indices: # Only add if both exist
                 # --- Comment out the explicit precedence constraint --- START
                 # time_dimension = routing.GetDimensionOrDie('Time')
                 # for l_idx in linehaul_indices:
                 #     for b_idx in backhaul_indices:
                 #          routing.solver().Add(time_dimension.CumulVar(l_idx) <= time_dimension.CumulVar(b_idx))
                 # --- Comment out the explicit precedence constraint --- END
                 # print(f"Info: Added Backhaul precedence constraint for {problem} using Time dimension.")
                 print(f"Info: Relying on capacity dimension for Backhaul handling for {problem}.")
             #else: # Optional: Log if no constraint needed
                 #print(f"Info: No explicit Backhaul precedence constraint needed for {problem} (no L/B nodes or no TW).")

        # 5. Set Search Parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.log_search = False
        search_parameters.time_limit.seconds = timelimit
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

        # 6. Solve
        start = time.time()
        assignment = routing.SolveWithParameters(search_parameters)
        duration = time.time() - start

        # 7. Process Results (Aligned with ortools_solver.py)
        cost = float('inf')
        flat_route_final = []

        # Check status *before* accessing assignment details
        if assignment:
            cost = assignment.ObjectiveValue() / SCALE # Unscale cost

            routes_list = [] # Store routes per vehicle first
            for vehicle_id in range(num_vehicles):
                # It's generally safer to check IsVehicleUsed *if* you only want used routes,
                # but iterating and checking the route content is also robust.
                # Let's iterate all and filter empty ones later.
                index = routing.Start(vehicle_id)
                route_nodes_vehicle = []
                while True: # Loop until IsEnd
                    node_index = manager.IndexToNode(index)
                    # Exclude depot and dummy depot from the route sequence itself
                    if node_index != depot_idx and (not is_open or node_index != dummy_depot_idx):
                        route_nodes_vehicle.append(node_index)

                    if routing.IsEnd(index):
                        break

                    # Get next index safely
                    try:
                         next_index_val = assignment.Value(routing.NextVar(index))
                         # Check for potential issues like stuck solver before moving
                         if next_index_val == index:
                              print(f"Warning: Solver appears stuck at node {node_index} for vehicle {vehicle_id}. Route truncated.")
                              break
                         index = next_index_val
                    except Exception as next_var_err:
                         print(f"Error getting NextVar for index {index}, vehicle {vehicle_id}: {next_var_err}")
                         route_nodes_vehicle = [] # Discard potentially incomplete route on error
                         break # Exit inner loop

                # Only add non-empty routes (visited at least one customer)
                if route_nodes_vehicle:
                    routes_list.append(route_nodes_vehicle)

            # Format the flat route list (customer indices, 0-separated)
            for route in routes_list:
                 if route:
                     flat_route_final.extend(route)
                     flat_route_final.append(0) # Depot separator
            if flat_route_final and flat_route_final[-1] == 0: # Remove trailing separator
                 flat_route_final.pop()

            # Log success (Minimal logging here, more detailed if needed)
            # print(f"Instance {name} solved. Cost: {cost:.4f}")


        else: # Handle Failure or No Solution Found
            print(">> OR-Tools failed to find a feasible solution for instance {} - Status: {}".format(name, routing.status()))
            cost = float('inf')
            flat_route_final = []

        # Save result tuple (cost, flat_route) regardless of success/failure
        save_dataset((cost, flat_route_final), output_filename, disable_print=True)
        # Write basic log regardless of success/failure
        with open(log_filename, 'w') as log_f:
             log_f.write(f"Instance: {name}\n")
             log_f.write(f"Problem: {problem}\n")
             log_f.write(f"Status: {routing.status()}\n")
             log_f.write(f"Cost: {cost}\n")
             log_f.write(f"Route: {flat_route_final}\n")
             log_f.write(f"Duration: {duration:.4f}s\n")


        return cost, flat_route_final, duration # Return consistent tuple

    except Exception as e:
        print(f"Error solving instance {name} ({problem}): {e}")
        traceback.print_exc()
        # Ensure output file indicates failure if error occurs before saving
        try:
             save_dataset((float('inf'), []), output_filename, disable_print=True)
        except: pass # Ignore errors during error handling
        return float('inf'), [], 0 # Indicate failure with inf cost, empty route, 0 duration


# Define run_func at the top level
def run_func(args, problem_type, timelimit, seed):
    """Worker function for multiprocessing. Parses instance data and calls the solver."""
    directory, name, *instance_data = args

    depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = None, None, None, None, None, None, None, None

    # Parse instance_data based on the problem_type passed via partial
    try:
        if problem_type in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
            if len(instance_data) >= 4:
                depot, loc, demand, capacity, *_ = instance_data
            else: raise ValueError("Insufficient data for basic VRP type")
        elif problem_type in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
            if len(instance_data) >= 7:
                depot, loc, demand, capacity, service_time, tw_start, tw_end, *_ = instance_data
            else: raise ValueError("Insufficient data for VRPTW type")
        elif problem_type in ["VRPL", "VRPBL", "OVRPL", "OVRPBL"]:
            if len(instance_data) >= 5:
                depot, loc, demand, capacity, route_limit, *_ = instance_data
            else: raise ValueError("Insufficient data for VRPL type")
        elif problem_type in ["VRPLTW", "VRPBLTW", "OVRPLTW", "OVRPBLTW"]:
            if len(instance_data) >= 8:
                depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end, *_ = instance_data
            else: raise ValueError("Insufficient data for VRPLTW type")
        else:
            raise NotImplementedError(f"Problem type '{problem_type}' parsing not implemented in run_func")
    except ValueError as e:
         print(f"Error parsing instance data for worker {name} (problem: {problem_type}): {e}")
         print(f"Received instance data length: {len(instance_data)}")
         return None

    # Ensure depot is in the correct format [x, y]
    if isinstance(depot, list) and len(depot) == 1 and isinstance(depot[0], list):
        depot = depot[0]
    elif not isinstance(depot, list) or len(depot) != 2:
         print(f"Error: Invalid depot format for worker {name}: {depot}")
         return None

    grid_size = 1

    # Call the solver, passing the explicit parameters
    return solve_or_tools_log(
        directory, name,
        depot=depot, loc=loc, demand=demand, capacity=capacity, route_limit=route_limit,
        service_time=service_time, tw_start=tw_start, tw_end=tw_end,
        timelimit=timelimit, grid_size=grid_size, seed=seed, problem=problem_type
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OR-Tools baseline")
    parser.add_argument('--problem', type=str, default="CVRP", choices=["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                        "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                        "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument("--datasets", nargs='+', required=True, help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=None, help="Number of instances to process (defaults to all)")
    parser.add_argument('-timelimit', type=int, default=10, help="timelimit (seconds) for OR-Tools")
    parser.add_argument('-seed', type=int, default=1234, help="random seed")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")

    opts = parser.parse_args()
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        if not os.path.isfile(check_extension(dataset_path)):
             print(f"Error: Dataset file not found at {dataset_path}")
             continue

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        results_dir = os.path.join(opts.results_dir, f"{opts.problem}_or_tools")
        os.makedirs(results_dir, exist_ok=True)

        if opts.o is None:
             target_dir, filename = os.path.split(dataset_path)
             # Place results in the same directory as the input dataset
             out_file = os.path.join(target_dir, "or_tools_{}s_{}".format(opts.timelimit, filename))
        else:
            # Place results in the specified results_dir
            out_file = os.path.join(results_dir, opts.o)

        # Directory for logs for this specific run
        target_run_dir = os.path.join(results_dir, "{}_tl{}s_run".format(dataset_basename, opts.timelimit))

        print(f">> Input dataset: {dataset_path}")
        print(f">> Output results file: {out_file}")
        print(f">> Log/Tour directory: {target_run_dir}")

        if not opts.f and os.path.isfile(out_file):
             print(f"Error: Output file {out_file} already exists! Try running with -f option to overwrite.")
             continue

        start_t = time.time()
        use_multiprocessing = True
        os.makedirs(target_run_dir, exist_ok=True)

        dataset = load_dataset(dataset_path)
        if not dataset:
            print(f"Error: Failed to load or empty dataset from {dataset_path}")
            continue

        num_instances_to_process = len(dataset) if opts.n is None else opts.n
        if opts.offset >= len(dataset):
             print(f"Warning: Offset {opts.offset} is >= dataset size {len(dataset)}. No instances to process.")
             continue
        # Adjust n if offset+n exceeds dataset size
        num_instances_to_process = min(num_instances_to_process, len(dataset) - opts.offset)
        if num_instances_to_process <= 0:
            print("Warning: No instances to process after applying offset and n.")
            continue

        # Prepare arguments for the pool, including explicit opts parameters
        # Use functools.partial to pass fixed arguments to run_func
        pool_func = partial(run_func, problem_type=opts.problem, timelimit=opts.timelimit, seed=opts.seed)

        # Run using the pool
        results, parallelism = run_all_in_pool(
            pool_func, # Pass the partial function
            target_run_dir,
            dataset, # Pass the full dataset, run_all_in_pool handles slicing
            opts,    # Pass opts for run_all_in_pool (offset, n)
            use_multiprocessing=use_multiprocessing,
            disable_tqdm=False # Enable progress bar
        )

        # Filter out None results more carefully
        valid_results = [res for res in results if res is not None and res[0] is not None and res[1] is not None]
        num_attempted = len(results) # Number of tasks submitted to pool
        num_solved = len(valid_results)
        num_failed = num_attempted - num_solved

        print(f">> Attempted {num_attempted} instances.")
        print(f">> Successfully solved {num_solved} instances.")
        if num_failed > 0:
             print(f"Warning: {num_failed} instances failed to solve or return valid results.")


        if not valid_results:
            print("Error: No instances were successfully solved.")
            continue

        costs, tours, durations = zip(*valid_results)
        print(">> Stats based on {} successfully solved instances:".format(num_solved))
        valid_costs = [c for c in costs if c != float('inf')] # Exclude inf costs from stats
        valid_durations = list(durations)

        if valid_costs:
            print("Average cost (solved & feasible): {} +- {}".format(np.mean(valid_costs), 2 * np.std(valid_costs) / np.sqrt(len(valid_costs)) if len(valid_costs) > 0 else 0))
        else:
            print("No feasible solutions with valid costs found.")

        if valid_durations:
            print("Average serial duration (solved): {} +- {}".format(np.mean(valid_durations), 2 * np.std(valid_durations) / np.sqrt(len(valid_durations)) if len(valid_durations) > 0 else 0))
            if parallelism > 0:
                 print("Average parallel duration (solved): {}".format(np.mean(valid_durations) / parallelism))
                 print("Calculated total duration (estimated parallel): {}".format(timedelta(seconds=int(np.sum(valid_durations) / parallelism))))
            else:
                 print("Parallelism was 0 or 1, parallel duration is same as serial.")
        else:
             print("No valid durations found.")

        # Save final results including failures
        final_results_to_save = []
        result_idx = 0
        # Need to iterate based on the slice that run_all_in_pool processed
        start_idx = opts.offset
        end_idx = start_idx + len(results) # end index of processed slice
        # Ensure we handle cases where dataset slice was smaller than len(results) if pool failed early
        if end_idx > start_idx + num_attempted: end_idx = start_idx + num_attempted

        for i in range(start_idx, end_idx):
            # Get the result corresponding to this original index (relative to results list)
            current_result_idx = i - start_idx
            if current_result_idx < len(results):
                 current_result = results[current_result_idx]
                 if current_result is not None and current_result[0] is not None and current_result[1] is not None:
                     cost, tour, _ = current_result
                     final_results_to_save.append((cost, tour))
                 else: # Handle case where run_func returned None or failed tuple
                     final_results_to_save.append((float('inf'), []))
            else: # Should not happen if loop range is correct, but safeguard
                 print(f"Warning: Result index {current_result_idx} out of bounds for results list (len {len(results)}). Saving failure.")
                 final_results_to_save.append((float('inf'), []))

        # If opts.n was smaller than dataset size, add failures for remaining instances
        # This part might be incorrect if run_all_in_pool already handles slicing correctly.
        # Let's assume run_all_in_pool processed exactly the number of instances intended.
        # The loop above should cover all results returned by the pool.

        save_dataset(final_results_to_save, out_file)

        # Optional: Cleanup
        # import shutil
        # try:
        #     if num_failed == 0: # Only remove if all succeeded?
        #          shutil.rmtree(target_run_dir)
        #          print(f"Cleaned up intermediate directory: {target_run_dir}")
        #     else:
        #          print(f"Kept intermediate directory due to failures: {target_run_dir}")
        # except OSError as e:
        #     print(f"Error removing intermediate directory {target_run_dir}: {e}")
