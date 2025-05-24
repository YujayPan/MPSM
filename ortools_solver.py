import numpy as np
import math
import time # Import time module
from functools import partial
from six.moves import xrange
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import traceback
import logging # Use logging module

# Get a logger instance
logger = logging.getLogger(__name__)

# Constants
SCALE = 100000  # Scale for OR-Tools integer conversion
SPEED = 1.0     # Assumed speed for time calculations

# 所有VRP变体
VRP_VARIANTS = [
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
]

# 每种变体的问题数据包含的内容 - 用于解析输入元组
# 注意：顺序很重要！
VRP_DATA_FORMAT = {
    "CVRP": ["depot_xy", "node_xy", "demand", "capacity"],
    "OVRP": ["depot_xy", "node_xy", "demand", "capacity"],
    "VRPB": ["depot_xy", "node_xy", "demand", "capacity"], # demand distinguishes linehaul/backhaul
    "OVRPB": ["depot_xy", "node_xy", "demand", "capacity"],
    "VRPTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "OVRPL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "VRPBL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "OVRPBL": ["depot_xy", "node_xy", "demand", "capacity", "route_limit"],
    "VRPBTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPBTW": ["depot_xy", "node_xy", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "VRPBLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPBLTW": ["depot_xy", "node_xy", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"]
}

def parse_vrp_instance_tuple(instance_tuple, problem_type):
    """Parses the raw VRP instance tuple based on VRP_DATA_FORMAT."""
    if problem_type not in VRP_DATA_FORMAT:
        raise ValueError(f"Unknown problem type: {problem_type}")

    format_keys = VRP_DATA_FORMAT[problem_type]
    if len(instance_tuple) != len(format_keys):
        raise ValueError(f"Instance tuple length ({len(instance_tuple)}) does not match expected format length ({len(format_keys)}) for {problem_type}")

    instance_dict = {}
    for i, key in enumerate(format_keys):
        instance_dict[key] = instance_tuple[i]

    # Basic validation
    if not instance_dict.get("depot_xy") or not isinstance(instance_dict["depot_xy"], list) or not instance_dict["depot_xy"][0]:
         raise ValueError("Invalid 'depot_xy' format.")
    if not instance_dict.get("node_xy") or not isinstance(instance_dict["node_xy"], list):
         raise ValueError("Invalid 'node_xy' format.")
    if not instance_dict.get("demand") or not isinstance(instance_dict["demand"], list):
         raise ValueError("Invalid 'demand' format.")
    if len(instance_dict["node_xy"]) != len(instance_dict["demand"]):
        raise ValueError("Length mismatch between 'node_xy' and 'demand'.")

    # Add num_nodes for convenience
    instance_dict["num_nodes"] = len(instance_dict["node_xy"])

    # Add default values if optional fields are missing (should not happen if format matches)
    if 'service_time' not in instance_dict and 'TW' in problem_type:
        instance_dict['service_time'] = [0.0] * instance_dict["num_nodes"]
    if 'tw_start' not in instance_dict and 'TW' in problem_type:
        instance_dict['tw_start'] = [0.0] * instance_dict["num_nodes"]
    if 'tw_end' not in instance_dict and 'TW' in problem_type:
        # A large default end time if missing
        instance_dict['tw_end'] = [float('inf')] * instance_dict["num_nodes"]
    if 'route_limit' not in instance_dict and 'L' in problem_type:
        instance_dict['route_limit'] = float('inf') # Default to no limit if missing

    return instance_dict

def ortools_solve_vrp(instance_tuple, problem_type, timelimit,
                        stagnation_duration=10, min_stagnation_improvement_pct=0.5):
    """
    用 ortools 方法解决单个 VRP 问题。
    接收原始元组格式的 instance。

    Args:
        instance_tuple (tuple): 符合 VRP_DATA_FORMAT 定义的原始 VRP 实例元组。
        problem_type (str): 问题类型，支持 VRP_VARIANTS 列出的所有变体。
        timelimit (int): 时间限制（秒）。

    Returns:
        tuple: (cost, flat_route)
               cost (float): 找到的解的总成本(距离)，无解则为 float('inf')。
               flat_route (list): 包含所有客户节点索引（1-based）的扁平化列表，
                                  不同车辆的路径用 0 分隔。无解则为 []。
    """
    try:
        # 1. 解析原始 Instance 元组
        instance = parse_vrp_instance_tuple(instance_tuple, problem_type)

        # 2. 确定问题特性
        is_open = problem_type.startswith('O')
        has_backhaul = 'B' in problem_type
        has_time_windows = 'TW' in problem_type
        has_length_limit = 'L' in problem_type

        # 3. 准备 OR-Tools 数据模型 (Data Dictionary)
        data = {}
        to_int = lambda x: int(x * SCALE + 0.5) if isinstance(x, (int, float)) and not math.isinf(x) else int(99999 * SCALE) # Handle inf

        # Locations and Basic Info
        depot_coord = instance['depot_xy'][0] # Assuming depot_xy is [[x, y]]
        node_coords = instance['node_xy']
        locations = [depot_coord] + node_coords
        num_locations = len(locations) # Includes depot
        data['num_locations'] = num_locations
        data['locations'] = [(to_int(x), to_int(y)) for (x, y) in locations] # Scaled locations
        data['real_locations'] = locations # Keep original floats if needed later
        data['depot'] = 0
        # Set num_vehicles high, let solver minimize
        data['num_vehicles'] = max(1, num_locations - 1)

        # Demands and Capacity
        demands = instance['demand']
        data['demands'] = [0] + [int(d) for d in demands] # Use integer demands
        # Handle potentially infinite capacity from TSP subproblem
        raw_capacity = instance['capacity']
        if isinstance(raw_capacity, float) and math.isinf(raw_capacity):
            # Use a very large integer to represent infinite capacity for OR-Tools
            data['vehicle_capacity'] = 999999999 # Or another suitably large number
        else:
            data['vehicle_capacity'] = int(raw_capacity)

        # Dummy Depot for Open Routes
        data['dummy_depot'] = None
        num_manager_nodes = num_locations
        if is_open:
            data['dummy_depot'] = num_locations # Index for dummy depot
            num_manager_nodes += 1 # Add dummy node to manager

        # Time Windows
        if has_time_windows:
            service_times = instance['service_time']
            tw_starts = instance['tw_start']
            tw_ends = instance['tw_end']
            # Assume depot TW is [0, large_value]
            depot_tw_start = 0
            depot_tw_end = max(tw_ends) * 2 if any(not math.isinf(tw) for tw in tw_ends) else 3 # Heuristic horizon
            depot_tw_end = to_int(depot_tw_end)
            data['time_windows'] = [(depot_tw_start, depot_tw_end)] + \
                                   [(to_int(s), to_int(e)) for s, e in zip(tw_starts, tw_ends)]
            data['service_times'] = [0] + [to_int(st) for st in service_times] # Scaled service times
            if len(data['time_windows']) != num_locations or len(data['service_times']) != num_locations:
                raise ValueError("TW/Service Time list length mismatch with number of locations")

        # Distance Limit
        if has_length_limit:
            data['distance_limit'] = to_int(instance['route_limit'])

        # Precompute Distance Matrix (Scaled)
        _distances = {}
        for i in range(num_locations):
            _distances[i] = {}
            for j in range(num_locations):
                if i == j: _distances[i][j] = 0
                else:
                    loc1 = data['locations'][i]
                    loc2 = data['locations'][j]
                    # Integer Euclidean distance on scaled coordinates
                    _distances[i][j] = int(math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2))
        data['distance_matrix'] = _distances

        # Precompute Time Matrix if TW (Scaled)
        if has_time_windows:
            _times = {}
            for i in range(num_locations):
                _times[i] = {}
                for j in range(num_locations):
                    if i == j: _times[i][j] = 0
                    else:
                        # Travel time = scaled distance (assuming speed=1)
                        # Service time is applied AT node i before departing TO node j
                        travel_time = data['distance_matrix'][i][j]
                        service_time_at_i = data['service_times'][i]
                        _times[i][j] = service_time_at_i + travel_time
            data['time_matrix'] = _times

        # --- 4. Create Routing Index Manager ---
        depot_idx = data['depot']
        num_vehicles = data['num_vehicles']
        if is_open:
            dummy_depot_idx = data['dummy_depot']
            starts = [depot_idx] * num_vehicles
            ends = [dummy_depot_idx] * num_vehicles
            manager = pywrapcp.RoutingIndexManager(num_manager_nodes, num_vehicles, starts, ends)
        else:
            starts = [depot_idx] * num_vehicles
            ends = [depot_idx] * num_vehicles
            manager = pywrapcp.RoutingIndexManager(num_manager_nodes, num_vehicles, starts, ends)

        # --- 5. Create Routing Model ---
        routing = pywrapcp.RoutingModel(manager)

        # --- 6. Define Callbacks and Constraints ---

        # Distance Callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Handle dummy depot for open routes
            if is_open and to_node == data['dummy_depot']: return 0
            # Check bounds before accessing matrix
            if from_node >= num_locations or to_node >= num_locations: return 99999999 # Should not happen if manager is correct
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Demand Callback & Capacity Constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            # Handle dummy depot
            if is_open and from_node == data['dummy_depot']: return 0
            if from_node >= num_locations: return 0 # Should not happen
            return data['demands'][from_node] # Return integer demand

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        # Correctly set start_cumul_to_zero for backhaul
        start_cap_zero = not has_backhaul
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            # Ensure vehicle capacity is a list for AddDimensionWithVehicleCapacity
            [data['vehicle_capacity']] * num_vehicles, # capacity for each vehicle
            start_cap_zero,
            'Capacity')

        # Time Window Constraint
        if has_time_windows:
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                 # Handle dummy depot for open routes
                if is_open and to_node == data['dummy_depot']: return 0
                 # Check bounds before accessing matrix
                if from_node >= num_locations or to_node >= num_locations: return 99999999
                return data['time_matrix'][from_node][to_node]

            time_callback_index = routing.RegisterTransitCallback(time_callback)
            horizon = data['time_windows'][data['depot']][1] # Use depot end time as horizon
            routing.AddDimension(
                time_callback_index,
                horizon, # Slack (allowed waiting time)
                horizon, # Max time per vehicle
                False,   # Don't force start to zero, TWs handle this
                'Time')
            time_dimension = routing.GetDimensionOrDie('Time')
            for loc_idx in range(num_locations): # Apply to real locations + depot
                index = manager.NodeToIndex(loc_idx)
                tw = data['time_windows'][loc_idx]
                time_dimension.CumulVar(index).SetRange(tw[0], tw[1])
            # Set start/end time constraints for vehicles
            for v_id in range(num_vehicles):
                 start_index = routing.Start(v_id)
                 time_dimension.CumulVar(start_index).SetRange(data['time_windows'][depot_idx][0], data['time_windows'][depot_idx][1])
                 if is_open: # Allow ending anytime if open route
                      end_index = routing.End(v_id)
                      time_dimension.CumulVar(end_index).SetRange(0, horizon)


        # Distance Limit Constraint
        if has_length_limit:
            routing.AddDimension(
                transit_callback_index, # Use distance callback
                0, # No slack
                data['distance_limit'], # Max distance
                True, # Force start cumul to zero
                'Distance'
            )

        # Backhaul Constraint (Basic Implementation)
        if has_backhaul:
            # Identify linehaul (demand > 0) and backhaul (demand < 0) nodes
            # Note: Assumes demand sign reliably distinguishes them. Depot demand is 0.
            linehaul_indices = [manager.NodeToIndex(i) for i, d in enumerate(data['demands']) if d > 0]
            backhaul_indices = [manager.NodeToIndex(i) for i, d in enumerate(data['demands']) if d < 0]

            # Ensure Time dimension exists if we are using it for precedence
            if has_time_windows:
                time_dimension = routing.GetDimensionOrDie('Time')
                # Add order constraint: All linehauls must be visited before any backhauls on the same route
                # --- Comment out the explicit precedence constraint --- START
                # for l_idx in linehaul_indices:
                #     for b_idx in backhaul_indices:
                #         # Use dimension constraints for precedence
                #         routing.solver().Add(time_dimension.CumulVar(l_idx) <= time_dimension.CumulVar(b_idx))
                # --- Comment out the explicit precedence constraint --- END
                # print("Info: Relying on capacity dimension (start_cumul_to_zero=False) for Backhaul handling.")
            # else:
            #     # If no Time dimension (e.g., VRPB, OVRPB without TW), precedence is harder.
            #     # A simpler approach might be needed, or using the Distance dimension if L exists.
            #     # For now, we'll skip the explicit precedence constraint if no TW dimension is available.
            #     print(f"Warning: Cannot add explicit Backhaul precedence for {problem_type} without Time Windows.")


        # --- 7. Set Search Parameters ---
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = timelimit
        search_parameters.log_search = False # Disable OR-Tools verbose logging

        # --- 7.5 Add Stagnation Monitor --- 
        stagnation_monitor = StagnationMonitor(routing.solver(), routing, 
                                               stagnation_duration_seconds=stagnation_duration, 
                                               min_improvement_percentage=min_stagnation_improvement_pct)
        routing.AddSearchMonitor(stagnation_monitor)

        # --- 8. Solve ---
        assignment = routing.SolveWithParameters(search_parameters)

        # --- 9. Process Solution ---
        if assignment:
            cost = assignment.ObjectiveValue() / SCALE # Unscale the cost
            routes = []
            flat_route_unfiltered = [] # Intermediate list including depots

            for vehicle_id in range(num_vehicles):
                index = routing.Start(vehicle_id)
                route_for_vehicle = []
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    # Skip adding dummy depot to the route list
                    if not is_open or node_index != data['dummy_depot']:
                        route_for_vehicle.append(node_index)
                    index = assignment.Value(routing.NextVar(index))

                # Append the end node only if it's not the dummy depot
                end_node_index = manager.IndexToNode(routing.End(vehicle_id))
                if not is_open or end_node_index != data['dummy_depot']:
                     # Only add end node if it's the real depot (closed route)
                     if not is_open:
                         route_for_vehicle.append(end_node_index)

                # Add the route if it contains more than just the start node (or start/end depot)
                if len(route_for_vehicle) > 1:
                    routes.append(route_for_vehicle)
                    flat_route_unfiltered.extend(route_for_vehicle) # Add raw sequence

            # Convert to the required flat format (customer indices, 0 separated)
            flat_route_final = []
            current_vehicle_route = []
            # Iterate through the raw sequence including depots
            for node_idx in flat_route_unfiltered:
                 if node_idx == depot_idx: # If depot encountered
                      # If current route has customers, add it and a separator
                      if current_vehicle_route:
                           flat_route_final.extend(current_vehicle_route)
                           flat_route_final.append(0) # Add depot separator
                           current_vehicle_route = [] # Reset for next vehicle
                      # Handle consecutive depots (start/end) - don't add multiple zeros
                      elif not flat_route_final or flat_route_final[-1] != 0:
                           # Add starting depot only if list is empty or last wasn't depot
                           pass # We format later, don't add starting depot here
                 else: # Customer node
                      current_vehicle_route.append(node_idx) # Add customer index (1-based)

            # Add the last vehicle's route if any customers were visited
            if current_vehicle_route:
                flat_route_final.extend(current_vehicle_route)

            # Clean up potential trailing zero if last action was returning to depot
            if flat_route_final and flat_route_final[-1] == 0:
                 flat_route_final.pop()

            # Recalculate cost based on original coordinates for verification (optional but good)
            # verified_cost = calculate_total_distance(routes, instance['depot_xy'][0], instance['node_xy'], is_open)
            # print(f"Debug: OR-Tools Cost={cost:.4f}, Verified Cost={verified_cost:.4f}")

            return (cost, flat_route_final) # Return unscaled cost and flat route list
        else:
            print(f"OR-Tools found no solution within {timelimit}s for {problem_type}. Status: {routing.status()}")
            return (float('inf'), [])

    except Exception as e:
        # traceback.print_exc() # Avoid direct print to stdout, use logger
        logger.error(f"Error during OR-Tools solving for {problem_type}: {e}", exc_info=True)
        return (float('inf'), [])

# --- Custom Search Monitor for Stagnation --- 
class StagnationMonitor(pywrapcp.SearchMonitor):
    """Monitor to stop search if the objective hasn't improved sufficiently for a certain time."""
    def __init__(self, solver, routing, stagnation_duration_seconds, min_improvement_percentage):
        pywrapcp.SearchMonitor.__init__(self, solver)
        self._routing = routing
        self._stagnation_duration = stagnation_duration_seconds
        self._min_improvement_percentage = min_improvement_percentage # e.g., 0.5 for 0.5%
        self._cost_at_last_significant_update = float('inf')
        self._time_of_last_significant_update = 0.0
        self._initial_solution_found = False # Track if any solution has been found

    def EnterSearch(self):
        self._cost_at_last_significant_update = float('inf') # Reset cost
        self._time_of_last_significant_update = time.time() # Start timer when search begins
        self._initial_solution_found = False
        # logger.debug(f"StagnationMonitor: EnterSearch. Duration: {self._stagnation_duration}s, MinImprovement: {self._min_improvement_percentage}%")

    def PeriodicCheck(self):
        # Check if time limit exceeded first (safety, though OR-Tools has its own)
        # if time.time() - self.solver().wall_time() > self._overall_time_limit_seconds: 
        #     logger.info("StagnationMonitor: Overall time limit reached (checked in monitor).")
        #     self.solver().FinishCurrentSearch()
        #     return

        if self._initial_solution_found: # Only check stagnation if at least one solution was found
            current_time = time.time()
            time_since_last_significant_update = current_time - self._time_of_last_significant_update
            
            if time_since_last_significant_update > self._stagnation_duration:
                logger.info(
                    f"StagnationMonitor: Terminating search. No significant improvement (>{self._min_improvement_percentage}%) "
                    f"for {time_since_last_significant_update:.2f}s (limit: {self._stagnation_duration}s). "
                    f"Last significant cost: {self._cost_at_last_significant_update if self._cost_at_last_significant_update != float('inf') else 'None'}."
                )
                self.solver().FinishCurrentSearch() # More graceful termination
                # self.solver().Fail() # Alternative, more abrupt
        return True # Must return True to continue search if not failing

    def AcceptSolution(self):
        # This method is called when a *better* solution is found and accepted.
        # Update based on whether this improvement is significant.
        newly_accepted_cost_obj = self._routing.CostVar().Min() # Get the current best objective value from the routing model

        if not self._initial_solution_found:
            self._initial_solution_found = True
            # First solution is always a significant improvement over infinity
            logger.debug(f"StagnationMonitor: First solution found. Cost: {newly_accepted_cost_obj}")
            self._cost_at_last_significant_update = newly_accepted_cost_obj
            self._time_of_last_significant_update = time.time()
            return True # Accept the solution

        # If a solution was already found, check for significant improvement
        if self._cost_at_last_significant_update == float('inf'): # Should not happen if _initial_solution_found is True
            improvement_percentage = float('inf') # Treat as infinite improvement
        elif newly_accepted_cost_obj < self._cost_at_last_significant_update: # Ensure it's an improvement
            # Calculate percentage improvement: (old - new) / old * 100
            # Ensure old cost is not zero to avoid division by zero
            if abs(self._cost_at_last_significant_update) > 1e-9: # Avoid division by zero or near-zero
                improvement_percentage = ((self._cost_at_last_significant_update - newly_accepted_cost_obj) /
                                          abs(self._cost_at_last_significant_update)) * 100
            else: # If old cost was (near) zero, any reduction is significant
                improvement_percentage = float('inf') if newly_accepted_cost_obj < self._cost_at_last_significant_update else 0
        else: # Cost did not improve or worsened (should not happen if it's an accepted better solution)
            improvement_percentage = 0

        if improvement_percentage >= self._min_improvement_percentage:
            logger.debug(
                f"StagnationMonitor: Significant improvement detected ({improvement_percentage:.4f}% >= {self._min_improvement_percentage}%). "
                f"New best cost: {newly_accepted_cost_obj}. Old significant cost: {self._cost_at_last_significant_update}. Timer reset."
            )
            self._cost_at_last_significant_update = newly_accepted_cost_obj
            self._time_of_last_significant_update = time.time()
        # else:
            # logger.debug(
            #     f"StagnationMonitor: Insignificant improvement ({improvement_percentage:.4f}% < {self._min_improvement_percentage}%). "
            #     f"New best cost: {newly_accepted_cost_obj}. Timer NOT reset."
            # )
        return True # Always accept the solution if OR-Tools called this method


# Helper to calculate distance for verification (optional)
def calculate_total_distance(routes, depot_coord, node_coords, is_open):
    total_dist = 0
    coords = [depot_coord] + node_coords # Combine depot + customers
    for route in routes:
        current_pos = depot_coord
        for node_idx in route: # route should contain 1-based indices
            if node_idx > 0 and node_idx <= len(node_coords):
                next_pos = node_coords[node_idx-1] # Get 0-based index for node_coords
                dist = math.sqrt((current_pos[0] - next_pos[0])**2 + (current_pos[1] - next_pos[1])**2)
                total_dist += dist
                current_pos = next_pos
            else:
                 print(f"Warning: Invalid node index {node_idx} in route for verification.")
        if not is_open: # Add return to depot distance for closed routes
             dist = math.sqrt((current_pos[0] - depot_coord[0])**2 + (current_pos[1] - depot_coord[1])**2)
             total_dist += dist
    return total_dist


# --- Main Execution Block ---
def main():
    """
    主函数，处理实例并求解VRP问题
    """
    # --- Example 1: OVRPBLTW ---
    instance_ovrpbltw = (
        [[0.38717731833457947, 0.7680646777153015]], # depot_xy
        [[0.8603761196136475, 0.3659379482269287], [0.3716965615749359, 0.016600994393229485], [0.03751150146126747, 0.9240059852600098], [0.46383723616600037, 0.7510457038879395], [0.00245261425152421, 0.23914557695388794], [0.8249342441558838, 0.2129097282886505], [0.5955009460449219, 0.5178664922714233], [0.1641962230205536, 0.920869767665863], [0.1945953369140625, 0.09998763352632523], [0.6403608322143555, 0.6938608288764954], [0.8090419769287109, 0.9630672931671143], [0.32533732056617737, 0.3506787121295929], [0.14271079003810883, 0.5939927697181702], [0.7615496516227722, 0.47877877950668335], [0.5535492300987244, 0.1691935807466507], [0.6367502808570862, 0.7412551045417786], [0.02052283100783825, 0.5240311026573181], [0.4157567024230957, 0.5333794355392456], [0.33593007922172546, 0.7489286065101624], [0.19028769433498383, 0.8654850721359253], [0.7671440243721008, 0.17276859283447266], [0.5872756838798523, 0.12061207741498947], [0.6903773546218872, 0.004159960895776749], [0.4096517264842987, 0.30592790246009827], [0.400923490524292, 0.6924741268157959], [0.45641452074050903, 0.3142610192298889], [0.3298740088939667, 0.7694856524467468], [0.7329878211021423, 0.41820383071899414], [0.6989092826843262, 0.2837562561035156], [0.7716724276542664, 0.2228616327047348], [0.5345510244369507, 0.2924153804779053], [0.04710880666971207, 0.18952974677085876], [0.6749705076217651, 0.017189249396324158], [0.15119607746601105, 0.9864438772201538], [0.04623565822839737, 0.15069952607154846], [0.039018452167510986, 0.5267100930213928], [0.7169221043586731, 0.6855891346931458], [0.13014140725135803, 0.6018301248550415], [0.8262268304824829, 0.9441471099853516], [0.16020193696022034, 0.2031029313802719], [0.8336463570594788, 0.9082401990890503], [0.565040111541748, 0.6368759274482727], [0.05043938383460045, 0.5499301552772522], [0.8434057831764221, 0.2598307728767395], [0.23548659682273865, 0.5724191069602966], [0.5514792799949646, 0.9673106074333191], [0.15499304234981537, 0.8294548988342285], [0.5438411831855774, 0.24332968890666962], [0.6862167119979858, 0.9935036897659302], [0.23863287270069122, 0.040021393448114395]], # node_xy (N=50)
        [-4.0, 5.0, 5.0, 3.0, -8.0, 6.0, 4.0, 6.0, 1.0, -3.0, 5.0, -3.0, 9.0, 3.0, -9.0, 8.0, 2.0, 7.0, 9.0, 8.0, 4.0, 5.0, 7.0, 6.0, 2.0, -2.0, 3.0, 7.0, -5.0, 2.0, 7.0, 6.0, 1.0, 6.0, 2.0, 3.0, 6.0, 4.0, -4.0, 9.0, 4.0, 5.0, -2.0, 4.0, 4.0, 2.0, 4.0, -2.0, 2.0, 3.0], # demands (negative are backhaul)
        40.0, # capacity
        3.0, # route_limit ('L')
        [0.2]*50, # service_time ('TW')
        [1.267, 0.948, 0.864, 2.080, 0.736, 0.994, 1.077, 0.445, 0.001, 1.643, 0.064, 0.0, 1.108, 0.517, 0.255, 0.867, 1.456, 1.727, 1.040, 1.613, 0.850, 0.489, 0.161, 1.691, 0.763, 1.118, 1.178, 1.122, 0.660, 0.090, 0.155, 1.130, 0.821, 0.0, 1.124, 0.211, 1.271, 0.740, 1.072, 0.598, 0.396, 1.199, 0.299, 0.756, 0.907, 0.376, 0.236, 2.011, 0.0, 0.469], # tw_start ('TW')
        [2.044, 1.454, 2.647, 2.888, 2.093, 1.498, 2.042, 1.594, 1.525, 2.166, 0.912, 1.270, 1.739, 1.061, 1.158, 2.127, 1.949, 3.0, 1.724, 2.329, 2.705, 2.290, 1.540, 2.709, 1.659, 1.527, 1.810, 1.454, 1.716, 1.756, 2.048, 1.555, 1.579, 1.117, 3.0, 1.379, 3.0, 1.073, 2.743, 0.872, 0.724, 2.186, 1.322, 1.160, 2.774, 1.985, 1.263, 2.488, 1.396, 1.661]  # tw_end ('TW')
    )
    print("\n--- Solving OVRPBLTW Example ---")
    problem_type_1 = "OVRPBLTW"
    timelimit_1 = 20 # seconds

    cost_1, route_1 = ortools_solve_vrp(instance_ovrpbltw, problem_type_1, timelimit_1)

    print(f"--- {problem_type_1} Results ---")
    if cost_1 != float('inf'):
        print(f"Total Distance: {cost_1:.4f}")
        print(f"Route: {route_1}")
        # Visualize routes
        routes_viz_1 = []
        current_route_1 = []
        for node in route_1:
            if node == 0:
                if current_route_1: routes_viz_1.append(current_route_1)
                current_route_1 = []
            else: current_route_1.append(node)
        if current_route_1: routes_viz_1.append(current_route_1)
        print(f"Found {len(routes_viz_1)} routes:")
        for i, r in enumerate(routes_viz_1): print(f"  Route {i+1}: 0 -> {' -> '.join(map(str, r))}") # Open route may not end at 0
    else:
        print("No solution found.")

    # --- Example 2: CVRP ---
    instance_cvrp = (
        [[0.5, 0.5]],  # Depot
        [[0.1, 0.2], [0.8, 0.9], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]], # Nodes (N=5)
        [10, 5, 8, 6, 9],  # Demands
        30.0  # Capacity
    )
    print("\n--- Solving CVRP Example ---")
    problem_type_2 = "CVRP"
    timelimit_2 = 20 # seconds

    cost_2, route_2 = ortools_solve_vrp(instance_cvrp, problem_type_2, timelimit_2)

    print(f"--- {problem_type_2} Results ---")
    if cost_2 != float('inf'):
        print(f"Total Distance: {cost_2:.4f}")
        print(f"Route: {route_2}")
        # Visualize routes
        routes_viz_2 = []
        current_route_2 = []
        for node in route_2:
            if node == 0:
                if current_route_2: routes_viz_2.append(current_route_2)
                current_route_2 = []
            else: current_route_2.append(node)
        if current_route_2: routes_viz_2.append(current_route_2)
        print(f"Found {len(routes_viz_2)} routes:")
        for i, r in enumerate(routes_viz_2): print(f"  Route {i+1}: 0 -> {' -> '.join(map(str, r))} -> 0")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()