import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pickle
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from utils import load_dataset
import matplotlib.cm as cm
# --- Added Imports ---
import torch
from partitioner_solver_utils import (
    load_moe_model,
    partition_instance,
    pad_subproblem_batch,
    prepare_batch_tensor_data,
    get_env,
    solve_vrp_batch,
    merge_solved_instances,
    create_subproblem_instance, # For potential dummy data if needed later
    DEFAULT_MODEL_PARAMS as DEFAULT_PARTITIONER_SOLVER_MODEL_PARAMS
)
from collections import Counter # Import Counter for finding mode
import colorsys # Import colorsys for HSV/RGB conversions
from collections import defaultdict # Import defaultdict for grouping route indices
from matplotlib.lines import Line2D
# --- End Added Imports ---

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # Add fallback font
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Define Z-order constants for consistent layering
Z_GRID = 0
Z_ROUTE = 1
Z_PADDING_MARKER = 2
Z_CUSTOMER_NODE = 3
Z_DEPOT = 4
Z_ANNOTATION = 5

# 定义所有VRP变体
VRP_VARIANTS = [
    "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
    "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
    "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"
]

# 定义每种变体的数据格式
VRP_DATA_FORMAT = {
    "CVRP": ["depot", "loc", "demand", "capacity"],
    "OVRP": ["depot", "loc", "demand", "capacity"],
    "VRPB": ["depot", "loc", "demand", "capacity"],
    "OVRPB": ["depot", "loc", "demand", "capacity"],
    "VRPTW": ["depot", "loc", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPTW": ["depot", "loc", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPL": ["depot", "loc", "demand", "capacity", "route_limit"],
    "OVRPL": ["depot", "loc", "demand", "capacity", "route_limit"],
    "VRPBL": ["depot", "loc", "demand", "capacity", "route_limit"],
    "OVRPBL": ["depot", "loc", "demand", "capacity", "route_limit"],
    "VRPBTW": ["depot", "loc", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "OVRPBTW": ["depot", "loc", "demand", "capacity", "service_time", "tw_start", "tw_end"],
    "VRPLTW": ["depot", "loc", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPLTW": ["depot", "loc", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "VRPBLTW": ["depot", "loc", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"],
    "OVRPBLTW": ["depot", "loc", "demand", "capacity", "route_limit", "service_time", "tw_start", "tw_end"]
}


# --- Translation Dictionary --- 
TRANSLATIONS = {
    'en': {
        'instance_info_header': "=== {problem_type} Problem Instance Information ===",
        'depot_coords': "Depot Coordinates: {depot}",
        'num_customers': "Number of Customers: {count}",
        'vehicle_capacity': "Vehicle Capacity: {capacity}",
        'demand_range': "Demand Range: [{min_d}, {max_d}]",
        'service_time_range': "Service Time Range: [{min_s}, {max_s}]",
        'tw_start_range': "Time Window Start Range: [{min_ts}, {max_ts}]",
        'tw_end_range': "Time Window End Range: [{min_te}, {max_te}]",
        'route_limit': "Route Length Limit: {limit}",
        'solution_info_header': "=== Solution Information ===",
        'solution_score': "Solution Score: {obj}",
        'num_routes': "Number of Routes: {count}",
        'route_header': "Route {i}: {r}",
        'route_anno': "Route {i}, Demand:{demand}/{capacity}",
        'route_limit_anno': "Limit:{limit}",
        'x_coord': "X Coordinate",
        'y_coord': "Y Coordinate",
        'depot_label': "Depot",
        'customer_label': "Customers",
        'partitioned_viz_header': "=== Visualizing Partitioned {problem_type} Instance ===",
        'num_subproblems': "Number of subproblems: {count}",
        'error_original_instance': "Error: Original instance data is missing.",
        'warning_no_subproblems': "Warning: No subproblem tuples provided to visualize.",
        'error_parse_original': "Error parsing original instance data: {e}. Check instance format.",
        'subproblem_label': "Subproblem {i} ({count} nodes)",
        'warning_coord_not_found': "Warning: Coordinate {coord} from subproblem {i} not found in original instance locations.",
        'error_process_subproblem': "Error processing subproblem tuple {i}: {e}. Skipping.",
        'warning_unassigned_nodes': "Warning: {count} nodes were not assigned to any visualized subproblem: {indices}",
        'unassigned_label': "Unassigned",
        'partitioned_title': "Partitioned {problem_type} Instance (N={n}) - {k} Subproblems",
        'error_dataset_not_found': "Error: Dataset file not found at {path}",
        'error_instance_not_found': "Error: Instance {idx} not found in {path}",
        'attempting_partition': "\nAttempting to partition the instance...",
        'partition_success': "Partitioning function call successful.",
        'partition_fail': "Partitioning failed. Cannot visualize partitioned instance.",
        'raw_sequence_info': "Raw sequence generated before failure (first 50): {seq}",
        'solution_not_found': "Solution file not found at {path}, skipping solution visualization.",
        'error_solution_load': "Error loading or visualizing solution: {e}",
        'import_error': "Import Error: {e}. Make sure partitioner_solver_utils.py is accessible.",
        'unexpected_error': "An unexpected error occurred: {e}",
        'instance_info_title': "{ptype} Instance\nTotal Customers: {n}\nCapacity: {cap}",
        'instance_info_title_limit': "\nRoute Length Limit: {limit}",
        'solution_title': "Score: {obj}"
    },
    'zh': {
        'instance_info_header': "=== {problem_type} 问题实例信息 ===",
        'depot_coords': "仓库坐标: {depot}",
        'num_customers': "客户点数量: {count}",
        'vehicle_capacity': "车辆容量: {capacity}",
        'demand_range': "需求范围: [{min_d}, {max_d}]",
        'service_time_range': "服务时间范围: [{min_s}, {max_s}]",
        'tw_start_range': "时间窗开始范围: [{min_ts}, {max_ts}]",
        'tw_end_range': "时间窗结束范围: [{min_te}, {max_te}]",
        'route_limit': "路线长度限制: {limit}",
        'solution_info_header': "=== 解信息 ===",
        'solution_score': "解的Score值: {obj}",
        'num_routes': "路线数量: {count}",
        'route_header': "路线{i}: {r}",
        'route_anno': "路线{i}，需求:{demand}/{capacity}",
        'route_limit_anno': "限制:{limit}",
        'x_coord': "X坐标",
        'y_coord': "Y坐标",
        'depot_label': "仓库",
        'customer_label': "客户点",
        'partitioned_viz_header': "=== 分区 {problem_type} 实例可视化 ===",
        'num_subproblems': "子问题数量: {count}",
        'error_original_instance': "错误：原始实例数据丢失。",
        'warning_no_subproblems': "警告：未提供子问题元组用于可视化。",
        'error_parse_original': "解析原始实例数据时出错：{e}。请检查实例格式。",
        'subproblem_label': "子问题 {i} ({count} 个节点)",
        'warning_coord_not_found': "警告：来自子问题 {i} 的坐标 {coord} 在原始实例位置中未找到。",
        'error_process_subproblem': "处理子问题元组 {i} 时出错：{e}。跳过。",
        'warning_unassigned_nodes': "警告：{count} 个节点未分配到任何可视化子问题：{indices}",
        'unassigned_label': "未分配",
        'partitioned_title': "已分区 {problem_type} 实例 (N={n}) - {k} 个子问题",
        'error_dataset_not_found': "错误：数据集文件未在 {path} 找到",
        'error_instance_not_found': "错误：实例 {idx} 未在 {path} 中找到",
        'attempting_partition': "\n正在尝试对实例进行分区...",
        'partition_success': "分区函数调用成功。",
        'partition_fail': "分区失败。无法可视化已分区的实例。",
        'raw_sequence_info': "失败前生成的原始序列（前50个）：{seq}",
        'solution_not_found': "解决方案文件未在 {path} 找到，跳过解决方案可视化。",
        'error_solution_load': "加载或可视化解决方案时出错：{e}",
        'import_error': "导入错误：{e}。请确保 partitioner_solver_utils.py 可访问。",
        'unexpected_error': "发生意外错误：{e}",
        'instance_info_title': "{ptype} 问题实例\n总客户点: {n}\n车辆容量: {cap}",
        'instance_info_title_limit': "\n路线长度限制: {limit}",
        'solution_title': "Score: {obj}"
    }
}

def get_text(key, lang='en', **kwargs):
    """Helper function to get translated text."""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key).format(**kwargs)


def print_vrp_instance_info(instance, problem_type, lang='en'):
    """print detailed information of VRP problem instance"""
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    print(txt('instance_info_header', problem_type=problem_type))
    
    # unpack instance data
    if problem_type in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
        depot, loc, demand, capacity = instance
        print(txt('depot_coords', depot=depot))
        print(txt('num_customers', count=len(loc)))
        print(txt('vehicle_capacity', capacity=capacity))
        print(txt('demand_range', min_d=min(demand), max_d=max(demand)))
        
    elif problem_type in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
        depot, loc, demand, capacity, service_time, tw_start, tw_end = instance
        print(txt('depot_coords', depot=depot))
        print(txt('num_customers', count=len(loc)))
        print(txt('vehicle_capacity', capacity=capacity))
        print(txt('demand_range', min_d=min(demand), max_d=max(demand)))
        print(txt('service_time_range', min_s=min(service_time), max_s=max(service_time)))
        print(txt('tw_start_range', min_ts=min(tw_start), max_ts=max(tw_start)))
        print(txt('tw_end_range', min_te=min(tw_end), max_te=max(tw_end)))
        
    elif problem_type in ["VRPL", "OVRPL", "VRPBL", "OVRPBL"]:
        depot, loc, demand, capacity, route_limit = instance
        print(txt('depot_coords', depot=depot))
        print(txt('num_customers', count=len(loc)))
        print(txt('vehicle_capacity', capacity=capacity))
        print(txt('demand_range', min_d=min(demand), max_d=max(demand)))
        print(txt('route_limit', limit=route_limit))
        
    elif problem_type in ["VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
        depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = instance
        print(txt('depot_coords', depot=depot))
        print(txt('num_customers', count=len(loc)))
        print(txt('vehicle_capacity', capacity=capacity))
        print(txt('demand_range', min_d=min(demand), max_d=max(demand)))
        print(txt('route_limit', limit=route_limit))
        print(txt('service_time_range', min_s=min(service_time), max_s=max(service_time)))
        print(txt('tw_start_range', min_ts=min(tw_start), max_ts=max(tw_start)))
        print(txt('tw_end_range', min_te=min(tw_end), max_te=max(tw_end)))


def visualize_vrp_instance(instance, problem_type, lang='en', show_annotations=False, ax=None):
    """
    visualize VRP problem instance
    :param instance: VRP instance data
    :param problem_type: problem type
    :param lang: Language ('en' or 'zh')
    :param show_annotations: Boolean, whether to show text annotations on the plot. Defaults to False.
    :param ax: Matplotlib axis object for subplotting. If None, a new figure is created.
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    # unpack instance data
    if problem_type in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
        depot, loc, demand, capacity = instance
        service_time = None
        tw_start = None
        tw_end = None
        route_limit = None
    elif problem_type in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
        depot, loc, demand, capacity, service_time, tw_start, tw_end = instance
        route_limit = None
    elif problem_type in ["VRPL", "OVRPL", "VRPBL", "OVRPBL"]:
        depot, loc, demand, capacity, route_limit = instance
        service_time = None
        tw_start = None
        tw_end = None
    elif problem_type in ["VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
        depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = instance
    else:
        print(f"Unknown problem type: {problem_type}")
        return
    
    depot_coord = depot[0] if isinstance(depot[0], (list, tuple)) else depot
    
    # --- Figure and Axis Setup ---
    if ax is None:
        fig_created, current_ax = plt.subplots(figsize=(12, 12))
    else:
        fig_created = None # No new figure created by this function
        current_ax = ax
    
    # draw depot
    current_ax.scatter(depot_coord[0], depot_coord[1], c='red', s=50, marker='*', label=txt('depot_label'), zorder=Z_DEPOT) # Reduced s from 200 to 100
    
    # draw customer nodes
    current_ax.scatter([x for x, y in loc], [y for x, y in loc], c='blue', s=25, label=txt('customer_label'), zorder=Z_CUSTOMER_NODE) # Reduced s from 100 to 50
    
    # annotate customer nodes
    for i, (x, y) in enumerate(loc):
        if show_annotations:
            current_ax.text(
                x, y, 
                str(i+1),
                fontsize=15, 
                ha='center', 
                va='bottom', 
                color='black',
                zorder=Z_ANNOTATION
            )
    
    # set title
    if show_annotations:
        title = txt('instance_info_title', ptype=problem_type, n=len(loc), cap=capacity)
        if route_limit is not None:
            title += txt('instance_info_title_limit', limit=route_limit)
    else:
        title = f"VRP Instance"
    current_ax.set_title(title, fontsize=14) # Changed fontsize
    
    current_ax.set_xlabel(txt('x_coord'))
    current_ax.set_ylabel(txt('y_coord'))
    current_ax.grid(True, alpha=0.3, zorder=Z_GRID)
    current_ax.legend(markerscale=0.7)

    if fig_created is not None: # Only show if this function created the figure
        plt.show()


def visualize_solution(instance, solution, problem_type, lang='en', show_annotations=False, ax=None, custom_title=None):
    """
    visualize OR-Tools solution
    :param instance: dataset
    :param solution: solution
    :param problem_type: problem type
    :param lang: Language ('en' or 'zh')
    :param show_annotations: Boolean, whether to show text annotations on the plot. Defaults to False.
    :param ax: Matplotlib axis object for subplotting. If None, a new figure is created.
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    obj, route = solution
    
    routes_data = [] # Renamed from routes to avoid conflict
    current_route_segment = []
    for node in route:
        if node == 0:  
            if current_route_segment:  
                routes_data.append(current_route_segment)
                current_route_segment = []
        else:
            current_route_segment.append(node) 
    if current_route_segment:  
        routes_data.append(current_route_segment)
    
    print(txt('solution_info_header'))
    print(txt('solution_score', obj=f"{obj:.4f}"))
    print(txt('num_routes', count=len(routes_data)))
    if show_annotations:
        for i, r_seg in enumerate(routes_data):
            print(txt('route_header', i=i+1, r=r_seg))
    
    # unpack instance data (logic unchanged)
    if problem_type in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
        depot, loc, demand, capacity = instance
        service_time = None; tw_start = None; tw_end = None; route_limit = None
    elif problem_type in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
        depot, loc, demand, capacity, service_time, tw_start, tw_end = instance
        route_limit = None
    elif problem_type in ["VRPL", "OVRPL", "VRPBL", "OVRPBL"]:
        depot, loc, demand, capacity, route_limit = instance
        service_time = None; tw_start = None; tw_end = None
    elif problem_type in ["VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
        depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = instance
    else:
        print(f"Unknown problem type: {problem_type}")
        return

    depot_coord = depot[0] if isinstance(depot[0], (list, tuple)) else depot
    
    fig_created = False # Flag to track if this function created the figure
    if ax is None:
        fig, current_ax = plt.subplots(figsize=(12, 12))
        fig_created = True
    else:
        current_ax = ax
    
    current_ax.scatter(depot_coord[0], depot_coord[1], c='red', s=50, marker='*', label=txt('depot_label'), zorder=Z_DEPOT)
    current_ax.scatter([x for x, y in loc], [y for x, y in loc], c='blue', s=25, label=txt('customer_label'), zorder=Z_CUSTOMER_NODE, alpha=0.5) # Added alpha
    
    if show_annotations: # Ensure annotations are shown if requested
        for i, (x, y) in enumerate(loc):
            current_ax.text(x, y, str(i+1), fontsize=15, ha='center', va='bottom', color='black', zorder=Z_ANNOTATION)
    
    colors = plt.cm.hsv(np.linspace(0.05, 0.95, len(routes_data))) if routes_data else []
    for i, r_seg in enumerate(routes_data):
        route_points = [depot_coord] + [loc[node-1] for node in r_seg] + [depot_coord]
        route_x = [p[0] for p in route_points]
        route_y = [p[1] for p in route_points]
        current_ax.plot(route_x, route_y, c=colors[i], linewidth=2, alpha=0.7, zorder=Z_ROUTE)
        
        route_demand_val = sum(demand[node-1] for node in r_seg) # Renamed variable
        if len(r_seg) > 0:
            if show_annotations:
                mid_x = sum(route_x) / len(route_x)
                mid_y = sum(route_y) / len(route_y)
                info = txt('route_anno', i=i+1, demand=route_demand_val, capacity=capacity)
                if route_limit is not None:
                    info += f"\n" + txt('route_limit_anno', limit=route_limit)
                current_ax.text(mid_x, mid_y, info, fontsize=8, ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7), zorder=Z_ANNOTATION)
    
    # Set title based on whether it's a subplot or a standalone plot
    if custom_title is not None:
        # current_ax.set_title(custom_title, fontsize=10 if "\n" in custom_title else plt.rcParams['axes.titlesize']) # Adjust fontsize for multi-line
        current_ax.set_title(custom_title, fontsize=14, fontweight='bold') # Changed fontsize to 14
    else:
        plot_title = f"Final Solution" 
        if obj is not None and not (isinstance(obj, float) and np.isnan(obj)):
            plot_title += f" - Score: {obj:.2f}"
        current_ax.set_title(plot_title, fontweight='bold', fontsize=14) # Added fontsize=14

    current_ax.set_xlabel(txt('x_coord'))
    current_ax.set_ylabel(txt('y_coord'))
    current_ax.grid(True, alpha=0.3, zorder=Z_GRID)
    current_ax.legend(markerscale=0.7)

    if fig_created: # Only call plt.show() if this function created the figure
        plt.tight_layout() # Apply tight_layout before showing for standalone plots
        plt.show()


def visualize_colored_solution(instance, solution, problem_type, customer_node_colors, lang='en', show_annotations=False, ax=None, subproblem_legend_info=None):
    """
    Visualizes OR-Tools solution with custom colors for customer nodes.
    :param ax: Matplotlib axis object for subplotting. If None, a new figure is created.
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    obj, route_sequence = solution
    
    routes_data = [] # Renamed for clarity
    current_route_segment = []
    for node_idx in route_sequence:
        if node_idx == 0:
            if current_route_segment:
                routes_data.append(current_route_segment)
                current_route_segment = []
        else:
            current_route_segment.append(node_idx)
    if current_route_segment:
        routes_data.append(current_route_segment)
    
    print(txt('solution_info_header'))
    print(txt('solution_score', obj=f"{obj:.4f}"))
    print(txt('num_routes', count=len(routes_data)))
    if show_annotations:
        for i, r_seg in enumerate(routes_data):
            print(txt('route_header', i=i+1, r=r_seg))
    
    # Unpack instance data (logic unchanged)
    if problem_type in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
        depot, loc, demand, capacity = instance
        route_limit = None
    elif problem_type in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
        depot, loc, demand, capacity, _, _, _ = instance 
        route_limit = None
    elif problem_type in ["VRPL", "OVRPL", "VRPBL", "OVRPBL"]:
        depot, loc, demand, capacity, route_limit = instance
    elif problem_type in ["VRPLTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]:
        depot, loc, demand, capacity, route_limit, _, _, _ = instance 
    else:
        print(f"Unknown problem type: {problem_type}")
        return

    depot_coord = depot[0] if isinstance(depot[0], (list, tuple)) else depot
    
    if ax is None:
        fig_created, current_ax = plt.subplots(figsize=(12, 12))
    else:
        fig_created = None
        current_ax = ax
    
    current_ax.scatter(depot_coord[0], depot_coord[1], c='red', s=50, marker='*', label=txt('depot_label'), zorder=Z_DEPOT) # Reduced s from 200 to 100
    
    if len(loc) != len(customer_node_colors):
        print(f"Warning: Number of locations ({len(loc)}) does not match number of colors ({len(customer_node_colors)}). Using default color.")
        current_ax.scatter([x for x, y in loc], [y for x, y in loc], c='blue', s=25, label=txt('customer_label'), zorder=Z_CUSTOMER_NODE, alpha=0.5) # Added alpha
    else:
        current_ax.scatter([x for x, y in loc], [y for x, y in loc], c=customer_node_colors, s=25, alpha=0.5, zorder=Z_CUSTOMER_NODE) # Changed alpha from 0.9 to 0.7

    if show_annotations:
        for i, (x, y) in enumerate(loc):
            current_ax.text(x, y, str(i+1), fontsize=15, ha='center', va='bottom', color='black', zorder=Z_ANNOTATION)
    
    route_base_colors_with_indices = []
    for i, r_seg in enumerate(routes_data):
        base_color_for_this_route = (.5, .5, .5, 1.0)
        if r_seg:
            first_node_idx_0based = r_seg[0] - 1
            if 0 <= first_node_idx_0based < len(customer_node_colors):
                color_candidate = customer_node_colors[first_node_idx_0based]
                if isinstance(color_candidate, (list, np.ndarray)) and len(color_candidate) == 4:
                    base_color_for_this_route = tuple(color_candidate)
                elif isinstance(color_candidate, tuple) and len(color_candidate) == 4:
                    base_color_for_this_route = color_candidate 
                else:
                    logger.warning(f"Route {i} first node color is not a valid RGBA, using default. Color: {color_candidate}")
        route_base_colors_with_indices.append((base_color_for_this_route, i))

    final_route_colors = [None] * len(routes_data)
    grouped_by_base_color = defaultdict(list)
    for color_tuple, route_idx in route_base_colors_with_indices:
        grouped_by_base_color[color_tuple].append(route_idx)

    hue_shift_step = 0.045
    for base_color_rgba_tuple, route_indices in grouped_by_base_color.items():
        num_routes_for_this_base_color = len(route_indices)
        if not (isinstance(base_color_rgba_tuple, tuple) and (len(base_color_rgba_tuple) == 3 or len(base_color_rgba_tuple) == 4)):
            # logger.warning(f"Invalid base_color_tuple format: {base_color_rgba_tuple}. Skipping color variation for these routes: {route_indices}")
            # Use a simple print for now if logger is not configured in this scope
            print(f"Warning: Invalid base_color_tuple format: {base_color_rgba_tuple}. Using gray for routes: {route_indices}")
            for route_idx in route_indices:
                final_route_colors[route_idx] = 'gray' 
            continue
        if num_routes_for_this_base_color == 1:
            final_route_colors[route_indices[0]] = list(base_color_rgba_tuple)
        else:
            base_r, base_g, base_b = base_color_rgba_tuple[:3]
            original_alpha = base_color_rgba_tuple[3] if len(base_color_rgba_tuple) == 4 else 1.0
            try:
                h_orig, s_orig, v_orig = colorsys.rgb_to_hsv(base_r, base_g, base_b)
            except Exception as e:
                print(f"Error converting base color {base_color_rgba_tuple[:3]} to HSV: {e}. Using base color for routes {route_indices}.")
                for route_idx in route_indices:
                    final_route_colors[route_idx] = list(base_color_rgba_tuple)
                continue
            s_adjusted = s_orig
            v_adjusted = v_orig
            for i, route_idx in enumerate(route_indices):
                hue_offset = (i - (num_routes_for_this_base_color - 1) / 2.0) * hue_shift_step
                h_new = (h_orig + hue_offset) % 1.0
                r_new, g_new, b_new = colorsys.hsv_to_rgb(h_new, s_adjusted, v_adjusted)
                final_route_colors[route_idx] = [r_new, g_new, b_new, original_alpha]

    for i, r_seg in enumerate(routes_data):
        route_points = [depot_coord] + [loc[node_idx-1] for node_idx in r_seg] + [depot_coord]
        route_x = [p[0] for p in route_points]
        route_y = [p[1] for p in route_points]
        current_route_plot_color = final_route_colors[i] if final_route_colors[i] is not None else 'darkgrey'
        current_ax.plot(route_x, route_y, c=current_route_plot_color, linewidth=2, alpha=0.8, zorder=Z_ROUTE)
        
        route_demand_val = sum(demand[node_idx-1] for node_idx in r_seg) # Renamed
        if show_annotations and len(r_seg) > 0:
            mid_x = sum(route_x) / len(route_x)
            mid_y = sum(route_y) / len(route_y)
            info = txt('route_anno', i=i+1, demand=route_demand_val, capacity=capacity)
            if route_limit is not None:
                info += f"\n" + txt('route_limit_anno', limit=route_limit)
            current_ax.text(mid_x, mid_y, info, fontsize=8, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7), zorder=Z_ANNOTATION)
    
    title = f"{problem_type} Solution"
    current_ax.set_title(title, fontsize=14) # Changed fontsize
    current_ax.set_xlabel(txt('x_coord'))
    current_ax.set_ylabel(txt('y_coord'))
    current_ax.grid(True, alpha=0.3, zorder=Z_GRID)
    # current_ax.legend(handles=[depot_proxy], labels=[txt('depot_label')]) # Old depot-only legend

    legend_handles = []
    legend_labels = []

    # 1. Depot legend item (always add)
    depot_proxy = Line2D([0], [0], linestyle='none', marker='*', color='red', markersize=10, label=txt('depot_label'))
    legend_handles.append(depot_proxy)
    legend_labels.append(txt('depot_label'))

    # 2. Customer Node legend items (based on subproblem_legend_info)
    if subproblem_legend_info:
        # Ensure we only add one legend item per unique subproblem color/label presented in the actual plot
        # customer_node_colors contains the color for each plotted node.
        # subproblem_legend_info contains info for all original subproblems.
        # We need to find which of these original subproblems are actually represented by nodes in the current merged_instance.
        
        # Get unique colors present in customer_node_colors (these are the colors of nodes ACTUALLY PLOTTED)
        unique_plotted_node_colors_tuples = sorted(list(set(map(tuple, customer_node_colors))), key=lambda c: (c[0], c[1], c[2])) if customer_node_colors else []
        
        for color_info_dict in subproblem_legend_info:
            # color_info_dict is like {"color": (r,g,b,a), "label": "Subproblem 1", "id": 0}
            color_tuple_from_info = tuple(color_info_dict["color"]) # Ensure it's a tuple for comparison
            base_subproblem_label = color_info_dict["label"]

            # Check if this subproblem's color is actually used by any plotted customer node
            if color_tuple_from_info in unique_plotted_node_colors_tuples:
                node_proxy = Line2D([0], [0], linestyle='none', marker='o', color=list(color_tuple_from_info), markersize=8, label=f"{base_subproblem_label} Nodes")
                legend_handles.append(node_proxy)
                legend_labels.append(f"{base_subproblem_label} Nodes")

    # 3. Route legend items (based on dominant subproblem color of routes)
    added_route_legend_for_base_color = set() # To track base colors for which route legends have been added
    if subproblem_legend_info: # Need this to map route base colors back to subproblem labels
        # Create a quick lookup from a subproblem's base color (tuple) to its base label
        color_to_base_label_lookup = {tuple(info["color"]) : info["label"] for info in subproblem_legend_info}
        
        for base_color_rgba_tuple, route_indices_for_this_color in grouped_by_base_color.items():
            # base_color_rgba_tuple is the key from grouped_by_base_color (should be a tuple)
            # We only want one legend item per base color of routes
            if base_color_rgba_tuple not in added_route_legend_for_base_color and base_color_rgba_tuple in color_to_base_label_lookup:
                subproblem_base_label_for_route = color_to_base_label_lookup[base_color_rgba_tuple]
                # Use the base_color_rgba_tuple (without hue shift) for the legend line
                route_proxy = Line2D([0], [0], linestyle='-', color=list(base_color_rgba_tuple), linewidth=2, label=f"Routes for {subproblem_base_label_for_route}")
                legend_handles.append(route_proxy)
                legend_labels.append(f"Routes for {subproblem_base_label_for_route}")
                added_route_legend_for_base_color.add(base_color_rgba_tuple)
    
    # Display legend if there are items
    if len(legend_handles) > 1: # Only show legend if more than just depot or if specifically configured
        if show_annotations: # More detailed legend placement if annotations are on
            current_ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            if fig_created: plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust for external legend
        else: # Simpler legend for no annotations
            current_ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize='small', markerscale=0.8)
            if fig_created: plt.tight_layout()
    elif len(legend_handles) == 1 and fig_created : # Only depot, standard tight layout
        current_ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize='small', markerscale=0.8)
        plt.tight_layout()
    elif fig_created: # No legend items at all, but figure was created here
        plt.tight_layout()

    if fig_created is not None:
        plt.show()


def visualize_partitioned_instance(original_instance, subproblem_tuples, problem_type, lang='en', show_annotations=False, ax=None):
    """
    Visualizes the VRP instance highlighting nodes belonging to different subproblems.
    :param ax: Matplotlib axis object for subplotting. If None, a new figure is created.
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    print(txt('partitioned_viz_header', problem_type=problem_type))
    print(txt('num_subproblems', count=len(subproblem_tuples)))

    if not original_instance:
        print(txt('error_original_instance'))
        return

    try:
        depot_xy = original_instance[0]
        loc = original_instance[1] 
        capacity_idx = VRP_DATA_FORMAT[problem_type].index('capacity')
        capacity = original_instance[capacity_idx] if capacity_idx < len(original_instance) else 'N/A'
    except (IndexError, KeyError, TypeError) as e:
        print(txt('error_parse_original', e=e))
        return

    depot_coord = depot_xy[0] if isinstance(depot_xy[0], (list, tuple)) else depot_xy

    if ax is None:
        fig_created, current_ax = plt.subplots(figsize=(12, 12))
    else:
        fig_created = None
        current_ax = ax

    current_ax.scatter(depot_coord[0], depot_coord[1], c='red', s=50, marker='*', label=txt('depot_label'), zorder=Z_DEPOT) # Reduced s from 200 to 100

    num_subproblems = len(subproblem_tuples)
    colors_nodes = plt.cm.hsv(np.linspace(0.05, 0.95, num_subproblems)) if num_subproblems > 0 else [] 
    plotted_original_indices = set()

    for i, sub_tuple in enumerate(subproblem_tuples):
        try:
            sub_loc = sub_tuple[1]
            sub_num_nodes = len(sub_loc)
            if not sub_loc: continue

            original_indices_for_sub = []
            sub_loc_tuples = [tuple(coord) for coord in sub_loc]
            original_loc_tuples = [tuple(coord) for coord in loc]
            for sl_tuple in sub_loc_tuples:
                 try:
                      original_idx = original_loc_tuples.index(sl_tuple)
                      original_indices_for_sub.append(original_idx)
                      plotted_original_indices.add(original_idx)
                 except ValueError:
                      sl_original_format = sub_loc[sub_loc_tuples.index(sl_tuple)]
                      print(txt('warning_coord_not_found', coord=sl_original_format, i=i+1))

            current_color = colors_nodes[i % len(colors_nodes)] if colors_nodes.any() else 'gray' # Added modulo for safety
            label = txt('subproblem_label', i=i+1, count=sub_num_nodes) if show_annotations else None

            sub_x = [p[0] for p in sub_loc]
            sub_y = [p[1] for p in sub_loc]
            current_ax.scatter(sub_x, sub_y, color=current_color, s=25, label=label, alpha=0.8, zorder=Z_CUSTOMER_NODE) # Reduced s from 100 to 50

            if show_annotations:
                for k, idx in enumerate(original_indices_for_sub):
                    current_ax.text(sub_x[k], sub_y[k], str(idx + 1), fontsize=9, ha='center', va='center',
                                    color='white' if np.mean(current_color[:3]) < 0.5 else 'black', zorder=Z_ANNOTATION)
        except (IndexError, TypeError) as e:
            print(txt('error_process_subproblem', i=i+1, e=e))
            continue

    unassigned_loc = []
    unassigned_indices = []
    for idx, location in enumerate(loc):
        if idx not in plotted_original_indices:
            unassigned_loc.append(location)
            unassigned_indices.append(idx)

    if unassigned_loc:
        unassigned_node_indices_str = ", ".join(map(str, [idx+1 for idx in unassigned_indices]))
        print(txt('warning_unassigned_nodes', count=len(unassigned_loc), indices=unassigned_node_indices_str))
        unassigned_x = [p[0] for p in unassigned_loc]
        unassigned_y = [p[1] for p in unassigned_loc]
        unassigned_label_text = txt('unassigned_label') if show_annotations else None
        current_ax.scatter(unassigned_x, unassigned_y, c='gray', s=25, marker='x', label=unassigned_label_text, alpha=0.6, zorder=Z_CUSTOMER_NODE) # Reduced s from 80 to 40
        if show_annotations:
            for k, idx in enumerate(unassigned_indices):
                 current_ax.text(unassigned_x[k], unassigned_y[k], str(idx + 1), fontsize=9, ha='center', va='center', color='black', zorder=Z_ANNOTATION)

    if show_annotations:
        title = txt('partitioned_title', problem_type=problem_type, n=len(loc), k=num_subproblems)
    else:
        title = f"Partitioned and Merged Subproblems"
    current_ax.set_title(title, fontsize=14) # Changed fontsize
    current_ax.set_xlabel(txt('x_coord'))
    current_ax.set_ylabel(txt('y_coord'))
    current_ax.grid(True, alpha=0.3, zorder=Z_GRID)

    legend_handles = []
    legend_labels = []

    # Depot legend item (always add)
    depot_proxy = Line2D([0], [0], linestyle='none', marker='*', color='red', markersize=10, label=txt('depot_label'))
    legend_handles.append(depot_proxy)
    legend_labels.append(txt('depot_label'))

    # Subproblem nodes legend items
    # Create a unique set of colors and their corresponding first-seen subproblem index for legend
    added_subproblem_colors_to_legend = {} # Store color_tuple -> label_text
    if subproblem_tuples: # Ensure there are subproblems
        for i, sub_tuple in enumerate(subproblem_tuples):
            if not sub_tuple or len(sub_tuple) < 2 or not sub_tuple[1]: # Skip if subproblem is malformed or has no locations
                continue
            current_color_for_legend = tuple(colors_nodes[i % len(colors_nodes)]) if colors_nodes.any() else (.5,.5,.5,1) # Use tuple for dict key
            
            if current_color_for_legend not in added_subproblem_colors_to_legend:
                sub_num_nodes = len(sub_tuple[1])
                if show_annotations:
                    label_text = txt('subproblem_label', i=i+1, count=sub_num_nodes)
                else:
                    # For "Subproblem {i} ({count} nodes)", we want "Subproblem {i}"
                    # For "子问题 {i} ({count} 个节点)", we want "子问题 {i}"
                    raw_label_template = txt('subproblem_label', i=i+1, count='TEMPORARY_COUNT_PLACEHOLDER')
                    label_text = raw_label_template.split(' (TEMPORARY_COUNT_PLACEHOLDER')[0].strip()
                added_subproblem_colors_to_legend[current_color_for_legend] = label_text
    
    # Add collected unique subproblem colors to legend handles/labels
    for color_tuple, label_text in added_subproblem_colors_to_legend.items():
        sub_proxy = Line2D([0], [0], linestyle='none', marker='o', color=list(color_tuple), markersize=8, label=label_text)
        legend_handles.append(sub_proxy)
        legend_labels.append(label_text)

    # Unassigned nodes legend item (if any were plotted)
    if unassigned_loc: # Check if unassigned_loc list is not empty
        unassigned_proxy = Line2D([0], [0], linestyle='none', marker='x', color='gray', markersize=8, label=txt('unassigned_label'))
        legend_handles.append(unassigned_proxy)
        legend_labels.append(txt('unassigned_label'))

    # Display legend if there are items to display
    if legend_handles:
        # Adjust legend positioning based on show_annotations (as before)
        if show_annotations:
            if num_subproblems > 10:
                 current_ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
                 if fig_created: plt.tight_layout(rect=[0, 0, 0.8, 1])
            else:
                 current_ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 if fig_created: plt.tight_layout(rect=[0, 0, 0.85, 1])
        else: # When show_annotations is False, use a simpler legend placement
            current_ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize='small', markerscale=0.8)
            if fig_created: plt.tight_layout() # Standard tight layout
    elif fig_created: # Apply standard tight_layout only if a new figure was made and no legend items generated
        plt.tight_layout()

    if fig_created is not None:
        plt.show()


def visualize_padded_subproblems(original_instance, original_subproblem_tuples, padded_subproblem_batch_tuples, problem_type, lang='en', show_annotations=False, ax=None):
    """
    Visualizes the VRP instance highlighting original subproblem nodes and indicating padding.
    :param ax: Matplotlib axis object for subplotting. If None, a new figure is created.
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    
    if not original_instance:
        print(txt('error_original_instance'))
        return

    try:
        depot_xy = original_instance[0]
        all_original_loc = original_instance[1]
        capacity_idx = VRP_DATA_FORMAT[problem_type].index('capacity')
        capacity = original_instance[capacity_idx] if capacity_idx < len(original_instance) else 'N/A'
    except (IndexError, KeyError, TypeError) as e:
        print(txt('error_parse_original', e=e))
        return

    depot_coord = depot_xy[0] if isinstance(depot_xy[0], (list, tuple)) else depot_xy

    if ax is None:
        fig_created, current_ax = plt.subplots(figsize=(12, 12))
    else:
        fig_created = None
        current_ax = ax
    
    current_ax.scatter(depot_coord[0], depot_coord[1], c='red', s=50, marker='*', label=txt('depot_label'), zorder=Z_DEPOT)

    num_original_subproblems = len(original_subproblem_tuples)
    colors_nodes = plt.cm.hsv(np.linspace(0.05, 0.95, num_original_subproblems)) if num_original_subproblems > 0 else []
    plotted_original_indices_global = set()

    for i, orig_sub_tuple in enumerate(original_subproblem_tuples):
        if not orig_sub_tuple or len(orig_sub_tuple) < 2:
            print(f"Warning: Original subproblem {i+1} is malformed or empty, skipping.")
            continue
            
        orig_sub_loc = orig_sub_tuple[1]
        if not orig_sub_loc:
            num_padding_nodes_added = len(padded_subproblem_batch_tuples[i][1]) if padded_subproblem_batch_tuples and i < len(padded_subproblem_batch_tuples) else 0
        else:
            num_padding_nodes_added = len(padded_subproblem_batch_tuples[i][1]) - len(orig_sub_loc)

        current_color = colors_nodes[i % len(colors_nodes)] if colors_nodes.any() else 'gray'

        if orig_sub_loc:
            sub_x = [p[0] for p in orig_sub_loc]
            sub_y = [p[1] for p in orig_sub_loc]
            label = txt('subproblem_label', i=i+1, count=len(orig_sub_loc)) if show_annotations else None
            current_ax.scatter(sub_x, sub_y, color=current_color, s=25, label=label, alpha=0.8, zorder=Z_CUSTOMER_NODE) # Reduced s from 100 to 50

            orig_loc_tuples = [tuple(coord) for coord in all_original_loc]
            for k, sl_coord in enumerate(orig_sub_loc):
                try:
                    original_idx = orig_loc_tuples.index(tuple(sl_coord))
                    plotted_original_indices_global.add(original_idx)
                    if show_annotations:
                        current_ax.text(sub_x[k], sub_y[k], str(original_idx + 1), fontsize=9, ha='center', va='center',
                                        color='white' if np.mean(current_color[:3]) < 0.5 else 'black', zorder=Z_ANNOTATION)
                except ValueError:
                    pass

        if num_padding_nodes_added > 0:
            padding_repr_offset_factor = 0.035
            if orig_sub_loc:
                centroid_x = np.mean([p[0] for p in orig_sub_loc])
                centroid_y = np.mean([p[1] for p in orig_sub_loc])
                direction_x = centroid_x - depot_coord[0]
                direction_y = centroid_y - depot_coord[1]
                norm = np.sqrt(direction_x**2 + direction_y**2)
                if norm > 0:
                    offset_x = (direction_x / norm) * (max(all_original_loc, key=lambda p: p[0])[0] - min(all_original_loc, key=lambda p: p[0])[0]) * padding_repr_offset_factor
                    offset_y = (direction_y / norm) * (max(all_original_loc, key=lambda p: p[1])[1] - min(all_original_loc, key=lambda p: p[1])[1]) * padding_repr_offset_factor
                else: 
                    offset_x = (max(all_original_loc, key=lambda p: p[0])[0] - min(all_original_loc, key=lambda p: p[0])[0]) * padding_repr_offset_factor * 0.5
                    offset_y = 0
            else: 
                offset_x = (max(all_original_loc, key=lambda p: p[0])[0] - min(all_original_loc, key=lambda p: p[0])[0]) * padding_repr_offset_factor * ( (i%4-1.5)*0.5 )
                offset_y = (max(all_original_loc, key=lambda p: p[1])[1] - min(all_original_loc, key=lambda p: p[1])[1]) * padding_repr_offset_factor * ( (i%2-0.5)*0.5 )

            padding_repr_x = depot_coord[0] + offset_x
            padding_repr_y = depot_coord[1] + offset_y
            
            current_ax.scatter(padding_repr_x, padding_repr_y, color=current_color, 
                                s=50, marker='s', alpha=0.8, zorder=Z_PADDING_MARKER, # Reduced s from 100 to 50
                                edgecolors='black', linewidths=1.0)

            # Calculate y-range for text offset to position text above marker
            min_y_coord_val = min(loc[1] for loc in all_original_loc) if all_original_loc else 0
            max_y_coord_val = max(loc[1] for loc in all_original_loc) if all_original_loc else 1
            y_coord_data_range = max_y_coord_val - min_y_coord_val
            if y_coord_data_range <= 1e-6: # If range is zero or very small, use a default
                y_coord_data_range = 1.0 
            
            text_vertical_offset_factor = 0.01 # Adjust this factor as needed (e.g., 0.02 to 0.05 of y-range)
            text_y_position = padding_repr_y + (y_coord_data_range * text_vertical_offset_factor)

            current_ax.text(padding_repr_x, text_y_position, f"×{num_padding_nodes_added}", 
                             fontsize=10, ha='center', va='bottom', color='black', zorder=Z_ANNOTATION)

    unassigned_loc = []
    if all_original_loc:
        for idx, location in enumerate(all_original_loc):
            if idx not in plotted_original_indices_global:
                unassigned_loc.append(location)
        if unassigned_loc:
            unassigned_x = [p[0] for p in unassigned_loc]
            unassigned_y = [p[1] for p in unassigned_loc]
            label = txt('unassigned_label') if show_annotations else None
            current_ax.scatter(unassigned_x, unassigned_y, c='lightgray', s=25, marker='x', label=label, alpha=0.5, zorder=Z_CUSTOMER_NODE) # Reduced s from 80 to 40
            if show_annotations:
                for k, loc_coord in enumerate(unassigned_loc):
                    try:
                        original_idx = [tuple(c) for c in all_original_loc].index(tuple(loc_coord))
                        current_ax.text(loc_coord[0], loc_coord[1], str(original_idx + 1), fontsize=9, ha='center', va='center', color='dimgray', zorder=Z_ANNOTATION)
                    except ValueError:
                        pass

    if show_annotations:
        title = f"{problem_type} Padded Subproblems ({num_original_subproblems} Original Subproblems)"
        if all_original_loc:
            title += f" - N={len(all_original_loc)}"
    else:
        title = f"Padded Subproblems"
    current_ax.set_title(title, fontsize=14) # Ensure fontsize is 14
    current_ax.set_xlabel(txt('x_coord'))
    current_ax.set_ylabel(txt('y_coord'))
    current_ax.grid(True, alpha=0.3, zorder=Z_GRID)

    legend_handles = []
    legend_labels = []

    # Depot legend item
    depot_proxy = Line2D([0], [0], linestyle='none', marker='*', color='red', markersize=10, label=txt('depot_label'))
    legend_handles.append(depot_proxy)
    legend_labels.append(txt('depot_label'))

    # Subproblem nodes and Padding markers legend items
    added_subproblem_colors_to_legend_padded = {} # color_tuple -> label_text for nodes
    # For padding markers, we might not need individual legend items if they share color with nodes
    # and their meaning is clear by shape + text. However, if we want a generic "Padding" marker legend:
    # has_padding_marker_legend_item = False

    if original_subproblem_tuples:
        for i, orig_sub_tuple in enumerate(original_subproblem_tuples):
            if not orig_sub_tuple or len(orig_sub_tuple) < 2: continue
            
            current_color_for_legend = tuple(colors_nodes[i % len(colors_nodes)]) if colors_nodes.any() else (.5,.5,.5,1)

            # Legend for original subproblem nodes
            if orig_sub_tuple[1]: # If there are original nodes in this subproblem
                if current_color_for_legend not in added_subproblem_colors_to_legend_padded:
                    sub_num_nodes = len(orig_sub_tuple[1]) # Number of original nodes in this subproblem
                    if show_annotations:
                        label_text = txt('subproblem_label', i=i+1, count=sub_num_nodes)
                    else:
                        raw_label_template = txt('subproblem_label', i=i+1, count='TEMPORARY_COUNT_PLACEHOLDER')
                        label_text = raw_label_template.split(' (TEMPORARY_COUNT_PLACEHOLDER')[0].strip()
                    added_subproblem_colors_to_legend_padded[current_color_for_legend] = label_text
            
            # Potentially add a single legend item for Padding Markers if desired
            # num_padding_nodes_added = ... (calculation as before)
            # if num_padding_nodes_added > 0 and not has_padding_marker_legend_item:
            #     padding_proxy = Line2D([0], [0], linestyle='none', marker='s', color='grey', markersize=8, label='Padding Site')
            #     legend_handles.append(padding_proxy)
            #     legend_labels.append('Padding Site')
            #     has_padding_marker_legend_item = True

    for color_tuple, label_text in added_subproblem_colors_to_legend_padded.items():
        sub_node_proxy = Line2D([0], [0], linestyle='none', marker='o', color=list(color_tuple), markersize=8, label=label_text)
        legend_handles.append(sub_node_proxy)
        legend_labels.append(label_text)

    # Unassigned nodes legend item
    # Check if unassigned_loc was populated during plotting section
    # To do this properly, we might need to see if any unassigned_loc.append was called.
    # For now, assume if the plotting logic for unassigned nodes runs, we add its legend.
    # This might require a flag set during plotting of unassigned nodes.
    # Let's assume 'unassigned_loc_plotted' is a boolean flag set if unassigned_loc had data and was plotted.
    # This flag would need to be set in the plotting section: e.g., unassigned_loc_plotted = True if unassigned_loc else False
    # For this edit, I will assume you'll add such a flag or check `if current_ax.collections` contains unassigned markers.
    # Simplified: if the code path for plotting unassigned nodes could have run.
    # We need to know if `unassigned_loc` (calculated earlier in the function) had items.
    # The `all_original_loc` and `plotted_original_indices_global` are used to determine `unassigned_loc`.
    # A bit of a reconstruct here for the condition:
    was_unassigned_plotted = False
    if all_original_loc: # From earlier in the function
        temp_unassigned = []
        for idx_check in range(len(all_original_loc)):
            if idx_check not in plotted_original_indices_global: # also from earlier
                temp_unassigned.append(all_original_loc[idx_check])
        if temp_unassigned:
            was_unassigned_plotted = True
            
    if was_unassigned_plotted:
        unassigned_proxy = Line2D([0], [0], linestyle='none', marker='x', color='gray', markersize=8, label=txt('unassigned_label'))
        legend_handles.append(unassigned_proxy)
        legend_labels.append(txt('unassigned_label'))

    if legend_handles:
        if show_annotations and num_original_subproblems > 0 :
            current_ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            if fig_created: plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            current_ax.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize='small', markerscale=0.8)
            if fig_created: plt.tight_layout()
    elif fig_created:
        plt.tight_layout()

    if fig_created is not None:
        plt.show()


def visualize_full_partition_solve_process(
    original_instance_data,
    problem_type,
    partitioner_checkpoint_path,
    partitioner_model_params, # Dict of params for partitioner model
    solver_checkpoint_path,
    solver_model_params,     # Dict of params for solver model
    merge_num,
    device_str='cuda', # 'cuda' or 'cpu'
    lang='en',
    show_annotations=False,
    max_seq_len_factor=2, # For partition_instance
    solver_aug_factor=8,   # For solve_vrp_batch
    adaptive_merge_target_size: int = 0, # New: Target size for adaptive merging. <=0 for dynamic.
    output_mode='subplots', # New: 'sequential' or 'subplots'
    save_path=None,         # New: Base path/filename for saving subplots image (e.g., './my_vrp_viz')
    save_formats=['pdf']    # New: List of formats to save, e.g., ['pdf', 'eps', 'png']
    ):
    """
    Visualizes the full VRP partitioning and solving process.
    1. Original VRP instance.
    2. Partitioned VRP instance (nodes colored by subproblem).
    3. Padded subproblems (original nodes colored, padding indicated).
    4. Final merged solution from solved subproblems (nodes colored by original subproblem).

    Args:
        original_instance_data: The raw instance data (e.g., from load_dataset).
        problem_type (str): Type of VRP problem (e.g., 'CVRP').
        partitioner_checkpoint_path (str): Path to the partitioner model checkpoint.
        partitioner_model_params (dict): Parameters for the partitioner model.
        solver_checkpoint_path (str): Path to the solver model checkpoint.
        solver_model_params (dict): Parameters for the solver model.
        merge_num (int): Merge number for partitioning. <=0 for adaptive merge.
        device_str (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
        lang (str, optional): Language for plot annotations ('en' or 'zh'). Defaults to 'en'.
        show_annotations (bool, optional): Whether to show node annotations. Defaults to False.
        max_seq_len_factor (int, optional): Factor for max sequence length in partitioner. Defaults to 2.
        solver_aug_factor (int, optional): Augmentation factor for solver. Defaults to 8.
        adaptive_merge_target_size (int, optional): Target node count for adaptive merging.
                                                  If >0, this value is used. 
                                                  If <=0, dynamic sizing based on problem_size is used. 
                                                  Defaults to 0 (dynamic).
        output_mode (str, optional): 'sequential' (multiple separate plots) or 'subplots' (single figure with subplots).
                                   Defaults to 'subplots'.
        save_path (str | None, optional): Base path/filename for saving plots. If None, plots are shown directly.
                                        Example: './visual_outputs/my_run'. Format extension is added automatically.
                                        Defaults to None.
        save_formats (list[str], optional): List of image formats to save (e.g., ['pdf', 'png']). 
                                          Applicable if save_path is not None. Defaults to ['pdf'].
    """
    txt = lambda key, **kwargs: get_text(key, lang, **kwargs)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__) # Use a logger
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    fig = None # Initialize fig and axes for potential subplot usage
    axes = None

    if output_mode == 'subplots':
        fig, axes = plt.subplots(2, 2, figsize=(24, 24)) # Create a 2x2 subplot grid
        # Flatten axes array for easier indexing if preferred, or use axes[r,c]
        # axes = axes.flatten() # Example: axes[0], axes[1], axes[2], axes[3]
        # For 2x2, direct indexing is also clear: axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    if not original_instance_data:
        print(txt('error_original_instance'))
        return

    # --- Step 1: Visualize Original Instance ---
    print_vrp_instance_info(original_instance_data, problem_type, lang=lang)
    # Pass ax if in subplots mode
    current_ax = axes[0, 0] if output_mode == 'subplots' and axes is not None else None
    visualize_vrp_instance(original_instance_data, problem_type, lang=lang, show_annotations=show_annotations, ax=current_ax)
    if output_mode == 'sequential':
        print("\n" + "="*30 + " Step 1 End " + "="*30 + "\n")

    # --- Step 2: Partition and Visualize Partitioned Instance ---
    subproblem_tuples = None
    raw_sequence = None
    partitioner_model = None
    subproblem_color_map = [] # Initialize color map
    subproblem_legend_info_for_solution_plot = [] # New: For passing to colored_solution legend

    try:
        print(txt('attempting_partition'))
        # Ensure partitioner_model_params is a copy if it's based on a default
        current_partitioner_params = partitioner_model_params.copy()
        current_partitioner_params['problem'] = problem_type # Ensure problem type is set for partitioner

        partitioner_model = load_moe_model(
            partitioner_checkpoint_path,
            device,
            model_type=current_partitioner_params.get('model_type'),
            model_params=current_partitioner_params
        )
        if not partitioner_model:
            raise ValueError("Failed to load partitioner model.")

        subproblem_tuples, raw_sequence = partition_instance(
            original_instance_tuple=original_instance_data,
            problem_type=problem_type,
            partitioner_checkpoint_path=None, # Model is pre-loaded
            merge_num=merge_num,
            device=device,
            partitioner_model_params=current_partitioner_params, # Still needed for env setup within partition_instance
            max_seq_len_factor=max_seq_len_factor,
            partitioner_model=partitioner_model, # Pass pre-loaded model
            target_node_count_for_merge=adaptive_merge_target_size # Pass the new target size
        )

        if subproblem_tuples is not None and subproblem_tuples: # Check if not None and not empty
            print(txt('partition_success'))
            if subproblem_tuples:
                num_actual_subproblems = len(subproblem_tuples)
                subproblem_color_map = plt.cm.hsv(np.linspace(0.05, 0.95, num_actual_subproblems)) if num_actual_subproblems > 0 else []
                # Prepare legend info for the solution plot
                for i in range(num_actual_subproblems):
                    raw_label_template = txt('subproblem_label', i=i+1, count='TEMPORARY_COUNT_PLACEHOLDER')
                    base_label = raw_label_template.split(' (TEMPORARY_COUNT_PLACEHOLDER')[0].strip()
                    subproblem_legend_info_for_solution_plot.append({
                        "color": tuple(subproblem_color_map[i]), # Ensure color is hashable for later use if needed
                        "label": base_label,
                        "id": i # Original index, could be useful
                    })

            current_ax = axes[0, 1] if output_mode == 'subplots' and axes is not None else None
            visualize_partitioned_instance(original_instance_data, subproblem_tuples, problem_type, lang=lang, show_annotations=show_annotations, ax=current_ax)
        else:
            print(txt('partition_fail'))
            if raw_sequence:
                print(txt('raw_sequence_info', seq=raw_sequence[:50]))
            # If partitioning failed, we might not be able to proceed with later visualizations.
            # For now, we'll let it try, but ideally, we'd handle this more gracefully.
            if not subproblem_tuples: subproblem_tuples = [] # Ensure it's an empty list if None for safety
    except Exception as e:
        logger.error(f"Error in partitioning (Step 2): {e}")
        print(txt('partition_fail'))
        if raw_sequence: print(txt('raw_sequence_info', seq=raw_sequence[:50]))
        subproblem_tuples = [] # Ensure it's an empty list on error
        subproblem_color_map = [] # Reset color map on error
        subproblem_legend_info_for_solution_plot = [] # Reset on error
    
    if output_mode == 'sequential':
        print("\n" + "="*30 + " Step 2 End " + "="*30 + "\n")

    # Ensure subproblem_tuples is a list, even if empty, for the next steps
    if subproblem_tuples is None:
        subproblem_tuples = []

    # --- Step 3: Pad Subproblems and Visualize Padded Subproblems ---
    padded_subproblem_batch_tuples = []
    actual_pad_size = 0
    if subproblem_tuples: # Only proceed if partitioning yielded subproblems
        try:
            print("Padding subproblems...")
            # Determine target_pad_size: pad to the size of the largest subproblem or a fixed N if desired
            # For now, let partitioner_solver_utils.pad_subproblem_batch decide (pads to max in batch)
            padded_subproblem_batch_tuples, actual_pad_size = pad_subproblem_batch(
                subproblem_tuples, problem_type, target_pad_size=None
            )
            if padded_subproblem_batch_tuples:
                print(f"Subproblems padded to size: {actual_pad_size}")
                # Pass the generated subproblem_color_map to visualize_padded_subproblems if it needs to use the same colors
                # For now, visualize_padded_subproblems generates its own colors based on original_subproblem_tuples length.
                # If consistency is strictly needed, visualize_padded_subproblems might need modification or to receive colors.
                current_ax = axes[1, 0] if output_mode == 'subplots' and axes is not None else None
                visualize_padded_subproblems(
                    original_instance_data,
                    subproblem_tuples, 
                    padded_subproblem_batch_tuples,
                    problem_type,
                    lang=lang,
                    show_annotations=show_annotations,
                    ax=current_ax
                )
            else:
                print("Failed to pad subproblems or no subproblems to pad.")
        except Exception as e:
            logger.error(f"Error in padding visualization (Step 3): {e}")
            print("Failed to visualize padded subproblems.")
    else:
        print("Skipping padding visualization as no subproblems were generated in Step 2.")
        
    print("\n" + "="*30 + " Step 3 End " + "="*30 + "\n")


    # --- Step 4: Solve Padded Subproblems, Merge, and Visualize Final Solution ---
    print(f"Step 4: Solve, Merge, and Visualize Final Solution (Partially implemented, coloring enhancement pending)") # Corrected print
    if not subproblem_tuples or not padded_subproblem_batch_tuples:
        print("Skipping Step 4 as prerequisite data (subproblems or padded subproblems) is missing.")
    else:
        try:
            # 4.1 Prepare tensor data for solver
            padded_batch_tensor_data = prepare_batch_tensor_data(padded_subproblem_batch_tuples, problem_type, device)
            if not padded_batch_tensor_data:
                raise ValueError("Failed to prepare batch tensor data for solver.")

            # 4.2 Load solver model
            current_solver_params = solver_model_params.copy()
            current_solver_params['problem'] = problem_type # Ensure problem is set

            solver_model = load_moe_model(
                solver_checkpoint_path,
                device,
                model_type=current_solver_params.get('model_type'),
                model_params=current_solver_params
            )
            if not solver_model:
                raise ValueError("Failed to load solver model.")

            # 4.3 Get solver environment
            EnvClassList = get_env(problem_type)
            if not EnvClassList:
                raise ValueError(f"Could not get environment class for problem type: {problem_type}")
            SolverEnvClass = EnvClassList[0]

            # 4.4 Solve batch
            print(f"Solving {len(padded_subproblem_batch_tuples)} padded subproblems (padded size: {actual_pad_size})...")
            solved_results_for_padded = solve_vrp_batch(
                solver_model=solver_model,
                solver_env_class=SolverEnvClass,
                original_instance_tuples=subproblem_tuples, 
                padded_batch_data=padded_batch_tensor_data,
                padded_problem_size=actual_pad_size,
                problem_type=problem_type,
                device=device,
                aug_factor=solver_aug_factor
            )
            print(f"Subproblems solved. Number of results: {len(solved_results_for_padded)}")

            cleaned_sub_solutions = []
            if len(solved_results_for_padded) != len(subproblem_tuples):
                print(f"Warning: Mismatch between number of solved results ({len(solved_results_for_padded)}) and original subproblems ({len(subproblem_tuples)}). Cannot merge reliably.")
            else:
                for i, (score, padded_path) in enumerate(solved_results_for_padded):
                    if score == float('inf') or padded_path is None:
                        print(f"Warning: Subproblem {i+1} did not yield a valid solution (score: {score}). Skipping for merge.")
                        continue

                    original_sub_node_xy = subproblem_tuples[i][1]
                    num_original_nodes_in_sub = len(original_sub_node_xy)
                    
                    path_cleaned_for_merge = []
                    if padded_path:
                        for node_idx_in_padded_path in padded_path:
                            if node_idx_in_padded_path == 0:
                                path_cleaned_for_merge.append(0)
                            elif 1 <= node_idx_in_padded_path <= num_original_nodes_in_sub:
                                path_cleaned_for_merge.append(node_idx_in_padded_path)
                    cleaned_sub_solutions.append((score, path_cleaned_for_merge))
                
                print(f"Number of cleaned solutions for merge: {len(cleaned_sub_solutions)}")

            if cleaned_sub_solutions:
                print("Merging solved subproblem solutions...")
                merged_instance, merged_solution = merge_solved_instances(
                    raw_instance_datas=subproblem_tuples,
                    solved_sols=cleaned_sub_solutions
                )

                if merged_instance and merged_solution:
                    print("Solutions merged. Visualizing final solution...")
                    
                    # --- Construct final_node_colors for the merged_instance ---
                    final_node_colors = []
                    if subproblem_tuples and subproblem_color_map.any():
                        # merged_instance[1] contains all customer locations in the merged order
                        # subproblem_tuples[i][1] contains customer locations for the i-th subproblem
                        # The order in merged_instance[1] is sequential concatenation of subproblem_tuples[i][1]
                        for i, sub_tuple in enumerate(subproblem_tuples):
                            num_nodes_in_this_sub = len(sub_tuple[1]) # Number of customer nodes in this subproblem
                            color_for_this_sub = subproblem_color_map[i % len(subproblem_color_map)] # Cycle through colors if needed, though lengths should match
                            final_node_colors.extend([color_for_this_sub] * num_nodes_in_this_sub)
                    
                    if not final_node_colors or len(final_node_colors) != len(merged_instance[1]):
                        print("Warning: Could not generate valid colors for all merged nodes. Using default for final solution plot.")
                        # Fallback: create a default color list if generation failed
                        final_node_colors = ['blue'] * len(merged_instance[1]) if merged_instance[1] else []

                    print_vrp_instance_info(merged_instance, problem_type, lang=lang) 
                    current_ax = axes[1, 1] if output_mode == 'subplots' and axes is not None else None
                    visualize_colored_solution(merged_instance, merged_solution, problem_type, 
                                               customer_node_colors=final_node_colors, 
                                               lang=lang, show_annotations=show_annotations,
                                               ax=current_ax,
                                               subproblem_legend_info=subproblem_legend_info_for_solution_plot # Pass new legend info
                                               )
                    
                    # Remove the old placeholder notes about default colors
                    # print("\nNOTE: Final solution visualization currently uses default node colors.")
                    # print("Implement 'visualize_colored_solution' for subproblem-specific node colors in the final plot.")

                else:
                    print("Failed to merge solutions or no solutions to merge.")
        except Exception as e: # Added except block for the try statement at line 783
            logger.error(f"Error in solving/merging (Step 4): {e}")
            import traceback
            traceback.print_exc()
            print("Failed to solve/merge subproblems.")

    if output_mode == 'subplots' and fig is not None:
        # plt.tight_layout(pad=3.0) # Original tight_layout
        # Adjust layout to prevent overlap, then explicitly adjust hspace
        fig.tight_layout(pad=3.0) # Apply tight_layout first to get a base good layout
        fig.subplots_adjust(hspace=0.1) # Increase vertical spacing between subplots

        if save_path:
            # Ensure directory for save_path exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Created directory for saving figures: {save_dir}")
                except OSError as e:
                    print(f"Error creating directory {save_dir}: {e}. Figures may not be saved.")
            
            for fmt in save_formats:
                try:
                    full_save_path = f"{save_path}.{fmt}"
                    fig.savefig(full_save_path, format=fmt, bbox_inches='tight', dpi=300)
                    print(f"Subplots image saved to {full_save_path}")
                except Exception as save_e:
                    print(f"Error saving image to {full_save_path}: {save_e}")
        plt.show() # Show after trying to save
    elif output_mode == 'sequential': # Ensure final print for sequential is clear
        print("\n" + "="*30 + " Step 4 End (Sequential Mode) " + "="*30 + "\n")

    print("Full visualization process finished.")


# Example Usage (Add to the end of data_visualize.py or a separate script)
if __name__ == '__main__':
    import torch # Add necessary imports for partition_instance
    # Assume partitioner_solver_utils is in the path or same directory
    try:
        # Import the main new function we want to test
        # from data_visualize import visualize_full_partition_solve_process # Not needed if in the same file
        
        # Imports for partition_instance and other utils used in the example
        from partitioner_solver_utils import partition_instance, DEFAULT_MODEL_PARAMS as PsuDefaults, create_subproblem_instance
        partitioner_utils_available = True
    except ImportError as e:
        print(f"Warning: Could not import from partitioner_solver_utils ({e}). Some functionalities might be limited.")
        partitioner_utils_available = False
        def partition_instance(*args, **kwargs): return None, None # Dummy
        PsuDefaults = {} # Dummy
        def create_subproblem_instance(*args, **kwargs): return None # Dummy

    # Configure Logger for Testing (can be simplified for visualization script)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # --- Parameters for Testing ---
    problem_to_test = "CVRP"
    instance_size_to_test = 100
    instance_idx_to_test = 66
    dataset_path_to_test = f'./data/{problem_to_test}/{problem_to_test.lower()}{instance_size_to_test}_uniform.pkl'
    lang_to_use_test = 'en' # Set language here: 'en' or 'zh'
    show_annotations_test = False # Set to True to see more details on plots

    # --- Define Partitioner Model Parameters (MUST match the loaded checkpoint) ---
    # Example: Using parameters consistent with 'MOE_4e_CVRP_n200...'
    # Make sure these params are correct for your specific partitioner_checkpoint_path
    partitioner_checkpoint_path_for_test = os.path.join(
        'results', 'TAM_MoE', 'MOE_4e_CVRP_n200_20250425_050856', '40.pt'
    ) 
    # if you use a different partitioner, adjust these params:
    actual_partitioner_model_params = {
        "model_type": "MOE", 
        "embedding_dim": 128, 
        "sqrt_embedding_dim": 128**(1/2),
        "encoder_layer_num": 6, 
        "decoder_layer_num": 1, # Often 1 for solver-based partitioners
        "qkv_dim": 16, 
        "head_num": 8,
        "logit_clipping": 10.0, 
        "ff_hidden_dim": 512, 
        "num_experts": 4, # Matches '4e' in the example path
        "eval_type": "argmax", 
        "norm": "instance", 
        "norm_loc": "norm_last", 
        "topk": 2,
        "expert_loc": ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
        "routing_level": "node", 
        "routing_method": "input_choice",
        "problem": problem_to_test # This will be set correctly within the main function too
    }

    # --- Define Solver Model Parameters (MUST match the loaded checkpoint) ---
    # Example: Using parameters consistent with 'mvmoe_8e_n50...'
    # Make sure these params are correct for your specific solver_checkpoint_path
    solver_checkpoint_path_for_test = os.path.join(
        'pretrained', 'mvmoe_8e_n50', 'epoch-2500.pt'
    ) 
    # if you use a different solver, adjust these params:
    actual_solver_model_params = {
        "model_type": "MOE", 
        "embedding_dim": 128, 
        "sqrt_embedding_dim": 128**(1/2),
        "encoder_layer_num": 6, 
        "decoder_layer_num": 1, 
        "qkv_dim": 16, 
        "head_num": 8,
        "logit_clipping": 10.0, 
        "ff_hidden_dim": 512, 
        "num_experts": 8, # Matches '8e' in the example path
        "eval_type": "argmax", 
        "norm": "instance", 
        "norm_loc": "norm_last", 
        "topk": 2,
        "expert_loc": ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
        "routing_level": "node", 
        "routing_method": "input_choice",
        "problem": problem_to_test # This will be set correctly
    }
    
    merge_num_to_test = 3 # Example, how many raw subproblems to merge
    device_to_use_test = "cuda" if torch.cuda.is_available() else "cpu"
    txt_main = lambda key, **kwargs: get_text(key, lang_to_use_test, **kwargs)

    logger.info(f"Using device: {device_to_use_test}")
    logger.info(f"Testing with problem: {problem_to_test}, size: {instance_size_to_test}, instance_idx: {instance_idx_to_test}")
    logger.info(f"Partitioner: {partitioner_checkpoint_path_for_test}")
    logger.info(f"Solver: {solver_checkpoint_path_for_test}")
    logger.info(f"Merge num: {merge_num_to_test}")

    # Check if dataset exists
    if not os.path.exists(dataset_path_to_test):
        print(txt_main('error_dataset_not_found', path=dataset_path_to_test))
    else:
        try:
            dataset = load_dataset(dataset_path_to_test)
            if not dataset or instance_idx_to_test >= len(dataset):
                 print(txt_main('error_instance_not_found', idx=instance_idx_to_test, path=dataset_path_to_test))
                 exit()
            instance_data_to_test = dataset[instance_idx_to_test]

            # --- Call the main visualization function --- 
            print("\nStarting the full visualization process...")
            visualize_full_partition_solve_process(
                original_instance_data=instance_data_to_test,
                problem_type=problem_to_test,
                partitioner_checkpoint_path=partitioner_checkpoint_path_for_test,
                partitioner_model_params=actual_partitioner_model_params,
                solver_checkpoint_path=solver_checkpoint_path_for_test,
                solver_model_params=actual_solver_model_params,
                merge_num=merge_num_to_test,
                adaptive_merge_target_size=50,
                device_str=device_to_use_test,
                lang=lang_to_use_test,
                show_annotations=show_annotations_test,
                output_mode='subplots',
                save_path='./figures/output_visualization', # 基础文件名，会生成 ./output_visualization.pdf 等
                save_formats=['pdf', 'eps']
                # Optional: Adjust these if needed for your models/instance sizes
                # max_seq_len_factor=2, 
                # solver_aug_factor=8 
            )
            print("\nTest execution of visualize_full_partition_solve_process finished.")

            # --- Old code (commented out as visualize_full_partition_solve_process handles these steps) ---
            # print_vrp_instance_info(instance_data_to_test, problem_to_test, lang=lang_to_use_test)
            # visualize_vrp_instance(instance_data_to_test, problem_to_test, lang=lang_to_use_test, show_annotations=show_annotations_test)
            
            # subproblem_tuples_old = None
            # if partitioner_utils_available:
            #     print(txt_main('attempting_partition'))
            #     # ... (old partitioning code) ... 
            # else:
            #     # ... (old dummy data code) ...

            # if subproblem_tuples_old:
            #      visualize_partitioned_instance(instance_data_to_test, subproblem_tuples_old, problem_to_test, lang=lang_to_use_test, show_annotations=show_annotations_test)
            
            # solution_path_old = f'./data/{problem_to_test}/hgs_{problem_to_test.lower()}{instance_size_to_test}_uniform.pkl' 
            # if os.path.exists(solution_path_old):
            #     # ... (old solution visualization code) ...
            # else:
            #     print(txt_main('solution_not_found', path=solution_path_old))

        except FileNotFoundError as e:
            print(e) # Should be caught by os.path.exists above, but as a fallback
        except ImportError as e:
             print(txt_main('import_error', e=e))
        except Exception as e:
            print(txt_main('unexpected_error', e=e))
            import traceback
            traceback.print_exc()
