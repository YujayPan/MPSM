import os
import sys
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from other directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cvrp_lib_integration.cvrp_lib_parser import parse_vrp_file, to_internal_tuple
from data_visualize import visualize_vrp_instance, print_vrp_instance_info # Assuming data_visualize.py is in root

def test_single_cvrp_lib_instance(file_path):
    """
    Tests reading, parsing, converting, and visualizing a single CVRP-LIB instance.
    """
    print(f"--- Testing Single CVRP-LIB Instance: {file_path} ---")

    # 1. Parse the .vrp file
    parsed_data = parse_vrp_file(file_path)

    if not parsed_data:
        print(f"Failed to parse {file_path}. Exiting test.")
        return False

    print("\nSuccessfully parsed VRP data:")
    # print_vrp_instance_info might expect a slightly different format or more fields than raw parsed_data
    # For now, we print selected items:
    print(f"  Name: {parsed_data.get('name')}")
    print(f"  Type: {parsed_data.get('problem_type')}")
    print(f"  Dimension: {parsed_data.get('dimension')}")
    print(f"  Capacity: {parsed_data.get('capacity')}")
    print(f"  Depot XY (parsed): {parsed_data.get('depot_xy')}")
    print(f"  Num Customer Coords (parsed): {len(parsed_data.get('node_xy', []))}")
    print(f"  Num Customer Demands (parsed): {len(parsed_data.get('demand', []))}")

    # 2. Convert to internal tuple format
    # (depot_xy, node_xy, demand, capacity)
    internal_instance_tuple = to_internal_tuple(parsed_data)

    if not internal_instance_tuple:
        print("Failed to convert parsed data to internal tuple format. Exiting test.")
        return False

    print("\nSuccessfully converted to internal tuple format:")
    # print(f"  Internal Tuple: {internal_instance_tuple}") # Can be very long
    print(f"  Depot XY (internal): {internal_instance_tuple[0]}")
    print(f"  Num Customer Coords (internal): {len(internal_instance_tuple[1])}")
    print(f"  Num Customer Demands (internal): {len(internal_instance_tuple[2])}")
    print(f"  Capacity (internal): {internal_instance_tuple[3]}")

    # 3. Data Preprocessing Considerations (Normalization)
    # For now, we assume models can handle raw scales or internal normalization exists.
    # If issues arise, normalization of coordinates (e.g., to [0,1]) would be added here.
    print("\nSkipping explicit normalization for now. Using raw coordinate scales.")

    # 4. Visualize the instance
    problem_type_from_file = parsed_data.get("problem_type", "CVRP")
    print(f"\nVisualizing instance using problem type: {problem_type_from_file}")

    try:
        # Create an instance dictionary suitable for print_vrp_instance_info and visualize_vrp_instance
        # These visualization functions expect a dictionary with keys like VRP_DATA_FORMAT
        # viz_instance_dict = {
        #     "depot_xy": internal_instance_tuple[0],
        #     "node_xy": internal_instance_tuple[1],
        #     "demand": internal_instance_tuple[2],
        #     "capacity": internal_instance_tuple[3]
        #     # Add other fields like 'name' if your visualizers use them
        # }
        # if parsed_data.get('name'):
        #      viz_instance_dict['name'] = parsed_data.get('name')

        print("--- Instance Info (from data_visualize) ---")
        print_vrp_instance_info(internal_instance_tuple, problem_type_from_file, lang='en')
        print("-------------------------------------------")

        fig, ax = plt.subplots(figsize=(10, 10))
        # visualize_vrp_instance also expects a tuple.
        visualize_vrp_instance(internal_instance_tuple, problem_type_from_file, lang='en', show_annotations=True, ax=ax)
        ax.set_title(f"Parsed Instance: {parsed_data.get('name', 'Unknown')} ({problem_type_from_file})")
        plt.show()
        print("Visualization displayed. Please check the plot.")
        print("Test PASSED (manual verification of plot needed).")
        return True
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        print("Test FAILED during visualization.")
        return False

if __name__ == "__main__":
    # --- !!! IMPORTANT: Adjust this path to your CVRP-LIB file !!! ---
    # Example: ../CVRP-LIB/X-n101-k25.vrp if cvrp_lib_integration is at the same level as CVRP-LIB
    # Or an absolute path.
    # For testing, assuming CVRP-LIB is one level up from the script's parent directory (project_root)
    test_file = os.path.join(project_root, "CVRP-LIB", "X-n101-k25.vrp") 
    
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found at calculated path: {test_file}")
        print("Please adjust the 'test_file' variable in test_single_instance_processing.py")
    else:
        test_single_cvrp_lib_instance(test_file) 