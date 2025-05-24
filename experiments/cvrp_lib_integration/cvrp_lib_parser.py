import re

def parse_vrp_file(file_path):
    """
    Parses a VRP file in CVRP-LIB format.

    Args:
        file_path (str): The path to the .vrp file.

    Returns:
        dict or None: A dictionary containing the parsed VRP instance data
                      (depot_xy, node_xy, demand, capacity, problem_type, dimension, name),
                      or None if parsing fails.
                      'depot_xy': [[x, y]]
                      'node_xy': [[x1, y1], [x2, y2], ...] for customers
                      'demand': [d1, d2, ...] for customers
                      'capacity': float
    """
    instance_data = {
        "name": None,
        "problem_type": None,
        "dimension": None,
        "capacity": None,
        "depot_coord": None, # Temporary storage for the first node encountered (depot)
        "node_coords": [], # List of [id, x, y]
        "demands": [], # List of [id, demand_value]
        # Final structured data
        "depot_xy": None,
        "node_xy": [],
        "demand": []
    }
    
    reading_coords = False
    reading_demands = False

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("COMMENT"):
                    continue

                if ":" in line:
                    key, value = [part.strip() for part in line.split(":", 1)]
                    if key == "NAME":
                        instance_data["name"] = value
                    elif key == "TYPE":
                        instance_data["problem_type"] = value
                        if instance_data["problem_type"] != "CVRP":
                            print(f"Warning: This parser is primarily for CVRP. Found type: {instance_data['problem_type']}")
                    elif key == "DIMENSION":
                        instance_data["dimension"] = int(value)
                    elif key == "CAPACITY":
                        instance_data["capacity"] = float(value)
                
                if line == "NODE_COORD_SECTION":
                    reading_coords = True
                    reading_demands = False
                    continue
                elif line == "DEMAND_SECTION":
                    reading_coords = False
                    reading_demands = True
                    continue
                elif line == "DEPOT_SECTION":
                    reading_coords = False
                    reading_demands = False
                    continue # Depot ID handled by matching demand 0 or being the first node with demand 0
                elif line == "EOF":
                    break

                if reading_coords:
                    parts = re.split(r'\s+', line) # Split by any whitespace
                    if len(parts) == 3:
                        node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                        instance_data["node_coords"].append([node_id, x, y])
                    else:
                        print(f"Warning: Skipping malformed NODE_COORD_SECTION line: {line}")


                elif reading_demands:
                    parts = re.split(r'\s+', line)
                    if len(parts) == 2:
                        node_id, demand_val = int(parts[0]), float(parts[1])
                        instance_data["demands"].append([node_id, demand_val])
                    else:
                        print(f"Warning: Skipping malformed DEMAND_SECTION line: {line}")
        
        # --- Post-processing and structuring ---
        if not instance_data["node_coords"] or not instance_data["demands"]:
            print("Error: NODE_COORD_SECTION or DEMAND_SECTION missing or empty.")
            return None
        if instance_data["dimension"] is None: # Fallback if DIMENSION not specified
             instance_data["dimension"] = len(instance_data["node_coords"])


        # Sort by node_id to ensure correct order
        instance_data["node_coords"].sort(key=lambda item: item[0])
        instance_data["demands"].sort(key=lambda item: item[0])

        if len(instance_data["node_coords"]) != instance_data["dimension"] or \
           len(instance_data["demands"]) != instance_data["dimension"]:
            print(f"Warning: Dimension mismatch. Expected {instance_data['dimension']}, "
                  f"got {len(instance_data['node_coords'])} coords, {len(instance_data['demands'])} demands.")
            # Attempt to proceed if critical data (like depot) can be found

        # Find depot (typically node 1 with demand 0, or just node 1 if all demands are positive)
        # Some CVRP-LIB files list depot demand as 0, some don't list it explicitly but it's node 1.
        depot_node_id_from_coords = instance_data["node_coords"][0][0] # Assume first listed coord is depot if all else fails

        depot_id_from_demand = None
        for nid, dem_val in instance_data["demands"]:
            if dem_val == 0:
                depot_id_from_demand = nid
                break
        
        depot_node_id = None
        if depot_id_from_demand is not None:
            depot_node_id = depot_id_from_demand
        elif instance_data["node_coords"] and instance_data["node_coords"][0][0] == 1: # Default to node 1
            # Check if node 1's demand (if present) is 0. If node 1 demand is not 0, this is tricky.
            node1_demand_entry = next((d for id_d, d in instance_data["demands"] if id_d == 1), None)
            if node1_demand_entry == 0.0 or node1_demand_entry is None: # None means demand for node 1 might not be listed if 0
                depot_node_id = 1
            else: # Node 1 has non-zero demand, this is unusual for standard CVRP depot
                print(f"Warning: Node 1 has non-zero demand ({node1_demand_entry}). This might not be the depot.")
                # Fallback to the first node in coordinate list if its demand is 0
                first_coord_node_id = instance_data["node_coords"][0][0]
                first_coord_node_demand = next((d for id_d, d in instance_data["demands"] if id_d == first_coord_node_id), None)
                if first_coord_node_demand == 0.0:
                    depot_node_id = first_coord_node_id
                else:
                    print(f"Error: Cannot reliably identify depot. Node 1 demand is {node1_demand_entry}, first coord node {first_coord_node_id} demand is {first_coord_node_demand}.")
                    return None

        else: # Could not find a demand=0 node, and node 1 is not first in coords or has demand
            print(f"Error: Could not determine depot. No node with demand 0, and node 1 assumptions failed.")
            return None

        # Extract depot coordinates
        depot_coord_entry = next((coord for coord in instance_data["node_coords"] if coord[0] == depot_node_id), None)
        if not depot_coord_entry:
            print(f"Error: Depot node {depot_node_id} coordinates not found.")
            return None
        instance_data["depot_xy"] = [[depot_coord_entry[1], depot_coord_entry[2]]]

        # Extract customer nodes and their demands
        for node_id_coord, x, y in instance_data["node_coords"]:
            if node_id_coord == depot_node_id:
                continue
            
            demand_entry = next((dem for node_id_dem, dem in instance_data["demands"] if node_id_dem == node_id_coord), None)
            if demand_entry is None:
                print(f"Warning: Demand not found for customer node {node_id_coord}. Skipping.")
                continue
            if demand_entry == 0:
                 print(f"Warning: Customer node {node_id_coord} has 0 demand. Including anyway.")

            instance_data["node_xy"].append([x, y])
            instance_data["demand"].append(demand_entry)

        if not instance_data["node_xy"]:
            print("Error: No customer nodes found after processing.")
            return None
        if instance_data["problem_type"] is None: instance_data["problem_type"] = "CVRP" # Default if not in file
        if instance_data["capacity"] is None:
            print("Error: Capacity not found in VRP file.")
            return None
            
        # Final check on dimension vs actual customer nodes
        # Dimension in .vrp usually includes depot.
        if instance_data["dimension"] != len(instance_data["node_xy"]) + 1:
             print(f"Warning: Final customer node count ({len(instance_data['node_xy'])}) + 1 (depot) "
                   f"does not match VRP file DIMENSION ({instance_data['dimension']}). Using actual count.")
             instance_data["dimension"] = len(instance_data["node_xy"]) + 1


        return {
            "name": instance_data["name"],
            "problem_type": instance_data["problem_type"],
            "dimension": instance_data["dimension"], # Total nodes including depot
            "capacity": instance_data["capacity"],
            "depot_xy": instance_data["depot_xy"],
            "node_xy": instance_data["node_xy"], # Customer coordinates
            "demand": instance_data["demand"]    # Customer demands
        }

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def to_internal_tuple(parsed_data):
    """
    Converts parsed VRP data dictionary to the internal tuple format.
    (depot_xy, node_xy, demand, capacity)
    """
    if not parsed_data:
        return None
    try:
        # depot_xy is already [[x,y]]
        # node_xy is already [[x1,y1], ...]
        # demand is already [d1, d2, ...]
        # capacity is a float
        # For CVRP: (depot_xy, node_xy, demand, capacity)
        instance_as_tuple = (
            parsed_data["depot_xy"], 
            parsed_data["node_xy"], 
            parsed_data["demand"], 
            parsed_data["capacity"]
        )
        problem_type = parsed_data.get("problem_type", "CVRP") # Default to CVRP

        # Return only the instance tuple and the problem type string
        return instance_as_tuple, problem_type
    except KeyError as e:
        print(f"Error converting to internal tuple: Missing key {e}")
        return None

if __name__ == '__main__':
    # Test function for the parser
    def test_parser():
        print("Testing CVRP-LIB Parser...")
        # Create a dummy .vrp file for testing
        dummy_file_content = """
NAME : DummyTest1
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 10 10
2 20 20
3 30 30
4 40 40
DEMAND_SECTION
1 0
2 10
3 15
4 20
DEPOT_SECTION
1
-1
EOF
        """
        dummy_file_path = "dummy_test.vrp"
        with open(dummy_file_path, 'w') as f:
            f.write(dummy_file_content)

        parsed_data = parse_vrp_file(dummy_file_path)
        if parsed_data:
            print("Parsed Data:")
            for key, value in parsed_data.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (len: {len(value)})")
                else:
                    print(f"  {key}: {value}")
            
            internal_tuple = to_internal_tuple(parsed_data)
            if internal_tuple:
                print("\nInternal Tuple Format:")
                print(f"  Depot XY: {internal_tuple[0]}")
                print(f"  Node XY (customers): {internal_tuple[1][:3]}... (len: {len(internal_tuple[1])})")
                print(f"  Demand (customers): {internal_tuple[2][:3]}... (len: {len(internal_tuple[2])})")
                print(f"  Capacity: {internal_tuple[3]}")
                
                # Basic checks
                expected_dimension = 4
                expected_customers = 3
                if parsed_data['dimension'] == expected_dimension and \
                   len(parsed_data['node_xy']) == expected_customers and \
                   parsed_data['depot_xy'] == [[10,10]] and \
                   parsed_data['node_xy'][0] == [20,20] and parsed_data['demand'][0] == 10 and \
                   parsed_data['capacity'] == 100:
                   print("\nTest PASSED for dummy_test.vrp")
                else:
                   print("\nTest FAILED for dummy_test.vrp")
                   print(f"Expected dimension {expected_dimension}, got {parsed_data['dimension']}")
                   print(f"Expected {expected_customers} customers, got {len(parsed_data['node_xy'])}")

            else:
                print("\nFailed to convert to internal tuple.")
                print("\nTest FAILED for dummy_test.vrp conversion")

        else:
            print("Failed to parse dummy_test.vrp")
            print("\nTest FAILED for dummy_test.vrp parsing")

        # Test with a real file if available (provide path)
        # real_file_path = "../CVRP-LIB/X-n101-k25.vrp" # Adjust path as needed
        # print(f"\n--- Testing with real file: {real_file_path} ---")
        # if os.path.exists(real_file_path):
        #     parsed_real_data = parse_vrp_file(real_file_path)
        #     if parsed_real_data:
        #         print("Parsed Real Data (first 5 items for lists):")
        #         for key, value in parsed_real_data.items():
        #             if isinstance(value, list) and len(value) > 5:
        #                 print(f"  {key}: {value[:5]}... (len: {len(value)})")
        #             else:
        #                 print(f"  {key}: {value}")
        #         internal_real_tuple = to_internal_tuple(parsed_real_data)
        #         if internal_real_tuple:
        #             print("\nInternal Tuple Format (Real Data):")
        #             print(f"  Depot XY: {internal_real_tuple[0]}")
        #             print(f"  Node XY (customers): {internal_real_tuple[1][:3]}... (len: {len(internal_real_tuple[1])})")
        #             print(f"  Demand (customers): {internal_real_tuple[2][:3]}... (len: {len(internal_real_tuple[2])})")
        #             print(f"  Capacity: {internal_real_tuple[3]}")
        #             print("\nTest PASSED for real file parsing and conversion (structure check).")
        #         else:
        #             print("\nFailed to convert real data to internal tuple.")
        #             print("\nTest FAILED for real file conversion.")
        #     else:
        #         print(f"Failed to parse {real_file_path}")
        #         print("\nTest FAILED for real file parsing.")
        # else:
        #     print(f"Skipping real file test: {real_file_path} not found.")
        
        # Cleanup dummy file
        import os
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

    test_parser() 