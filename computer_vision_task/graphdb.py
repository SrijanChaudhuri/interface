import sqlite3
import json
from datetime import datetime

class GraphDB:
    def __init__(self, db_path="graph_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        # Table to record each processed graph (e.g., one per PDF page)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_number INTEGER,
                timestamp TEXT,
                xml_data TEXT  -- Optional: XML representation of the complete graph
            )
        ''')
        # Table for nodes containing only id, specs, and bounding_box.
        # Using a composite primary key (graph_id, id) to allow node IDs to be reused across graphs.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                graph_id INTEGER,
                id INTEGER,
                specs TEXT,
                bounding_box TEXT,  -- Stored as a JSON string (e.g., [x, y, x+w, y+h])
                PRIMARY KEY (graph_id, id),
                FOREIGN KEY (graph_id) REFERENCES graphs(id)
            )
        ''')
        # Table for edges representing connectivity between nodes in the graph.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id INTEGER,
                node1_id INTEGER,
                node2_id INTEGER,
                connection_type TEXT,  -- Optional: e.g., "piping", "signal", etc.
                FOREIGN KEY (graph_id) REFERENCES graphs(id)
            )
        ''')
        self.conn.commit()
    
    def store_graph_data(self, page_number, nodes, connections, xml_data=None):
        """
        Stores the topology graph data for a processed P&ID page.
        
        Args:
            page_number (int): The page number from the PDF.
            nodes (list of dict): Each node must contain:
                - "id": detection identifier.
                - "specs": additional specifications or OCR results.
                - "bounding_box": coordinates as a tuple/list (e.g., [x, y, x+w, y+h]).
            connections (list): Each connection is either a tuple (node1_id, node2_id) or a dict
                                containing keys "node1_id", "node2_id", and optionally "connection_type".
            xml_data (str, optional): XML representation of the complete graph (if available).
        
        Returns:
            int: The ID of the stored graph.
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        # Insert a new graph record.
        cursor.execute('''
            INSERT INTO graphs (page_number, timestamp, xml_data)
            VALUES (?, ?, ?)
        ''', (page_number, timestamp, xml_data))
        graph_id = cursor.lastrowid
        
        # Insert each node. The bounding box is stored as a JSON string.
        for node in nodes:
            bounding_box_json = json.dumps(node.get("bounding_box"))
            cursor.execute('''
                INSERT INTO nodes (graph_id, id, specs, bounding_box)
                VALUES (?, ?, ?, ?)
            ''', (graph_id,
                  node["id"],
                  node.get("specs", ""),
                  bounding_box_json))
        
        # Insert connections (edges).
        for conn in connections:
            if isinstance(conn, dict):
                node1 = conn.get("node1_id")
                node2 = conn.get("node2_id")
                connection_type = conn.get("connection_type", "")
            else:
                node1, node2 = conn
                connection_type = ""
            cursor.execute('''
                INSERT INTO connections (graph_id, node1_id, node2_id, connection_type)
                VALUES (?, ?, ?, ?)
            ''', (graph_id, node1, node2, connection_type))
        
        self.conn.commit()
        return graph_id

    def query_graph(self, graph_id):
        """
        Retrieves the stored graph data (metadata, nodes, and connections) for the given graph ID.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM graphs WHERE id = ?", (graph_id,))
        graph = cursor.fetchone()
        
        cursor.execute("SELECT * FROM nodes WHERE graph_id = ?", (graph_id,))
        nodes = cursor.fetchall()
        
        cursor.execute("SELECT * FROM connections WHERE graph_id = ?", (graph_id,))
        connections = cursor.fetchall()
        
        return {"graph": graph, "nodes": nodes, "connections": connections}
    
    def display_all_data(self):
        """
        Display all data stored in the database for debugging and verification.
        """
        cursor = self.conn.cursor()
        
        # Display all graphs
        print("\n===== GRAPHS =====")
        cursor.execute("SELECT * FROM graphs")
        graphs = cursor.fetchall()
        if graphs:
            for graph in graphs:
                print(f"Graph ID: {graph[0]}, Page: {graph[1]}, Timestamp: {graph[2]}")
        else:
            print("No graphs found in database.")
        
        # Display all nodes
        print("\n===== NODES =====")
        cursor.execute("SELECT * FROM nodes")
        nodes = cursor.fetchall()
        if nodes:
            print(f"Total nodes: {len(nodes)}")
            for node in nodes:
                graph_id, node_id, specs, bounding_box = node
                print(f"Graph: {graph_id}, Node ID: {node_id}")
                print(f"  Specs: {specs[:50]}{'...' if len(specs or '') > 50 else ''}")
                print(f"  Bounding Box: {bounding_box}")
        else:
            print("No nodes found in database.")
        
        # Display all connections
        print("\n===== CONNECTIONS =====")
        cursor.execute("SELECT * FROM connections")
        connections = cursor.fetchall()
        if connections:
            print(f"Total connections: {len(connections)}")
            for conn in connections:
                id, graph_id, node1_id, node2_id, conn_type = conn
                print(f"Connection ID: {id}, Graph: {graph_id}, Type: {conn_type or 'default'}")
                print(f"  {node1_id} -> {node2_id}")
        else:
            print("No connections found in database.")
        
        # Database summary
        print("\n===== DATABASE SUMMARY =====")
        cursor.execute("SELECT COUNT(*) FROM graphs")
        graph_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM connections")
        connection_count = cursor.fetchone()[0]
        
        print(f"Total Graphs: {graph_count}")
        print(f"Total Nodes: {node_count}")
        print(f"Total Connections: {connection_count}")
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # Simple test to verify the database functionality.
    db = GraphDB()
    db.display_all_data()
    print("Database tables created and ready.")
    db.close()
