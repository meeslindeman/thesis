class RelationshipBuilder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nodes = []  # List of tuples (node_name, gender)
        self.edges = []  # List of tuples (source_node, target_node, relationship)
        self.read_file()

    def get_relationship_encoding(self):
        unique_relationships = set(edge[2] for edge in self.edges)
        return {rel: i for i, rel in enumerate(unique_relationships)}

    def read_file(self):
        with open(self.file_path, 'r') as file:
            num_nodes, num_edges = map(int, file.readline().split())

            # Read node information
            for _ in range(num_nodes):
                node_name, gender = file.readline().strip().split()
                self.nodes.append((node_name, gender))

            # Read edge information
            for _ in range(num_edges):
                source, target, relationship = file.readline().strip().split()
                self.edges.append((source, target, relationship))

# relationship_builder = RelationshipBuilder('/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families/arryn.txt')
# print("Nodes:", relationship_builder.nodes)
# print("Edges:", relationship_builder.edges)