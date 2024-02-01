import torch
import torch.nn.functional as F

def process_graph_to_sequence(graph, padding_len: int):
    # Vocabulary
    vocab = {'(': 1, ')': 2, 'male': 3, 'female': 4, 'married-to': 5, 'child-of': 6, 'gave-birth-to': 7}

    # Function to get node information
    def get_node_info(graph):
        node_info = {}
        relationship_types = {tuple([1., 0., 0.]): 'married-to', tuple([0., 1., 0.]): 'child-of', tuple([0., 0., 1.]): 'gave-birth-to'}

        for index, features_tensor in enumerate(graph.x):
            features = features_tensor.tolist()
            gender = 'male' if features[0] == 0 else 'female'
            features_dict = {'gender': gender, 'features': features}

            relationships = []
            for i in range(graph.edge_index.shape[1]):
                if graph.edge_index[0, i] == index:
                    target_node = graph.edge_index[1, i]
                    rel_type = relationship_types[tuple(graph.edge_attr[i].tolist())]
                    relationships.append({'node': target_node.item(), 'relationship': rel_type})

            features_dict['relationships'] = relationships
            node_info[index] = features_dict

        return node_info

    # Function to sort tree
    def sort_tree(node_data, start_node):
        visited = {start_node}

        def get_sorted_children(node_index):
            relationships = [r for r in node_data[node_index]['relationships'] if r['node'] not in visited]
            married_child = [child['node'] for child in relationships if child['relationship'] == 'married-to']
            other_children = sorted([child['node'] for child in relationships if child['node'] not in married_child], 
                                    key=lambda x: node_data[x]['gender'] == 'male')
            visited.update(married_child + other_children)
            return married_child + other_children

        def build_tree(node_index):
            children_indices = get_sorted_children(node_index)
            children = [build_tree(child_index) for child_index in children_indices]
            return {"index": node_index, "children": children}

        return build_tree(start_node)

    # Function to build sequence
    def build_sequence(tree, node_data, vocab):
        def node_sequence(node):
            sequence = [vocab[node_data[node['index']]['gender']]]
            for child in node['children']:
                relationship = next(r['relationship'] for r in node_data[node['index']]['relationships'] if r['node'] == child['index'])
                sequence.append(vocab[relationship])

                if child['children']:
                    sequence.append(vocab['('])  # Opening parenthesis
                    sequence.extend(node_sequence(child))
                    sequence.append(vocab[')'])  # Closing parenthesis
                else:
                    sequence.extend(node_sequence(child))

            return sequence

        sequence = node_sequence(tree)
        return torch.tensor([sequence], dtype=torch.float32)

    # Process graph to sequence
    node_info = get_node_info(graph)
    tree = sort_tree(node_info, graph.target_node_idx)
    sequence_tensor = build_sequence(tree, node_info, vocab)

    # Because otherwise the sequence length is not constant and batching is not working, for now I set max_len to 9 since that is the largest sequence but I would need some  way of determining this automatically for larger graphs
    sequence_tensor = F.pad(sequence_tensor, (0, max(padding_len - sequence_tensor.shape[1], 0)), 'constant', 0) 

    return sequence_tensor