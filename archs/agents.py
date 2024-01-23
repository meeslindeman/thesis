import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.network import GAT, Transform

class SenderDual(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size, temperature):
        super(SenderDual, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature

        self.transform = Transform(self.num_node_features, embedding_size, heads)
        self.gat = GAT(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear(embedding_size, hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        batch_ptr = data.ptr
        target_node_idx = data.target_node_idx

        h_t = self.transform(data)
        h_g = self.gat(data)
        h = h_t + h_g

        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]

        target_embedding = h[adjusted_target_node_idx]       

        output = self.fc(target_embedding)                           

        return output.view(-1, self.hidden_size)

class ReceiverDual(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size):
        super(ReceiverDual, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size

        self.transform = Transform(self.num_node_features, embedding_size, heads)
        self.gat = GAT(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input

        h_t = self.transform(data)
        h_g = self.gat(data)
        h = h_t + h_g   

        # Reshape h for batched operation
        batch_size = data.num_graphs  # Assuming this attribute is available
        num_nodes_per_graph = data.num_nodes // batch_size  # Assuming equal number of nodes in each graph
        h = h.view(batch_size, num_nodes_per_graph, -1)

        message_embedding = self.fc(message)  
        message_embedding = message_embedding.view(batch_size, -1, 1)

        dot_products = torch.bmm(h, message_embedding).squeeze(-1)   

        probabilities = F.log_softmax(dot_products, dim=1)                      

        return probabilities
    
#===================================================================================================

class SenderGAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size, temperature):
        super(SenderGAT, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature

        self.gat = GAT(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear(embedding_size, hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        batch_ptr = data.ptr
        target_node_idx = data.target_node_idx

        h = self.gat(data)

        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]

        target_embedding = h[adjusted_target_node_idx]       

        output = self.fc(target_embedding)                           

        return output.view(-1, self.hidden_size)

class ReceiverGAT(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size):
        super(ReceiverGAT, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size

        self.gat = GAT(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input

        h = self.gat(data)

        # Reshape h for batched operation
        batch_size = data.num_graphs  # Assuming this attribute is available
        num_nodes_per_graph = data.num_nodes // batch_size  # Assuming equal number of nodes in each graph
        h = h.view(batch_size, num_nodes_per_graph, -1)

        message_embedding = self.fc(message)  
        message_embedding = message_embedding.view(batch_size, -1, 1)

        dot_products = torch.bmm(h, message_embedding).squeeze(-1)   

        probabilities = F.log_softmax(dot_products, dim=1)                      

        return probabilities
    
#===================================================================================================

class SenderTransform(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size, temperature):
        super(SenderTransform, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature
          
        self.transform = Transform(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear(embedding_size, hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        batch_ptr = data.ptr
        target_node_idx = data.target_node_idx

        h = self.transform(data)

        adjusted_target_node_idx = target_node_idx + batch_ptr[:-1]

        target_embedding = h[adjusted_target_node_idx]       

        output = self.fc(target_embedding)                           

        return output.view(-1, self.hidden_size)

class ReceiverTransform(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size):
        super(ReceiverTransform, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        
        self.transform = Transform(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input

        h = self.transform(data)

        # Reshape h for batched operation
        batch_size = data.num_graphs  # Assuming this attribute is available
        num_nodes_per_graph = data.num_nodes // batch_size  # Assuming equal number of nodes in each graph
        h = h.view(batch_size, num_nodes_per_graph, -1)

        message_embedding = self.fc(message)  
        message_embedding = message_embedding.view(batch_size, -1, 1)

        dot_products = torch.bmm(h, message_embedding).squeeze(-1)   

        probabilities = F.log_softmax(dot_products, dim=1)                      

        return probabilities
    
#===================================================================================================
    
class SenderRel(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size, temperature):
        super(SenderRel, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        self.hidden_size = hidden_size
        self.temp = temperature
          
        self.transform = Transform(self.num_node_features, embedding_size, heads) 
        self.fc = nn.Linear((embedding_size * 2), hidden_size) 

    def forward(self, x, _aux_input):
        data = _aux_input

        target_node_idx, root_idx = data.target_node_idx, data.root_idx

        h = self.transform(data)

        target = h[target_node_idx].squeeze()
        root = h[root_idx].squeeze()

        target_embedding = torch.cat((target, root))    

        output = self.fc(target_embedding)                           

        return output.view(-1, self.hidden_size)

class ReceiverRel(nn.Module):
    def __init__(self, num_node_features, embedding_size, heads, hidden_size):
        super(ReceiverRel, self).__init__()
        self.num_node_features = num_node_features
        self.heads = heads
        
        self.transform = Transform(self.num_node_features, embedding_size, heads)
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, message, _input, _aux_input):
        data = _aux_input

        h = self.transform(data)   

        message_embedding = self.fc(message)                 

        dot_products = torch.matmul(h, message_embedding.t()).t()   

        probabilities = F.log_softmax(dot_products, dim=1)                      

        return probabilities