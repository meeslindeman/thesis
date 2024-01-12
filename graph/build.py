import random
import torch
from torch_geometric.data import Data

class FamilyMember:
    """
    Represents a family member with gender, age, spouse, and children.
    """

    def __init__(self, gender, age):
        self.gender = gender
        self.age = age
        self.spouse = None
        self.children = []
        self.height = random.randint(150, 200) # in cm
        self.hair_color = random.choice(['black', 'brown', 'blonde', 'red'])

    def create_spouse(self):
        """
        Creates a spouse for the family member based on their gender and age.
        Returns the created spouse.
        """
        spouse_gender = 'f' if self.gender == 'm' else 'm'
        spouse_age = random.randint(max(18, self.age - 5), min(self.age + 5, 100))
        spouse = FamilyMember(spouse_gender, spouse_age)
        self.spouse = spouse
        spouse.spouse = self
        return spouse

    def create_children(self, max_children=4):
        """
        Creates children for the family member and their spouse.
        """
        children_count = 2 #random.randint(1, max_children)
        youngest_parent_age = min(self.age, self.spouse.age)
        for _ in range(children_count):
            child_gender = random.choice(['m', 'f'])
            max_child_age = max(0, youngest_parent_age - 20)
            min_child_age = max(0, youngest_parent_age - 30)
            child_age = random.randint(min_child_age, max_child_age)
            child = FamilyMember(child_gender, child_age)
            self.children.append(child)
            self.spouse.children.append(child)

def create_family_tree(generations):
    age_range = (80,100)
    root_age = random.randint(*age_range)
    root_member = FamilyMember('m', root_age)

    spouse = root_member.create_spouse()
    current_generation = [(root_member, spouse)]
    
    all_members = {0: root_member, 1: spouse}
    next_index = 2  # Start indexing from 2 as 0 and 1 are already used

    for gen in range(1, generations):
        next_generation = []
        for parent1, parent2 in current_generation:
            parent1.create_children()
            for child in parent1.children:
                all_members[next_index] = child
                next_index += 1
                if gen < generations - 1:
                    spouse = child.create_spouse()
                    all_members[next_index] = spouse
                    next_generation.append((child, spouse))
                    next_index += 1
        current_generation = next_generation

    return all_members

def create_data_object(all_members):
    # Convert genders to a binary representation and collect node features
    gender_to_binary = {'m': 0, 'f': 1}
    color_to_binary = {'black': 0, 'brown': 1, 'blonde': 2, 'red': 3}
    x = [[gender_to_binary[member.gender], member.age, color_to_binary[member.hair_color], member.height] for index, member in all_members.items()]

    # Prepare edge_index and edge_attr
    edge_index = []
    edge_attr = []

    for index, member in all_members.items():
        if member.spouse:
            spouse_index = list(all_members.keys())[list(all_members.values()).index(member.spouse)]
            # Add edges for spouses in both directions with the 'married' attribute
            edge_index.append([index, spouse_index])
            edge_index.append([spouse_index, index])
            edge_attr.extend([0, 0])  # 0 for 'married'

        for child in member.children:
            child_index = list(all_members.keys())[list(all_members.values()).index(child)]
            # Add edges from children to this member with the 'childOf' attribute
            edge_index.append([child_index, index])
            edge_attr.append(1)  # 1 for 'childOf'

    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create the data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data