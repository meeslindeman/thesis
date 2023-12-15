import random
import names
import torch
from torch_geometric.data import Data

class FamilyMember:
    def __init__(self, name, gender, age):
        self.name = name
        self.gender = gender
        self.age = age
        self.spouse = None
        self.children = []

def create_random_name(gender):
    return names.get_first_name(gender='male' if gender == 'm' else 'female')

def create_spouse(member, age_range):
    spouse_gender = 'f' if member.gender == 'm' else 'm'
    spouse_name = create_random_name(spouse_gender)
    
    # Ensure the age range for spouse selection is within the overall allowed range and within 10 years of the member's age
    min_age_limit = max(age_range[0], member.age - 10)
    max_age_limit = min(age_range[1], member.age + 10)
    min_age = max(min_age_limit, age_range[0])
    max_age = min(max_age_limit, age_range[1])

    # It's possible that the member's age is so low that the spouse's age would be below the legal age of 18, or that min_age > max_age
    # In this case, we adjust the spouse's age to be at least 18 or the minimum of the age range if higher
    if min_age > max_age or max_age < 18:
        min_age = max(age_range[0], 18)
        max_age = max(min_age, age_range[1])
    
    # If the age range is still invalid, we print an error message and return None
    if min_age > max_age:
        print(f"Cannot find a suitable age for a spouse for {member.name}.")
        return None
    
    spouse_age = random.randint(min_age, max_age)
    spouse = FamilyMember(spouse_name, spouse_gender, spouse_age)
    member.spouse = spouse
    spouse.spouse = member
    return spouse

def create_children(parent1, parent2, age_range):
    children_count = random.randint(1, 2)  # Assume each couple can have between 1 to 5 children
    for _ in range(children_count):
        child_gender = random.choice(['m', 'f'])
        child_name = create_random_name(child_gender)
        child_age = random.randint(*age_range)
        child = FamilyMember(child_name, child_gender, child_age)
        parent1.children.append(child)
        parent2.children.append(child)

def create_family_tree(generations):
    age_ranges = [(80, 100), (50, 70), (20, 40)]  # Define age ranges for the generations
    current_generation = []

    # Start with the root family members
    root_name = create_random_name('m')
    root_member = FamilyMember(root_name, 'm', random.randint(*age_ranges[0]))
    spouse = create_spouse(root_member, age_ranges[0])
    current_generation.append((root_member, spouse))

    all_members = {root_name: root_member, spouse.name: spouse}

    for gen in range(1, generations):
        next_generation = []
        for parent1, parent2 in current_generation:
            create_children(parent1, parent2, age_ranges[min(gen, len(age_ranges)-1)])
            for child in parent1.children:
                all_members[child.name] = child
                if gen < generations - 1:  # Add spouses and children if more generations are needed
                    spouse = create_spouse(child, age_ranges[min(gen+1, len(age_ranges)-1)])
                    all_members[spouse.name] = spouse
                    next_generation.append((child, spouse))
        current_generation = next_generation

    # Extract names from the family tree
    names_list = [member.name for member in all_members.values()]

    return all_members, names_list

def create_data_object(all_members):
    # Convert genders to a binary representation and collect node features
    gender_to_binary = {'m': 0, 'f': 1}
    x = [[gender_to_binary[member.gender], member.age] for name, member in all_members.items()]

    # Prepare edge_index and edge_attr
    edge_index = []
    edge_attr = []
    
    # Use a dictionary to map names to node indices
    name_to_index = {name: i for i, (name, member) in enumerate(all_members.items())}

    for name, member in all_members.items():
        if member.spouse:
            spouse_index = name_to_index[member.spouse.name]
            member_index = name_to_index[name]
            # Add edges for spouses in both directions with the 'married' attribute
            edge_index.append([member_index, spouse_index])
            edge_index.append([spouse_index, member_index])
            edge_attr.append(0)  # 0 for 'married'
            edge_attr.append(0)  # 0 for 'married'

        for child in member.children:
            child_index = name_to_index[child.name]
            member_index = name_to_index[name]
            # Add edges from children to this member with the 'childOf' attribute
            edge_index.append([child_index, member_index])
            edge_attr.append(1)  # 1 for 'childOf'

    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    # Create the data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data