# Language Emergence in Graph Neural Networks

**Bachelor Thesis Project • 2024**

Exploring the emergence of compositional language in multi-agent systems using Graph Neural Networks to model structured input and analyze communication strategies in referential games.

## Abstract

This thesis explores the impact of structured input on the development of communication strategies in artificial agents, aiming to foster human-like language through compositional generalization. Utilizing graph-structured environments, the study investigates how agents evolve language within a setup that simulates social interactions and relations. 

The findings reveal that agents achieve high task-solving accuracy and exhibit a positive correlation between message entropy and accuracy, indicating a propensity for complex language strategies in larger, more challenging environments. However, this complexity does not necessarily align with the efficiency of human language, as evidenced by higher entropy levels diverging from the minimization observed in human language. 

While smaller game sizes prompt more compositional language structures, complexity in tasks does not invariably lead to increased compositional language, challenging the assumption of its necessity for effective communication. The study underscores the effectiveness of graph-structured inputs in promoting topographic similarity, suggesting potential avenues for research into more human-like language structures in multi-agent systems.

## Research Focus

This project investigates how artificial agents develop communication strategies to describe **family relations** using Graph Neural Networks, analyzing:

- **Compositional Language Emergence**: How agents develop structured communication patterns
- **Multi-Agent Communication**: Referential games between artificial agents
- **Graph-Structured Input**: Using GNNs to model relational data and social interactions
- **Language Efficiency vs. Complexity**: Comparing agent communication strategies to human language patterns

## Getting Started

### 1. Dataset Initialization

Initialize the dataset or use pre-generated data:

```bash
python init.py
```

*Note: Set the number of children in the graph by modifying `graph/build.py`*

### 2. Configure Hyperparameters

Set general hyperparameters for experiments in:

```
options.py
```

### 3. Run Experiments

#### Multiple Experiments

Run a series of experiments with predefined configurations:

```bash
python main.py
```

Configure multiple experiment setups by modifying the `multiple_options` list:

```python
multiple_options = [
    Options(agents='dual', generations=2),
    Options(agents='dual', generations=3),
    Options(agents='dual', generations=4)
]
```

#### Single Experiment

Run a single experiment using the configuration from `options.py`:

```bash
python main.py --single
```

## Key Features

- **Graph Neural Networks**: Leverages GNNs to process structured relational data
- **Multi-Agent Framework**: Implements referential games between communicating agents
- **Compositional Analysis**: Measures and analyzes the compositional properties of emergent languages
- **Configurable Experiments**: Flexible parameter configuration for various experimental setups
- **Family Relations Dataset**: Specialized dataset focusing on kinship and social relationships

## Project Structure

```
├── init.py                 # Dataset initialization
├── main.py                 # Main experiment runner
├── options.py              # Hyperparameter configuration
├── graph/
│   └── build.py            # Graph construction utilities
└── data/                   # Generated datasets (after running init.py)
```

## Results Summary

- **High Task Accuracy**: Agents successfully solve referential tasks with high accuracy
- **Entropy-Accuracy Correlation**: Positive correlation between message complexity and task performance
- **Graph Structure Benefits**: Graph-structured inputs promote topographic similarity in learned representations
- **Compositional Patterns**: Smaller environments encourage more compositional language structures
- **Human-AI Language Gap**: Agent languages show higher entropy than efficient human communication patterns

## Technologies Used

- **Python**: Primary programming language
- **Graph Neural Networks**: For processing structured relational input
- **Multi-Agent Systems**: Framework for agent communication and learning

---

*Final Grade: 8.0/10*


