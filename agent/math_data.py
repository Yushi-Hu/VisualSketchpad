import json
import os
from openai import OpenAI
import time
import re
from ipdb import set_trace as bp
import os
import numpy as np
from queue import Queue
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import math
import shutil
import json
# import array_to_latex as a2l
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow

# api_key = os.environ.get("OPENAI_API_KEY")
ISO_INSTUCTION = """In the query example, the nodes are zero-indexed.
Respond with 'true' or 'false' on whether the two graphs are isomorphic to each other."""
# If they are isomorphic, first provide the bijection between the two graphs, and then explain your reasoning.
# If they are not isomorphic, explain why in details.


ISO_DEFINITION = f"""Definition of graph isomorphism:
        In graph theory, an isomorphism of graphs G and H is a bijection f between the vertex sets of G and H, denoted as f: V(G) -> V(H). G and H are said to be isomorphic when f satisfies the following: any two vertices u and v of G are adjacent in G if and only if f(u) and f(v) are adjacent in H. This kind of bijection is commonly described as "edge-preserving bijection", in accordance with the general notion of isomorphism being a structure-preserving bijection.
"""

MAXFLOW_DEFINITION = f"""Definition of Maxflow problem:
        In the max flow problem, we have a directed graph with a source node s and a sink node t, 
        and each edge has a capacity (integer valued, colored in green) that represents the maximum amount of flow that can be sent through it. 
        The goal is to find the maximum amount of flow that can be sent from s to t, while respecting the capacity constraints on the edges."""
MAXFLOW_INSTRUCTION = f"""In the query example, the nodes are zero-indexed."""
# Output the maximum flow from the source node to the sink node. 
# Please provide your calculations and then explain your reasoning.

# Answer:"""

def prepare_isomorphism_math_prompt(adjacency_left, adjacency_right):
    text_query = f"""You are given two adjacency matrices of graphs G and H.

    YOUR TASK is to determine whether the two graphs are isomorphic to each other.
    """
    text_query += ISO_DEFINITION
    text_query += f"""
    In the query example, the adjacency matrices are zero-indexed. 

    Adjacency Matrix of Graph G:
    {adjacency_left}

    Adjacency Matrix of Graph H:
    {adjacency_right}
    """
    text_query += ISO_INSTUCTION

    return text_query


def prepare_maxflow_math_prompt(adjacency_matrix, source, sink):
    text_query = f"""You are given an adjacency matrix of a graph and two query nodes.  (one source node and one sink node). The source node is the node where the flow starts and the sink node is the node where the flow ends.

    YOUR TASK is to solve the maxflow problem given the weighted directed graph. """

    text_query += MAXFLOW_DEFINITION
    text_query += f"""
    Query Example:
    adjacency matrix: {adjacency_matrix}
    Source node (zero-indexed): {source}
    Sink node (zero-indexed): {sink}
"""
    text_query += MAXFLOW_INSTRUCTION

    return text_query

TASK2PROMPT = {
    "math_breakpoint": "You are given a real-valued, scalar function f(x). YOUR TASK is to count the number of breakpoints in the plot. A breakpoint refers to a point on the function's domain at which the function's domain at which the funciton changes its slope. Here is the expression of f(x): {}\nRespond with the number of breakpoints (in Arab digits) first on how many breakpoints the function f(x) contains based on the definition and your observation of the function. You should IGNORE the left and right end point of the domain, i.e. if the function is defined on [a, b], you should only consider the domain (a, b).", 
    "math_convexity": "You are given a real-valued, scalar function f(x). YOUR TASK is to determine whether f(x) is an convex function or concave function.\nDefinition of a convex function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≤ t f (x) + (1 − t) f (y)\nDefinition of a concave function: A function such that for all x, y, and 0 <= t <= 1\nf (tx + (1 − t)y) ≥ t f (x) + (1 − t) f (y)\nHere is the expression of f(x), defined for all x>0. Here is the expression of f(x): {}\nRespond with 'convex' or 'concave' first on whether the function f (x) is convex or concave, based on the definitions and your observation of the function.",
    "math_parity": "You are given a real-valued, scalar function f (x). YOUR TASK is to determine whether f (x) is an even function, an odd function, or neither.\nDefinition of an odd function: A function such that\nf (−x) = − f (x)\nwhere the sign is reversed but the absolute value remains the same if the sign of the independent variable is reversed.\nA function is neither even nor odd if it does not satisfy either condition. Here is the expression of f(x): {}\nRespond with 'even', 'odd', 'neither' first on whether the function f(x) is even, odd, or neither, based on the definitions and your observation of the function.",
    "graph_connectivity": "You are given an adjacency matrix of a graph and two query nodes. \n\nYOUR TASK is to find if there is a path between the two nodes.\n\n    Definition of connectivity:\n        In an undirected graph G, two vertices u and v are called connected if G contains a path from u to v.\n        A path in a graph is a finite sequence of edges which joins a sequence of vertices.\nIn the query example, the nodes and the adjacency matrix are zero-indexed. \n\nQuery Example:\nAdjacency Matrix: {}\n    Query nodes indices (zero-indexed): {} and {}\nRespond with 'yes' or 'no' on whether the query nodes are connected or not in the graph.\nIf there is a path, provide the path as a sequence of vertices (nodes), and then explain your reasoning. \nIf there is no path, explain why in details.",
    "graph_maxflow": prepare_maxflow_math_prompt,
    "graph_isomorphism": prepare_isomorphism_math_prompt,
    "winner_id": '''Given the following fen of the chess game: {}\nDetermine the game's outcome. Who won: White or Black? \nAnswer can be 'white' or 'black' or 'draw'
    '''
}

chess_plot = '''
```python
    import chess
    import chess.svg
    from cairosvg import svg2png
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    board = chess.Board()

    # Moves from the given PGN
    moves = [
        "d4", "d5", "e3", "e6", "Bd3", "Nf6", "Nd2", "Be7", "c3", "O-O",
        "f4", "Nbd7", "Qe2", "c5", "Ngf3", "c4", "Bc2", "a6", "O-O", "b5",
        "Ne5", "Bb7", "a3", "Rb8", "e4", "dxe4", "Nxe4", "Nxe5", "fxe5", "Nd5",
        "Qg4", "a5", "Bh6", "f6", "Qxg7#"
    ]

    # Apply each move to the board
    for move in moves:
        board.push_san(move)

    # Generate an SVG image of the final board position
    svg_data = chess.svg.board(board)
    png_bytes = svg2png(bytestring=svg_data)
    image = Image.open(io.BytesIO(png_bytes))
    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
```
'''



# def compute_acc_single(ex, prediction, subtask):
#     # code = ex["code"]
#     # question = TASK2PROMPT[subtask].format(code)
#     solution = ex["label"]
#     # Use regex to find text between 'ANSWER:' and 'TERMINATE'
#     # result = re.search(r'FINAL ANSWER:\s*(.*?)\s*TERMINATE', prediction)
#     # bp()
#     predict_ans = prediction.split("FINAL ANSWER:")[1].split("TERMINATE")[0]
#     predict_ans = f"FINAL ANSWER: {predict_ans}"
#     # bp()
#     answer_correctness = match_multiple_choice(predict_ans, solution)

#     # print(answer_correctness)
#     # bp()
#     acc = 1 if answer_correctness == "yes" or answer_correctness == "Yes" or answer_correctness == "YES" else 0
#     return acc, solution, predict_ans


