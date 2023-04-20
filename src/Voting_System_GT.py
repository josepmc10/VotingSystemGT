import numpy as np
import matplotlib.pyplot as plt
import igraph
import cvxpy as cp
import random

class VotingSystemGT:
    def __init__(self,text_file):
        """
        It needs the name of the text file
        where the ballots are
        Expexted text file:
        option_ranked1,option_ranked2...
        option_ranked1,option_ranked2...
        option_ranked1,option_ranked2...
                    ...
        """
        self.ballots = []
        #Read all lines of the text file where each line is a ballot
        with open(text_file) as f:
            lines = f.readlines()
        self.total_voters = 0
        for line in lines:
            self.total_voters += 1
            self.ballots.append(line.replace("\n","").split(","))
        #The next atributes are for checking that the other atributes are created
        self.preferences_exist = False
        self.matrix_pref_exist = False
        self.margin_mat_exist = False
        self.computed_prob = False

    def counting_preference(self):
        """
        This function count how much people prefers option x1 rather than x2,...xn for all x
        A for each pair of options that appears in the ballots
        """
        self.preferences = dict()
        #Loop through ballots
        for ballot in self.ballots:
            #Loop through options in ballots
            for i,option in enumerate(ballot):
                #Loop through the next options in the ballot with less preference
                for index in range(i+1,len(ballot)):
                    pair_candidates = option + ballot[index]
                    if pair_candidates not in self.preferences:
                        self.preferences[pair_candidates] = 1
                    else:
                        self.preferences[pair_candidates] += 1
        self.preferences_exist = True
        return
    
    def build_matrix_preference(self):
        """
        This function builds the matrix which each row represents
        an option candidate and each column a candidate too. Ex: row A and col B represents
        number of voters that prefer A instead of B option (the principal diagonal is all 0)
        N = Matrix Preference
        """
        assert self.preferences_exist, "Preference Dictionary not Built"
        self.N = []
        self.order_options = list(set(self.ballots[0]))
        self.order_options.sort()
        for i,option1 in enumerate(self.order_options):
            row = []
            for j,option2 in enumerate(self.order_options):
                #There are some pair of options that are not in the dictionary
                #like one option with it self (the principal diagonal is all 0)
                if option1+option2 in self.preferences:
                    row.append(self.preferences[option1+option2])
                else:
                    row.append(0)
            self.N.append(row)
        self.N = np.array(self.N)
        self.matrix_pref_exist = True
        return
    
    def build_margin_matrix(self):
        """
        This function builds the margin matrix M from the matrix preference N
        where M(x,y) = -M(y,x) and M(x,y) = N(x,y) - N(y,x)
        The matrix M and N are square matrix.
        M = Margin Matrix
        """
        assert self.matrix_pref_exist, "Matrix Preference N not Built"
        self.M = []
        for x in range(self.N.shape[0]):
            row = []
            for y in range(self.N.shape[1]):
                # M(x,y) = N(x,y) - N(y,x)
                row.append(self.N[x][y] - self.N[y][x])
            self.M.append(row)
        self.M = np.array(self.M)
        self.margin_mat_exist = True
    
    def election_description(self):
        """
        It shows a description of how much people prefered one option than other
        """
        assert self.margin_mat_exist, "Margin Matrix M not Built"
        #In edges and weights there is the information we want to show
        edges,weights = self.edges_graph()
        print("Total voters:",self.total_voters)
        for i,edge in enumerate(edges):
            print(weights[i],"Voters prefers option",self.order_options[edge[0]]
                  ,"than option",self.order_options[edge[1]])
        
    def edges_graph(self):
        """
        This function build return the edges needed with the structure of the
        Margin Matrix
        """
        edges = []
        weights = []
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i][j] > 0:
                    edges.append((i,j))
                    weights.append(self.M[i][j])
        return edges,weights
    
    def construct_graph(self):
        """
        This function builds the graph from the margin matrix 
        For more graphic understanding.
        """
        assert self.margin_mat_exist, "Margin Matrix M not Built"
        
        self.g = igraph.Graph(directed=True)
        # Add 5 vertices
        self.g.add_vertices(len(self.order_options))
        # Add ids and labels to vertices
        for i,vert in enumerate(self.order_options):
            self.g.vs[i]["id"]= i
            self.g.vs[i]["label"]= vert
        edges,weights = self.edges_graph()
        # Add edges
        print("Edges",edges)
        print("Weights",weights)
        self.g.add_edges(edges)
        # Add weights and edge labels
        self.g.es['weight'] = weights
        self.g.es['label'] = weights
        
        return
    
    def plot_graph(self):
        """
        Plot the constructed graph
        """
        #HOW TO ADD EDGE LABELS (SOLVE)
        layout = self.g.layout("kamada_kawai")
        fig, ax = plt.subplots()
        igraph.plot(self.g, target=ax,layout=layout,vertex_size=25
                    ,vertex_label=self.order_options 
                    ,edge_label = self.g.es["label"],edge_width=3.25)
        
    
    def optimal_mixed_strategy(self):
        """
        This function solves the linear problem to achieve
        the probabilities for the general tie (if there is)
        """
        assert self.margin_mat_exist, "Margin Matrix M not Built"
        #We calculate w as the -min value of the matrix
        w = np.min(self.M)*-1
        #We increase the matrix M transposed by constant w
        M_positive = w + self.M.T
        #We define the linear programing problem to solve
        #We set the variables
        m = M_positive.shape[0]
        p = cp.Variable(shape=(m,1), name="p")
        #We define the constraints as p_x >= 0 and Mp >= e
        #where e is a column vector of length m containing ones
        e = np.ones(m).reshape(-1,1)
        constraints = [cp.matmul(M_positive, p) >= e,p>=0]
        #We define the objective
        objective = cp.Minimize(cp.sum(p, axis=0))
        #We define the problem and obtain the solution
        problem = cp.Problem(objective, constraints)
        solution = problem.solve()
        #We return p x w
        return p.value * w
    
    def compute_vt_metrics(self):
        """
        This function creates the vector of probabilities for the candidades
        """
        self.counting_preference()
        self.build_matrix_preference()
        self.build_margin_matrix()
        self.probability = self.optimal_mixed_strategy().T[0]
        self.computed_prob = True
        return
    
    def select_winner_option(self):
        """
        This function select a winner option given the 
        Margin Matrix, the optimal mixed strategy and random selecting 
        with the probabilities
        """
        assert self.computed_prob, "Probability metrics not computed"
        winner = random.choices(self.order_options,self.probability,k=1)
        #We show the probability rounded 
        prob_to_show = []
        for prob in self.probability:
            prob_to_show.append(round(prob,4))
        print("Candidates:",self.order_options)
        print("Probability for each candidate:",prob_to_show)
        print("Winner:",winner)
            