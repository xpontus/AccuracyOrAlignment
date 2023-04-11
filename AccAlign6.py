# %%
# Code for paper Accuracy or alignment: A conflict in the participatory modeling process?
# Andreas Nicolaidis Lindqvist, Pontus Svenson, Shane Carnohan, Bodil Karlssona

import mesa
import random
import json
import re
import math
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyvis


def graf_ToC():
    gnodes = ['Ba', 'Aa', 'Ta', 'Gia', 'Rl', 'Ngb', 'Nga',
              'A2', 'A1', 'A3', 'B1', 'B2', 'B3', 'Rs', 'Rf', 'Nrg']
    # B activity'
    # A's activity
    # Total activity
    # Gain per individual activity
    # Resource limit
    # Net gain for B
    # Net gain for A
    # A's effect 2
    # A's effect 1
    # A's effect 3
    # B's effect 1
    # B's effect 2
    # B's effect 3
    # Resource stock
    # Regeneration factor
    # Net resource growth

    gdata = {
        ('Ba', 'B1', 1),
        ('Ba', 'B2', -1),
        ('Aa', 'A1', 1),
        ('Aa', 'A3', -1),
        ('Ta', 'Gia', -1),
        ('Ta', 'Rs', -1),
        ('Gia', 'Ba', -1),
        ('Gia', 'Aa', -1),
        ('Gia', 'Nga', 1),
        ('Gia', 'B3', 1),
        ('Rl', 'Gia', -1),
        ('Nb', 'Ba', 1),
        ('Na', 'Aa', 1),
        ('A2', 'Nga', 1),
        ('A1', 'A2', 1),
        ('A3', 'Ta', -1),
        ('B1', 'Ta', 1),
        ('B2', 'Ngb', -1),
        ('B3', 'Ngb', 1),
        ('Rs', 'Rl', 1),
        ('Rs', 'Nrg', 1),
        ('Rf', 'Nrg', 1),
        ('Nrg', 'Rs', 1)
    }
    g = nx.DiGraph()
    g.add_nodes_from(gnodes)
    [g.add_edge(a, b, force=c) for a, b, c in gdata]

    return g


def del_random_elmts(li, n):  # delete n things from li
    indx_del = set(random.sample(range(len(li)), n))
    return [e for i, e in enumerate(li) if not i in indx_del]


def change_dir(li, n):  # li is list of (u,v,d), randomly reverse n items to (v,u,d)
    indx = set(random.sample(range(len(li)), n))
    to_change = [e for i, e in enumerate(li) if i in indx]
    re = [e for i, e in enumerate(li) if not i in indx]
    for e in to_change:
        (u, v, d) = e
        re.append((v, u, d))
    return re


def observe_graph(g, expertdomain, inerr=1, outerr=2):
    # c.	Number of errors per agent:
    # i.	Within expert domain an agent has one error
    # ii.	Outside expert domain an agent has 2-3 errors
    # Errors will always be in LINKS, all experts know about all nodes

    og = nx.DiGraph()
    og.add_nodes_from(list(g.nodes()))

    inlist = []
    outlist = []
    betweenlist = []

    misslist = []  # randomly add nodes that should be missed here -- see types of errors below

    nghostlinks = (inerr + outerr) // 2
    nmissedin = inerr - nghostlinks // 2
    if nmissedin < 0:
        nmissedin = 0
    nmissedout = outerr - nghostlinks // 2
    if nmissedout < 0:
        nmissedout = 0

    #print(f"inerr {inerr} outerr {outerr}, nghostlinks {nghostlinks}, nmissedin {nmissedin}, nmissedout {nmissedout}")

    realedges = list(g.edges(data=True))
    for e in realedges:
        (u, v, d) = e
        w = d['force']
        if u in misslist or v in misslist:  # Type ii error
            pass
        elif (u in expertdomain and v in expertdomain):
            inlist.append((u, v, d))
        elif u in expertdomain or v in expertdomain:
            betweenlist.append((u, v, d))
        else:  # must be outside of domain of expertise
            outlist.append((u, v, d))

    # b.	Five types of error can be introduced:
    # i.	The MM misses an existing link -- code below
    # ii.	The MM misses an existing node -- see misslist list above
    # iii.	The MM has wrong direction of a link -- see code below remove n randomly selected, add them with reversed direction
    # iv.	The MM contains a “ghost” node (a node that does not exist in reality)
    # v.	The MM contains a “ghost” link
    # Type iv and v not implemented: unless several experts have the same ghost node or link, they will be added when the expert has said them >threshold times

    # Tyoe i error: missing existing links:
    if nmissedin > len(inlist):
        nmissedin = len(inlist)
    if nmissedout > len(betweenlist + outlist):
        nmissedout = len(betweenlist + outlist)

    el = del_random_elmts(inlist, nmissedin) + \
        del_random_elmts(betweenlist+outlist, nmissedout)

    # Type iii error:
    nchangeddirs = 0
    el = change_dir(el, nchangeddirs)

    # Type v error:
    naddedlinks = nghostlinks
    for i in range(naddedlinks):
        # find something that can be added (ie, link that is not already in graph)
        while True:
            u = random.choice(list(g.nodes()))
            v = random.choice(list(g.nodes()))
            if not g.has_edge(u, v) and not g.has_edge(v, u):
                break
        # now know that neither u->v or v->u is in g
        el.append((u, v, 1))

    for e in el:
        (u, v, d) = e
        og.add_edge(u, v, force=d)

    return og


control_model = graf_ToC()
explist1 = ['A1', 'A2', 'Aa', 'Na', 'A3', 'Nga']
explist2 = ['Rf', 'Nfg', 'Rs', 'Rl', 'Ta', 'Gia']
explist3 = ['B1', 'B2', 'B3', 'Ba', 'Nb', 'Ngb']

obs1 = dict()
obs2 = dict()
obs3 = dict()

obs1[(1, 5)] = observe_graph(control_model, explist1, inerr=1, outerr=5)
obs1[(5, 5)] = observe_graph(control_model, explist1, inerr=5, outerr=5)
obs1[(0, 5)] = observe_graph(control_model, explist1, inerr=0, outerr=5)
obs1[(1, 1)] = observe_graph(control_model, explist1, inerr=1, outerr=1)

obs2[(1, 5)] = observe_graph(control_model, explist2, inerr=1, outerr=5)
obs2[(4, 5)] = observe_graph(control_model, explist2, inerr=4, outerr=5)
obs2[(5, 5)] = observe_graph(control_model, explist2, inerr=5, outerr=5)

obs3[(1, 5)] = observe_graph(control_model, explist3, inerr=1, outerr=5)
obs3[(5, 5)] = observe_graph(control_model, explist3, inerr=5, outerr=5)


def draw_graph(g, tit, savetit=None):
    nx.draw(g, with_labels=True)
    plt.title(tit)
    if savetit:
        plt.savefig(savetit, format="png")
    plt.show()


def jsonify(di):
    # di is dictionary with tuple (u,v) as key, convert to string since dict's aren't understood by JSON
    return [f"{key} : {val}" for key, val in di.items()]


def make_initial_history(controlgraph, expertgraph, expertdomain, invalue, outvalue):
    h = dict.fromkeys(list(controlgraph.edges()), outvalue-1)
    # for all u,v in expertgraph: if u and v in expdomain d = invalue else d = outvalue
    el = list(expertgraph.edges(data=False))
    for e in el:
        (u, v) = e
        if u in expertdomain and v in expertdomain:
            h[e] = invalue
        else:
            h[e] = outvalue
    return h


def jaccarddistance(graf1, graf2):
    # Jaccard distance between edge sets
    def jaccard(set1, set2):
        intersection = len(list(set1.intersection(set2)))
        union = (len(set1) + len(set2)) - intersection
        return float(intersection) / union

    def sorensendice(set1, set2):
        intersection = len(list(set1.intersection(set2)))
        union = (len(set1) + len(set2)) - intersection
        return float(2*intersection) / union

    s1 = set(graf1.edges(data=False))
    s2 = set(graf2.edges(data=False))
    if (s1 == s2):
        return 0
    else:
        return 1-jaccard(s1, s2)


def jaccarddistance2(graf1, graf2):
    # print(f"graf1 {list(graf1.edges())}, graf2 {list(graf2.edges())}")
    # print(f"graf1 {graf1}, graf2 {graf2}")
    r = jaccarddistance(graf1, graf2)
    # print(f"returrning {r}")
    return r


def first_elem_set(s):
    for e in s:
        break
    return e


class ExpertAgent(mesa.Agent):
    """ An expert with a domain of expertise.
    Parameters:
        Talkativeness = how much the expert wants to talk. normal talkativeness har värde 1, vissa har värde > 1 och pratar då mer, vissar har <1 och pratar då mindre
        Social status = the status of the agent in the eyes of the others. normal person har social status 1, vissa är viktigare än andra ich har > 1, vissa har mindre än andra och då värde < 1
        Expert area = Expertise domain -- list of nodes.
        Initial mental model = initial observation of the control model -- graph
        Focus memory = number of time steps that a suggestion made by any agent remains in the focus area memory of this agent. Integer
        Threshold = how many times does this expert need to hear a statement in order to believe it. Integer
        Initial history model = number of times that the expert has heard different statements from the control model earlier. 
                                Dictionary indexed by string representation of link. For each a-> stores number of times a->b is assumed to have been heard before

    Variables:
        Internal model = expert's view of the control model updated during the model building process. Graph
        History model = expert's memory of how many times they have heard a specific element of the model. 
                        Used when updating the mental model: if History model(statement) > threshold then statement is added to the internal mental model. 
                        Dictionary indexed by string representation of link: for each a->b stores number of times a->b has been mentioned
        Focus area = Expert's memory of what parts of the  modelling that have been talked about recently. Dictionary indexed by 

    """

    def __init__(self, unique_id, model, talkativeness, social_status, expertise_domain, initial_mental_model, focus_memory_size, threshold, initial_history_model):
        super().__init__(unique_id, model)

        self.talkativeness = talkativeness
        self.social_status = social_status
        # nodes that determine the expert domain
        self.expertise_domain = expertise_domain
        self.initial_mental_model = initial_mental_model.copy()
        self.focus_memory_size = focus_memory_size
        self.threshold = threshold
        self.initial_history_model = initial_history_model
        self.changes_made = 0

        self.mental_model = self.initial_mental_model
        self.history_model = self.initial_history_model
        self.focus_area = []

    def step(self):
        # model scheduler ensures that agents are called in order according to their talkativeness

        # suggest statement based on self.mental_model, self.model.consensus_model, self.focus_area -- choose the change that minimizes the distance
        # also uses focus area and self.focus_memory_size when making suggestion
        # suggestion can be addition or removal of link

        suggested_change = self.get_suggestion()

        what, _ = suggested_change
        if what != "Pass":
            # tell other agents to update their mental models -- they will update their histories and mental models if threshold in history is passed

            for other_agent in self.model.schedule.agents:
                if other_agent != self:
                    # use other agents method for this: self.dominance and possibly other factors in self should influence this
                    other_agent.update_mental_model(suggested_change, self)

            # get a vote and add suggestion to consensus model if approved
            # also update focus area.
            count = 1  # self assumed to vote Yes
            for other_agent in self.model.schedule.agents:
                if other_agent != self:
                    count += 1 if other_agent.vote(suggested_change) else 0
            if count >= self.model.votes_required_for_change:
                self.model.update_consensus(suggested_change)
                # move this and change True to False if want to add un-accepted suggestions to focus area
                for agent in self.model.schedule.agents:
                    agent.update_focus_area(suggested_change, True)
        else:
            pass


    def get_suggestion(self):
        # first step: c.	The agent randomly suggests a link from within his/her expert domain
        # b.	What determines if the agent wants to add a new link or remove an existing link?
        # i.	Determining factors are: what is the current, memory of the agent (the focus area) and the change that minimizes the jaccard distance between the consensus model and the MM of the agent.
        # ii.	If the focus area includes any part of my expert domain, I select the action (add or remove a link) that minimizes the jaccard distance between the consensus model and my MM.
        # iii.	If the focus area is completely outside my expert domain, I only suggest to add new links (still to minimize the jaccard distance) to avoid conflict (Conflict avoidance is assumed)

        # self.focus_area_nodes() are the nodes that are in my focus area/memory
        # self.expertise_domain are the nodes I am an expert on
        # self.mental_model is my internal model
        # self.model.consensus_model is the shared model which I want to get as close to self.mental_model as possible


        nodes_in_both = self.focus_area_nodes().intersection(self.expertise_domain)

        consensus_links = set(self.model.consensus_model.edges())
        mental_model_links = set(self.mental_model.edges())
        consensus_minus_mental_links = consensus_links.difference(
            mental_model_links)
        mental_minus_consensus_links = mental_model_links.difference(
            consensus_links)

        # add candidate moves that add a link and minimizes distance b/w self.mental_model and self.model.consensus_model
        # ie, find any link that is in self.mental_model and not in self.model.consensus_model

        candidate = None
        candidate2 = None
        if len(mental_minus_consensus_links) > 0:
            (u, v) = first_elem_set(mental_minus_consensus_links)
            attrs = nx.get_edge_attributes(self.mental_model, "force")
            w = attrs[(u, v)]
            candidate = ("Add", (u, v, {'force': w}))
        # focus area and expertise have overlap, so deletion is possible
        if len(nodes_in_both) > 0 and len(consensus_minus_mental_links) > 0:
            # either add any link in self.mental_model and not in self.model.consensus_model (see above) or
            # delete link that is not in self.mental_model but is in self.model.consensus_model and involves node in nodes_in_both
            cands = list(filter(
                lambda edg: edg[0] in nodes_in_both or edg[1] in nodes_in_both, consensus_minus_mental_links))
            if len(cands) > 0:
                # any link in consensus_minus_mental_links that involves at least one node from nodes_in_both
                (u, v) = first_elem_set(cands)
                candidate2 = ("Delete", (u, v))

        if not candidate2 is None:
            if candidate is None:
                candidate = candidate2
            elif random.random() < .5:
                candidate = candidate2
        if candidate is None:
            candidate = ("Pass", 1)
        return candidate
        if random.random() < .5:
            return ("Delete", ('Ba', 'B2'))
        else:
            return ("Add", ('B1', 'B2', {'force': 1}))

    def update_mental_model(self, suggestion, suggester_agent):
        # update self.history_model based on suggestion that was proposed by suggester_agent
        # if self.history_model(suggestion) > self.threshold also update self.mental_model

        what, edge = suggestion
        if what == "Pass":
            print("shouldn't get here")
            return
        weight = suggester_agent.social_status
        if what == "Add":
            u, v, w = edge
            if (not (u, v) in self.history_model):
                self.history_model[(u, v)] = 0
            self.history_model[(u, v)] += weight
            if self.history_model[(u, v)] >= self.threshold and not self.mental_model.has_edge(u, v):
                self.mental_model.add_edge(u, v, force=w)
                self.changes_made += 1
        else:
            u, v = edge
            if (not (u, v) in self.history_model):
                self.history_model[(u, v)] = 0
            self.history_model[(u, v)] -= weight
            if self.history_model[(u, v)] < self.threshold:
                if (self.mental_model.has_edge(u, v)):
                    self.mental_model.remove_edge(u, v)
                    self.changes_made += 1
                else:
                    pass



    def update_focus_area(self, suggestion, vote):
        # update self.focus_area with suggestion and based on results of vote

        what, edge = suggestion
        if what == "Pass":
            print("Shouldn't get here")
            return
        if what == "Add":
            u, v, _ = edge
        else:
            u, v = edge
        if len(self.focus_area) >= self.focus_memory_size:
            self.focus_area.pop()

        self.focus_area.append((u, v))


    def focus_area_nodes(self):
        return {n for el in self.focus_area for n in el}

    def vote(self, suggestion):
        (what, edge) = suggestion
        if what == "Add":
            (u, v, _) = edge
            if u in self.expertise_domain and v in self.expertise_domain and not self.mental_model.has_edge(u, v):
                return False
            else:
                return True
        elif what == "Delete":
            (u, v) = edge
            if u in self.expertise_domain and v in self.expertise_domain and self.mental_model.has_edge(u, v):
                return False
            else:
                return True
        return True

    # possible suggestions:
    # ("Add", (u, v, {'force': w}))
    # ("Delete", (u,v))


def all_dist(m):
    r = []
    for a1 in m.schedule.agents:
        for a2 in m.schedule.agents:
            if a1.unique_id < a2.unique_id:
                r.append(jaccarddistance(a1.mental_model, a2.mental_model))
    return r


class GMBModel3(mesa.Model):
    def __init__(self, Sp, Ep):
        self.control_model = control_model  # 18 noder och 23 länkar
        self.consensus_model = nx.DiGraph()

        # baseactivation runs the agents in the order they were addded -- so add them according to talkativeness
        self.schedule = mesa.time.BaseScheduler(self)
        self.running = True   # required for batch run -- run ends when self.running = False



        if Sp == 0:
            soc1, soc2, soc3 = .1, .1, .1
        elif Sp == 1:
            soc1, soc2, soc3 = 5, 1, 1
        elif Sp == 2:
            soc1, soc2, soc3 = 5, 1, .5
        elif Sp == 3:
            soc1, soc2, soc3 = .5, .5, .5
        elif Sp == 4:
            soc1, soc2, soc3 = 1, .5, .5
        elif Sp == 5:
            soc1, soc2, soc3 = .1, .5, .5
        elif Sp == 6:
            soc1, soc2, soc3 = 1, 1, 1
        elif Sp == 7:
            soc1, soc2, soc3 = 1, .5, .1
        else:
            print("shouldn't get here, Sp")

        if Ep == 0:  # use observe_graph(control_model, explist1, inerr = ie1, outerr = oe1 OUTSIDE of the average over iterations -- so compute them outside of GMBModel3
            og1, og2, og3 = obs1[(1, 5)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 1:
            og1, og2, og3 = obs1[(5, 5)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 2:
            og1, og2, og3 = obs1[(5, 5)], obs2[(4, 5)], obs3[(1, 5)]
        elif Ep == 3:
            og1, og2, og3 = obs1[(1, 5)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 4:
            og1, og2, og3 = obs1[(5, 5)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 5:
            og1, og2, og3 = obs1[(0, 5)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 6:
            og1, og2, og3 = obs1[(1, 1)], obs2[(1, 5)], obs3[(1, 5)]
        elif Ep == 7:
            og1, og2, og3 = obs1[(1, 1)], obs2[(5, 5)], obs3[(1, 5)]
        elif Ep == 8:
            og1, og2, og3 = obs1[(1, 1)], obs2[(5, 5)], obs3[(5, 5)]
        else:
            print("shouldn't happen Ep")


        # these should be sorted according to talkativeness!
        paramlist = [{'talkativeness': 1.2,
                      'social_status': soc1,
                      'expertise_domain': explist1,
                      'initial_mental_model': og1,
                      'focus_memory_size': 3,  # was 3 for all experts
                      'threshold': 3
                      },
                     {'talkativeness': 1,
                      'social_status': soc2,
                      'expertise_domain': explist2,
                      'initial_mental_model': og2,
                      'focus_memory_size': 3,
                      'threshold': 3
                      },
                     {'talkativeness': 0.8,
                      'social_status': soc3,
                      'expertise_domain': explist3,
                      'initial_mental_model': og3,
                      'focus_memory_size': 3,
                      'threshold': 3
                      }]

        self.num_agents = len(paramlist)
        self.votes_required_for_change = math.ceil(self.num_agents/2)

        prevagent = None
        firstag = None
        # sort according to talkativeness
        for i, params in enumerate(paramlist):
            # inithist is the initial history -- how many times an expert has seen a specific link
            # so h1[(u,v)] = d means that agent 1 has seen the link u->b d times
            # links within the expert domain have d = threshold +2, links outside have d = threshold
            # other links in control model have d = 0
            inithist = make_initial_history(
                control_model, params["initial_mental_model"], params["expertise_domain"], params["threshold"]+2, params["threshold"])
            a = ExpertAgent(i, self, params["talkativeness"], params["social_status"], params["expertise_domain"],
                            params["initial_mental_model"], params["focus_memory_size"], params["threshold"], inithist)
            self.schedule.add(a)
            if prevagent:
                prevagent.next_agent = a
            else:
                firstag = a
            prevagent = a
        a.next_agent = firstag

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={"M_jaccdist_control_consensus": lambda m: jaccarddistance(
                m.control_model, m.consensus_model)},
            agent_reporters={"A_jaccdist_consensus_agentmodel": lambda a: jaccarddistance(a.model.consensus_model, a.mental_model),
                             "A_jaccdist_next_agent": lambda a: jaccarddistance(a.mental_model, a.next_agent.mental_model),
                             "A_changes_made_to_mental_model": lambda a: a.changes_made})

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        # put this before step() to measure also initial state (losing final state)
        self.datacollector.collect(self)

    def update_consensus(self, accepted_change):
        #print(f"** Updating consensus model with {accepted_change}")
        what, edge = accepted_change
        #print(f"what {what}, edge {edge}")
        if what == "Add":
            # print("add")
            u, v, w = edge
            self.consensus_model.add_edge(u, v, force=w)
        else:
            # print("del")
            u, v = edge
            if (self.consensus_model.has_edge(u, v)):
                self.consensus_model.remove_edge(u, v)
            else:
                print(f"tried to delete non-existing edge {u}, {v}")


def dejsonify(str):
    # str is string representation of dict: "(u, v) : d" means di((u,v)) = d
    di = {}
    for l in str:
        k, v = l.split(':')
        d = int(v)
        u, v = re.sub("\ |\'|\(|\)", '', k).split(',')
        di[(u, v)] = d
    return di


def plot_values_by_agent(df_averaged_byagent, valstr, funstr, parstr):
    plotoptions = ['ro-', 'gx-', 'b+-']
    poind = 0
    for ag_df in df_averaged_byagent:
        (_, ag_df_df) = ag_df
        ag_df_df[(valstr, funstr)].plot(style=plotoptions[poind])
        poind = (poind + 1) % len(plotoptions)
    titstr = f'{valstr} - {funstr} - {parstr}'
    plt.title(titstr)
    plt.savefig(f"data/{titstr}.png", format="png")
    plt.show()


if __name__ == "__main__":

    draw_graph(control_model, "control model",
               savetit="data/Control model.png")
    nx.write_graphml(control_model, "data/Control model.graphml")
    nt_g = pyvis.network.Network('500px', '500px')
    nt_g.from_nx(control_model)
    nt_g.show('data/Control model.html')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    paramlist = {"Sp": [0, 1, 2, 3, 4, 5, 6, 7],
                 "Ep": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
    parameternames = list(paramlist.keys())

    agentvars = ['A_jaccdist_consensus_agentmodel',
                 'A_jaccdist_next_agent', 'A_changes_made_to_mental_model']
    modelvars = ['M_jaccdist_control_consensus']
    agentstatfuns = ['mean', 'std']
    modelstatfuns = ['mean', 'std']

    # compute observed graphs here -- needs to be outside of the averaged in batchrun

    results = mesa.batch_run(
        GMBModel3,
        parameters=paramlist,
        iterations=1,
        max_steps=10,
        number_processes=1,
        data_collection_period=1,
        display_progress=False
    )
    results_df = pd.DataFrame(results)

    results_df.to_excel("data/output.xlsx")

    tsteps = max(results_df.Step)+1

    # get an iterator over different parameter settings -- each item will contain all results from a specific parameter setting
    expiterator = results_df.groupby(['Sp', 'Ep'])

    for expresults in expiterator:
        (params, df) = expresults
        paramstr = f"Sp {params[0]}, Ep {params[1]}"
        print(paramstr)

        # aggregate over iterations: 
        df_averaged = df.groupby(['AgentID', 'Step']).agg({'M_jaccdist_control_consensus': ['mean', 'std'], 'A_jaccdist_consensus_agentmodel': [
            'mean', 'std'],  'A_jaccdist_next_agent': ['mean', 'std'], 'A_changes_made_to_mental_model': ['mean', 'std']})

        df_averaged_byagent = df_averaged.groupby(['AgentID'])

        for valstr in agentvars:
            for funstr in agentstatfuns:
                plot_values_by_agent(df_averaged_byagent,
                                     valstr, funstr, paramstr)

        # Model variable (M_*) are collected for model and stored for all agents

        df_model = df_averaged_byagent.get_group(0)

        for valstr in modelvars:
            for funstr in modelstatfuns:
                df_model[(valstr, funstr)].plot()
                titstr = f'{valstr} - {funstr} - {paramstr}'
                plt.title(titstr)
                plt.savefig(f"data/{titstr}.png", format="png")
                plt.show()

    endresults_temp = results_df[results_df["Step"] == tsteps-1]     # TODO
    endresults_averaged = endresults_temp.groupby(parameternames).agg({'M_jaccdist_control_consensus': ['mean', 'std'], 'A_jaccdist_consensus_agentmodel': [
        'mean', 'std'], 'A_jaccdist_next_agent': ['mean', 'std'],  'A_changes_made_to_mental_model': ['mean', 'std']})

    print("Values at final step, averaged")
    print(endresults_averaged)


# %%

# %%
