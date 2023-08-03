from AlphaJanggi_utils import *
import torch
from numpy.random import choice
import numpy as np


def ExpandTree(model, tree):
    node = tree.cur
    s = node.state
    path = []
    e = None
    while s.v_winner==-1:
        x, e = node.SelectMove()
        path.append((x, node))
        if e.Node == None:
            break
        node = e.Node
        s = node.state
    if e == None:
        return
    if e.Node == None:
        nextstate = s.move(*e.move)
        Leaf_Multi(model, [(nextstate, tree, e)])
    nxt = e.Node
    V = -e.Node.V
    for x, node in reversed(path): 
        if nxt.state.v_winner!=-1:
            if nxt.state.v_winner == node.state.turn:
                node.state.v_winner = node.state.turn
                node.winedge = x, node.edges[x]
                node.V = 1.0
            elif node.Q[x]>=-1:
                node.Q[x]=-2.0
                node.P[x]=0.0
                node.losenum-=1
                if node.losenum==0:
                    node.state.v_winner = 1-node.state.turn
                    node.V = -1.0
        else:
            node.Q[x]+=(V-node.Q[x])/node.NN[x]
        node.N += 1
        node.NN[x]+=1
        V=-V
        nxt = node

def ExpandTree_Batches2(model, tree, think_time = 32, verbose = False):
    stack = list()
    leaf = list()
    edges = list()
    node = tree.cur
    if node.state.v_winner != -1:
        return
    stack.append((node, think_time))

    while len(stack)>0:
        node, num = stack[-1]
        stack.pop()
        searchnum = min(num, node.losenum)
        if node.N<256 and searchnum>=2:
            searchnum = 2
        else:
            searchnum = 1
            
        subnum = num//searchnum
        ind = node.argPrior(searchnum)
        node.searchnum = 0
        for x, i in enumerate(ind):
            e = node.edges[i]
            if e.Node == None:
                s = node.state.move(*e.move)
                leaf.append((s, tree, e))
                edges.append(e)
                node.searchnum += 1
            elif e.Node.state.v_winner!=-1:
                print("must not happen 2")
                continue
            elif x == 0 and len(ind) >= 2:
                i2 = ind[1]
                emph = subnum
                num -= subnum
                C = 5*np.sqrt(node.N+emph)
                while num>=subnum and C*node.P[i]/(node.NN[i]+emph)+(node.Q[i]*(node.NN[i]-1)-emph)/(node.NN[i]-1+emph) >= C*node.P[i2]/node.NN[i2]+node.Q[i2]:
                    emph += subnum
                    num -= subnum
                    C = 5*np.sqrt(node.N+emph)
                stack.append((e.Node, emph))
                node.searchnum += 1
            else:
                stack.append((e.Node, subnum))
                node.searchnum += 1
                num -= subnum
            if num<subnum:
                break
    
    Leaf_Multi(model, leaf)
    
    while len(edges)>0:
        e = edges[-1]
        edges.pop()
        node = e.parnode
        x = e.num
        nxt = e.Node
        if node.state.v_winner!=-1:
            pass
        elif nxt.state.v_winner == node.state.turn:
            node.state.v_winner = node.state.turn
            node.winedge = x, e
            node.V = 1.0
            node.Q[x] = 1.0
            Nx = node.NN[x]-1
            node.N += - Nx + nxt.N
            node.NN[x] = nxt.N + 1
        elif nxt.state.v_winner == 1-node.state.turn:
            if node.Q[x]>=-1:
                N = node.N
                Nx = node.NN[x]-1
                N2 = N - Nx + nxt.N
                node.V = (node.V*N-node.Q[x]*Nx-nxt.N)/N2
                node.N = N2
                node.NN[x]=2
                node.Q[x]=-2.0
                node.P[x]=0.0
                node.losenum-=1
                if node.losenum==0:
                    node.state.v_winner = 1-node.state.turn
                    node.V = -1.0
            else:
                print("must not happen 1")
                node.N+=1
        else:
            N = node.N
            Nx = node.NN[x]-1
            N2 = N - Nx + nxt.N
            node.V = (node.V*N-node.Q[x]*Nx-nxt.V*nxt.N)/N2
            node.N = N2
            node.Q[x] = -nxt.V
            node.NN[x] = nxt.N+1
        node.searchnum-=1
        if node.searchnum==0 and not node.root:
            edges.append(node.paredge)
    stack.clear()
    leaf.clear()
    edges.clear()

def Find(model, tree, think_time, topk = 0):
    Node = tree.cur
    # if Node.edges == None:
    #     Node.MakeEdges()
    K = Node.edgenum
    while Node.N<think_time and Node.state.v_winner==-1:
        ExpandTree_Batches2(model, tree, 32)
    print(Node.state.v_winner)
    print(Node.losenum, Node.edgenum)
    if Node.state.v_winner==Node.state.turn:
        print("Winning State")
        print([Node.winedge[1].move])
        return [(Node.winedge[1].move, 1.0)]
    elif Node.state.v_winner==1-Node.state.turn:
        print("Losing State")
    sum = np.sum(Node.NN-1)
    moves = [(Node.NN[i]-1, i) for i in range(K)]
    moves.sort()
    movess = [(Node.edges[i].move, n/sum) for n, i in reversed(moves)]
    if topk == -1:
        topk = K
    else:
        topk = min(topk, K)
    for m, n in movess[:topk]:
        print(*m, n)
    return movess

def MoveTree(model, tree, move):
    error = True
    node = tree.cur
    s = node.state
    for e in node.edges:
        if e.move==move:
            error = False
            if e.Node==None:
                nextstate = s.move(*move)
                Leaf_Multi(model, [(nextstate, tree, e)])
            tree.cur = e.Node
            break
    return error

@torch.no_grad()
def Play(model, sangma, auto = (), default_think_time = 800, reverse = False):
    board = State(sangma)
    tree = MCTree()
    boards = [board]
    Leaf_Multi(model, [(board, tree, None)], begin = True)
    curs = [tree.cur]
    tree.cur.root = True
    quit = False
    while board.winner==-1 and not quit:
        #board.display(reverse)
        board.display()
        validmoves = board.validMoves()
        error = True
        while error:
            if board.turn in auto:
                s = "a "+str(default_think_time)
            else:
                print(">>", end=' ')
                s = input()
                if len(s)==0:
                    continue
            if s[0]=='u':
                boards = boards[:-1]
                curs[-1].root = False
                curs = curs[:-1]
                curs[-1].root = True
                board=boards[-1]
                tree.cur=curs[-1]
                break
            elif s[0]=='q':
                quit=True
                break
            elif s[0]=='a':
                #try:
                com, think_time = s.split()
                think_time = int(think_time)
                moves = Find(model, tree, think_time)
                move = moves[0][0]
                print(*move)
                MoveTree(model, tree, move)
                board = tree.cur.state
                boards.append(board)
                curs.append(tree.cur)
                break
                # except Exception as e:
                #     print("Input Error", e)
            elif s[0]=='t':
                try:
                    com, think_time, topk = s.split()
                    think_time = int(think_time)
                    topk = int(topk)
                    Find(model, tree, think_time, topk)
                except Exception as e:
                    print("Input Error", e)
            elif s[0]=='m':
                try:
                    com, a, b, c, d = s.split()
                    a, b, c, d = map(int, (a, b, c, d))
                    move = ((a, b), (c, d))
                    error = MoveTree(model, tree, move)
                    if error:
                        print("Invalid Move")
                        continue
                    board = tree.cur.state
                    boards.append(board)
                    curs.append(tree.cur)
                    break
                except Exception as e:
                    print("Input Error", e)
            else:
                try:
                    a, b, c, d = s.split()
                    a, b, c, d = map(int, (a, b, c, d))
                    move = ((a, b), (c, d))
                except Exception as e:
                    print("Input Error", e)
                error = MoveTree(model, tree, move)
                if error:
                    print("Invalid Move")
                    continue
                board = tree.cur.state
                boards.append(board)
                curs.append(tree.cur)
                break


if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_alpha = AlphaJanggi().to(device)
    model_alpha.load_state_dict(torch.load('AlphaJanggi_230605_40960.pth', map_location=torch.device(device)))
    _ = model_alpha.eval()
    model_alpha.device = device
    
    print("Choose Side")
    while True:
        n = int(input())
        if n == 0:
            reverse = False
            break
        elif n == 1:
            reverse = True
            break
        else:
            print("Choose Side 0 or 1")

    print("Choose Sang Ma")

    print("Han's Sang Ma")
    print("0 : SM SM\n1 : MS SM\n2 : SM MS\n3 : MS MS")
    x2 = int(input())

    print("Cho's Sang Ma")
    print("0 : SM SM\n1 : MS SM\n2 : SM MS\n3 : MS MS")
    x1 = int(input())

    print("Auto think time")
    think_time = int(input())
    auto_move = ()
    if think_time != 0:
        print("Auto Side")
        auto_move = tuple(map(int, input().split()))

    Play(model_alpha, [x1, x2], auto = auto_move, default_think_time = think_time, reverse = reverse)

