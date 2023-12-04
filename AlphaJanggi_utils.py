import copy
import numpy as np
from numpy.random import choice
import torch
from torch import nn
import torch.nn.functional as F
import pickle

score = [0, -2, 2, -3, 3, -5, 5, -7, 7, -13, 13, -3, 3, 0, 0]
dir = [(1, 0), (0, 1), (-1, 0), (0, -1)]

movecode = dict()

def fillmovecode():
    global movecode
    num = 0
    for i in range(1, 9):
        movecode[(i, 0)] = num
        num+=1
        movecode[(-i, 0)] = num
        num+=1
    for j in range(1, 10):
        movecode[(0, j)] = num
        num+=1
        movecode[(0, -j)] = num
        num+=1
    for i in (2, -2):
        for j in (-3, -1, 1, 3):
            movecode[(i, j)] = num
            num+=1
            movecode[(j, i)] = num
            num+=1
    for i in (-1, 1):
        for j in (-1, 1):
            movecode[(i, j)] = num
            num+=1
            movecode[(i*2, j*2)] = num
            num+=1
    movecode[(0, 0)]=num
    num+=1
fillmovecode()

class State:
    def __init__(self, sangma = (0, 0), copy_place = None):
        self.board = [[0 for i in range(10)] for j in range(9)]
        if copy_place == None:
            self.place = [[],
                [(0, 3), (2, 3), (4, 3), (6, 3), (8, 3)], # 졸 1
                [(0, 6), (2, 6), (4, 6), (6, 6), (8, 6)], # 병 2
                [(1+(sangma[0]&1), 0), (6+(sangma[0]>>1), 0)], # 초상 3
                [(1+(sangma[1]&1), 9), (6+(sangma[1]>>1), 9)], # 한상 4
                [(2-(sangma[0]&1), 0), (7-(sangma[0]>>1), 0)], # 초마 5
                [(2-(sangma[1]&1), 9), (7-(sangma[1]>>1), 9)], # 한마 6
                [(1, 2), (7, 2)], # 초포 7
                [(1, 7), (7, 7)], # 한포 8
                [(0, 0), (8, 0)], # 초차 9
                [(0, 9), (8, 9)], # 한차 10
                [(3, 0), (5, 0)], # 초사 11
                [(3, 9), (5, 9)], # 한사 12
                [(4, 1)], # 초궁 13
                [(4, 8)] # 한궁 14
                ]
        else:
            self.place = copy.deepcopy(copy_place)

        for i, p in enumerate(self.place):
            for x, y in p:
                self.board[x][y]=i
        self.turn = 0
        self.winner = -1
        self.num = 0
        self.endturn = 200
        self.lastmoves = [None for _ in range(8)]
        self.moves=None
        self.Checked = False
        self.v_winner = -1
        self.prehash = None

    def inBoard(self, i, j):
        return 0<=i and i<9 and 0<=j and j<10
    
    def sameSide(self, a, b):
        x = self.board[a[0]][a[1]]
        y = self.board[b[0]][b[1]]
        return x!=0 and y!=0 and (x-y)%2==0
    
    def validate(self, p0, p1, CheckList):
        if self.sameSide(p0, p1):
            return False
        ret = True
        pi, pj = p0
        i, j = p1
        x = self.board[pi][pj]
        y = self.board[i][j]
        self.board[i][j] = x
        self.board[pi][pj] = 0
        ind = None
        for indx, pp in enumerate(self.place[x]):
            if pp==p0:
                ind=indx
                break
        self.place[x][ind]=p1
        if y!=0:
            self.place[y].remove(p1)
        
        if x==13 or x==14:
            if self.isCheck(self.turn):
                ret = False
        else:
            for zi, zj, way in CheckList:
                z = self.board[zi][zj]
                if i==zi and j==zj:
                    continue
                if z==7 or z== 8:
                    jump = False
                    for pz in way:
                        if self.board[pz[0]][pz[1]]==7 or self.board[pz[0]][pz[1]]==8:
                            jump=False
                            break
                        if self.board[pz[0]][pz[1]]!=0:
                            if jump:
                                jump=False
                                break
                            else:
                                jump=True
                    if jump:
                        ret=False
                else:
                    ret = False
                    for pz in way:
                        if self.board[pz[0]][pz[1]]!=0:
                            ret=True
                            break
                if not ret:
                    break
            
        self.board[pi][pj] = x
        self.board[i][j] = y
        self.place[x][ind]=p0
        if y!=0:
            self.place[y].append(p1)

        return ret

    def MakeCheckList(self, player):
        CheckList = []
        if player == 0:
            kx, ky = self.place[13][0]
            if self.board[kx-1][ky] == 2:
                CheckList.append((kx-1, ky, []))
            if self.board[kx+1][ky] == 2:
                CheckList.append((kx+1, ky, []))
            if self.board[kx][ky+1]==2:
                CheckList.append((kx, ky+1, []))
            if self.board[4][1] == 2 and ky == 0 and kx != 4:
                CheckList.append((4, 1, []))

            if kx == 4 and ky == 1:
                if self.board[3][2] == 2:
                    CheckList.append((3, 2, []))
                if self.board[5][2] == 2:
                    CheckList.append((5, 2, []))
                for i in (3, 5):
                    for j in (0, 2):
                        if self.board[i][j] == 10:
                            CheckList.append((i, j, []))
            elif kx != 4 and ky != 1:
                if self.board[4][1] == 10:
                    CheckList.append((4, 1, []))
                y = self.board[8-kx][2-ky]
                if  y == 10 or y == 8:
                    CheckList.append((8-kx, 2-ky, [(4, 1)]))

        else:
            kx, ky = self.place[14][0]

            if self.board[kx-1][ky] == 1:
                CheckList.append((kx-1, ky, []))
            if self.board[kx+1][ky] == 1:
                CheckList.append((kx+1, ky, []))
            if self.board[kx][ky-1]==1:
                CheckList.append((kx, ky-1, []))
            if ky == 9 and self.board[4][8] == 1 and kx!=4:
                CheckList.append((4, 8, []))

            if kx == 4 and ky == 8:
                if self.board[3][7] == 1:
                    CheckList.append((3, 7, []))
                if self.board[5][7] == 1:
                    CheckList.append((5, 7, []))
                
                for i in (3, 5):
                    for j in (7, 9):
                        if self.board[i][j] == 9:
                            CheckList.append((i, j, []))
            elif kx != 4 and ky != 8:
                if self.board[4][8] == 9:
                    CheckList.append((4, 8, []))
                y = self.board[8-kx][16-ky]
                if  y == 9 or y == 7:
                    CheckList.append((8-kx, 16-ky, [(4, 8)]))

        for i, j in self.place[4-player]:
            if (abs(i-kx)==2 and abs(j-ky)==3) or (abs(i-kx)==3 and abs(j-ky)==2):
                ddi = 1 if i-kx>0 else -1
                ddj = 1 if j-ky>0 else -1
                if self.board[kx+ddi*2][ky+ddj*2]==0 or self.board[kx+ddi][ky+ddj]==0:
                    CheckList.append((i, j, [(kx+ddi*2, ky+ddj*2), (kx+ddi, ky+ddj)]))
        
        for i, j in self.place[6-player]:
            if (abs(i-kx)==1 and abs(j-ky)==2) or (abs(i-kx)==2 and abs(j-ky)==1):
                ddi = 1 if i-kx>0 else -1
                ddj = 1 if j-ky>0 else -1
                CheckList.append((i, j, [(kx+ddi, ky+ddj)]))
        
        for i, j in self.place[8 - player]:
            if kx == i:
                mine = 0
                yours = 0
                way = []
                for nj in range(min(ky, j)+1, max(ky, j)):
                    way.append((i, nj))
                    y = self.board[i][nj]
                    if y!=0:
                        if y%2!=player:
                            mine += 1
                        else:
                            yours += 1
                if mine+yours<=2 and yours<=1:
                    CheckList.append((i, j, way))
            elif ky == j:
                mine = 0
                yours = 0
                way = []
                for ni in range(min(kx, i)+1, max(kx, i)):
                    way.append((ni, j))
                    y = self.board[ni][j]
                    if y!=0:
                        if y%2!=player:
                            mine += 1
                        else:
                            yours += 1
                if mine+yours<=2 and yours<=1:
                    CheckList.append((i, j, way))

        for i, j in self.place[10 - player]:
            if kx == i:
                mine = 0
                yours = 0
                way = []
                for nj in range(min(ky, j)+1, max(ky, j)):
                    way.append((i, nj))
                    y = self.board[i][nj]
                    if y!=0:
                        if y%2!=player:
                            mine += 1
                        else:
                            yours += 1
                            break
                if mine<= 1 and yours==0:
                    CheckList.append((i, j, way))
            elif ky == j:
                mine = 0
                yours = 0
                way = []
                for ni in range(min(kx, i)+1, max(kx, i)):
                    way.append((ni, j))
                    y = self.board[ni][j]
                    if y!=0:
                        if y%2!=player:
                            mine += 1
                        else:
                            yours += 1
                            break
                if mine<= 1 and yours==0:
                    CheckList.append((i, j, way))
        
        Check=False
        for zi, zj, way in CheckList:
            z = self.board[zi][zj]
            if z==7 or z==8:
                jump = False
                for i, j in way:
                    if self.board[i][j]==7 or self.board[i][j]==8:
                        jump=False
                        break
                    if self.board[i][j]!=0:
                        if jump:
                            jump=False
                            break
                        else:
                            jump=True
                if jump:
                    Check=True

            else:
                Check=True
                for i, j in way:
                    if self.board[i][j]!=0:
                        Check=False
            if Check:
                break
        return CheckList, Check

    def addmove(self, l, p0, p1, CheckList):
        if self.validate(p0, p1, CheckList):
            l.append((p0, p1))
    
    def validMoves(self):
        if self.moves!=None:
            return self.moves
        t = self.turn
        CheckList, Check = self.MakeCheckList(t)
        self.Checked = Check
        moves = list()
        if t == 0:
            for p0 in self.place[1]:
                i, j = p0
                if i > 0:
                    self.addmove(moves, p0, (i-1, j), CheckList)
                if i < 8:
                    self.addmove(moves, p0, (i+1, j), CheckList)
                if j < 9:
                    self.addmove(moves, p0, (i, j+1), CheckList)
                if (i==3 or i==5) and j==7:
                    self.addmove(moves, p0, (4, 8), CheckList)
                elif i==4 and j==8:
                    self.addmove(moves, p0, (3, 9), CheckList)
                    self.addmove(moves, p0, (5, 9), CheckList)
            for pp in (11, 13):
                for p0 in self.place[pp]:
                    i, j = p0
                    if i == 4 and j == 1:
                        for ni in range(3, 6):
                            for nj in range(0, 3):
                                self.addmove(moves, p0, (ni, nj), CheckList)
                    else:
                        self.addmove(moves, p0, (4, 1), CheckList)
                        if j == 1:
                            self.addmove(moves, p0, (i, j-1), CheckList)
                            self.addmove(moves, p0, (i, j+1), CheckList)
                        elif i == 4:
                            self.addmove(moves, p0, (i-1, j), CheckList)
                            self.addmove(moves, p0, (i+1, j), CheckList)
                        else:
                            self.addmove(moves, p0, (i, 1), CheckList)
                            self.addmove(moves, p0, (4, j), CheckList)
        else:
            for p0 in self.place[2]:
                i, j = p0
                if i > 0:
                    self.addmove(moves, p0, (i-1, j), CheckList)
                if i < 8:
                    self.addmove(moves, p0, (i+1, j), CheckList)
                if j > 0:
                    self.addmove(moves, p0, (i, j-1), CheckList)
                if (i==3 or i==5) and j==2:
                    self.addmove(moves, p0, (4, 1), CheckList)
                elif i==4 and j==1:
                    self.addmove(moves, p0, (3, 0), CheckList)
                    self.addmove(moves, p0, (5, 0), CheckList)
            for pp in (12, 14):
                for p0 in self.place[pp]:
                    i, j = p0
                    if i == 4 and j == 8:
                        for ni in range(3, 6):
                            for nj in range(7, 10):
                                self.addmove(moves, p0, (ni, nj), CheckList)
                    else:
                        self.addmove(moves, p0, (4, 8), CheckList)
                        if j == 8:
                            self.addmove(moves, p0, (i, j-1), CheckList)
                            self.addmove(moves, p0, (i, j+1), CheckList)
                        elif i == 4:
                            self.addmove(moves, p0, (i-1, j), CheckList)
                            self.addmove(moves, p0, (i+1, j), CheckList)
                        else:
                            self.addmove(moves, p0, (i, 8), CheckList)
                            self.addmove(moves, p0, (4, j), CheckList)

        for p0 in self.place[3+t]:
            i, j = p0
            for di, dj in dir:
                mi = i+di
                mj = j+dj
                k = di+dj
                l = di-dj
                if self.inBoard(mi+k+k, mj+k+k) and self.board[mi][mj]==0 and self.board[mi+k][mj+k]==0:
                        self.addmove(moves, p0, (mi+k+k, mj+k+k), CheckList)
                if self.inBoard(mi+l+l, mj-l-l) and self.board[mi][mj]==0 and self.board[mi+l][mj-l]==0:
                        self.addmove(moves, p0, (mi+l+l, mj-l-l), CheckList)
        
        for p0 in self.place[5+t]:
            i, j = p0
            for di, dj in dir:
                mi=i+di
                mj=j+dj
                k=di+dj
                l=di-dj
                if (not self.inBoard(mi, mj)) or self.board[mi][mj]!=0:
                    continue
                if self.inBoard(mi+k, mj+k):
                    self.addmove(moves, p0, (mi+k, mj+k), CheckList)
                if self.inBoard(mi+l, mj-l):
                    self.addmove(moves, p0, (mi+l, mj-l), CheckList)

        for p0 in self.place[7+t]:
            i, j = p0
            jumped = False
            for ni in range(i+1, 9):
                y = self.board[ni][j]
                if y == 7 or y == 8:
                    break
                if jumped:
                    self.addmove(moves, p0, (ni, j), CheckList)
                    if y!=0:
                        break
                elif y!=0:
                    jumped=True
            jumped = False
            for ni in range(i-1, -1, -1):
                y = self.board[ni][j]
                if y == 7 or y == 8:
                    break
                if jumped:
                    self.addmove(moves, p0, (ni, j), CheckList)
                    if y!=0:
                        break
                elif y!=0:
                    jumped=True
            jumped = False
            for nj in range(j+1, 10):
                y = self.board[i][nj]
                if y == 7 or y == 8:
                    break
                if jumped:
                    self.addmove(moves, p0, (i, nj), CheckList)
                    if y!=0:
                        break
                elif y!=0:
                    jumped=True
            jumped = False
            for nj in range(j-1, -1, -1):
                y = self.board[i][nj]
                if y == 7 or y == 8:
                    break
                if jumped:
                    self.addmove(moves, p0, (i, nj), CheckList)
                    if y!=0:
                        break
                elif y!=0:
                    jumped=True
            if i==3 or i==5:
                if j==0 or j==2:
                    y = self.board[4][1]
                    z = self.board[8-i][2-j]
                    if y!=0 and y!=7 and y!=8 and z!=7 and z!=8:
                        self.addmove(moves, p0, (8-i, 2-j), CheckList)
                elif j==7 or j==9:
                    y = self.board[4][8]
                    z = self.board[8-i][16-j]
                    if y!=0 and y!=7 and y!=8 and z!=7 and z!=8:
                        self.addmove(moves, p0, (8-i, 16-j), CheckList)
        
        for p0 in self.place[9+t]:
            i, j = p0
            for ni in range(i+1, 9):
                self.addmove(moves, p0, (ni, j), CheckList)
                if self.board[ni][j]!=0:
                    break
            for ni in range(i-1, -1, -1):
                self.addmove(moves, p0, (ni, j), CheckList)
                if self.board[ni][j]!=0:
                    break
            for nj in range(j+1, 10):
                self.addmove(moves, p0, (i, nj), CheckList)
                if self.board[i][nj]!=0:
                    break
            for nj in range(j-1, -1, -1):
                self.addmove(moves, p0, (i, nj), CheckList)
                if self.board[i][nj]!=0:
                    break
            if i==3 or i==5:
                if j==0 or j==2:
                    self.addmove(moves, p0, (4, 1), CheckList)
                    if self.board[4][1]==0:
                        self.addmove(moves, p0, (8-i, 2-j), CheckList)
                elif j==7 or j==9:
                    self.addmove(moves, p0, (4, 8), CheckList)
                    if self.board[4][8]==0:
                        self.addmove(moves, p0, (8-i, 16-j), CheckList)
            elif i==4 and (j==1 or j==8):
                for ni in (3, 5):
                    for nj in (j-1, j+1):
                        self.addmove(moves, p0, (ni, nj), CheckList)

        if self.num>=8 and (not Check) and self.lastmoves[4][0][0]!=-1:
            a = self.lastmoves[4][0]
            b = self.lastmoves[4][1]
            if self.lastmoves[0][1] == b and self.lastmoves[2][0] == b and self.lastmoves[2][1] == a and self.lastmoves[6][0] == b and self.lastmoves[6][1] == a and (a, b) in moves:
                moves.remove((a, b))
        sd = self.score_diff()
        if (not Check) and ((self.turn == 0 and sd < 0) or (self.turn == 1 and sd > 0)):
            moves.append(((-1, -1), (-1, -1)))
        self.moves=moves
        return moves
    
    def isCheck(self, player):
        if player == 0:
            kx, ky = self.place[13][0]

            if self.board[kx-1][ky] == 2 or self.board[kx+1][ky] == 2 or self.board[kx][ky+1] == 2:
                return True            
            if ky == 0 and self.board[4][1] == 2:
                return True
            if kx == 4 and ky == 1 and (self.board[3][2] == 2 or self.board[5][2] == 2):
                return True
            if self.board[4][1] == 10:
                return True
            if kx == 4 and ky == 1:
                for i in (3, 5):
                    for j in (0, 2):
                        if self.board[i][j] == 10:
                            return True
            elif kx != 4 and ky != 1:
                y = self.board[8-kx][2-ky]
                if  y == 10 and self.board[4][1] == 0:
                    return True
                elif y == 8 and self.board[4][1] !=0 and self.board[4][1] != 7 and self.board[4][1] != 8:
                    return True
            

        else:
            kx, ky = self.place[14][0]

            if self.board[kx-1][ky] == 1 or self.board[kx+1][ky] == 1 or self.board[kx][ky-1] == 1:
                return True
            if ky == 9 and self.board[4][8] == 1:
                return True
            if kx == 4 and ky == 8 and (self.board[3][7] == 1 or self.board[5][7] == 1):
                return True
            
            if self.board[4][8] == 9:
                return True
            if kx == 4 and ky == 8:
                for i in (3, 5):
                    for j in (7, 9):
                        if self.board[i][j] == 9:
                            return True
            elif kx != 4 and ky != 8:
                y = self.board[8-kx][16-ky]
                if  y == 9 and self.board[4][8] == 0:
                    return True
                elif y == 7 and self.board[4][8] !=0 and self.board[4][8] != 7 and self.board[4][8] != 8:
                    return True

        for i, j in self.place[4 - player]:
            if (abs(i-kx)==2 and abs(j-ky)==3) or (abs(i-kx)==3 and abs(j-ky)==2):
                ddi = 1 if i-kx>0 else -1
                ddj = 1 if j-ky>0 else -1
                if self.board[kx+ddi*2][ky+ddj*2]==0 and self.board[kx+ddi][ky+ddj]==0:
                    return True
        
        for i, j in self.place[6 - player]:
            if (abs(i-kx)==1 and abs(j-ky)==2) or (abs(i-kx)==2 and abs(j-ky)==1):
                ddi = 1 if i-kx>0 else -1
                ddj = 1 if j-ky>0 else -1
                if self.board[kx+ddi][ky+ddj]==0:
                    return True
                
        for i, j in self.place[8 - player]:
            if kx == i:
                jumped = False
                for nj in range(min(ky, j)+1, max(ky, j)):
                    y = self.board[i][nj]
                    if y == 7 or y == 8:
                        jumped = False
                        break
                    if y!=0:
                        if jumped:
                            jumped = False
                            break
                        else:
                            jumped = True
                if jumped:
                    return True
            elif ky == j:
                jumped = False
                for ni in range(min(kx, i)+1, max(kx, i)):
                    y = self.board[ni][j]
                    if y == 7 or y == 8:
                        jumped = False
                        break
                    if y!=0:
                        if jumped:
                            jumped = False
                            break
                        else:
                            jumped = True
                if jumped:
                    return True

        for i, j in self.place[10-player]:
            if kx == i:
                block = False
                for nj in range(min(ky, j)+1, max(ky, j)):
                    if self.board[i][nj]!=0:
                        block = True
                        break
                if not block:
                    return True
            elif ky == j:
                block = False
                for ni in range(min(kx, i)+1, max(kx, i)):
                    if self.board[ni][j]!=0:
                        block = True
                        break
                if not block:
                    return True
        return False

    def score(self, player):
        ret = 0 if player == 0 else 1.5
        for p in range(player+1, 15, 2):
            ret+=score[p]*len(self.place[p])
        return abs(ret)

    def score_diff(self):
        s = 1.5
        for p in range(1, 15):
            s+=score[p]*len(self.place[p])
        return s

    def move(self, p0, p1):
        ret = State(copy_place = self.place)
        ret.num = 1 + self.num
        ret.turn = 1 - self.turn
        ret.lastmoves = self.lastmoves[1:]+[(p0, p1)]
        pi, pj = p0
        i, j = p1
        if i == -1:
            if ret.num >= self.endturn or ret.score(0)<=10 or ret.score(1)<=10 or (self.lastmoves[7] and self.lastmoves[7][0][0] == -1):
                s = ret.score_diff()
                if s>0:
                    ret.winner = 1
                    ret.v_winner = 1
                else:
                    ret.winner = 0
                    ret.v_winner = 0
            return ret
        
        x = ret.board[pi][pj]
        y = ret.board[i][j]
        ret.board[i][j] = ret.board[pi][pj]
        ret.board[pi][pj] = 0
        
        for xi in range(len(ret.place[x])):
            if ret.place[x][xi] == p0:
                ret.place[x][xi] = p1
                break

        if y!=0:
            ret.place[y].remove(p1)

        if y == 13:
            ret.winner = 1
            ret.v_winner = 1
        elif y == 14:
            ret.winner = 0
            ret.v_winner = 0
        else:
            nextMoves = ret.validMoves()
            if len(nextMoves)==0:
                ret.winner = self.turn
                ret.v_winner = self.turn
            elif y == 0 and (not ret.Checked) and (ret.num >= self.endturn or ret.score(0)<=10 or ret.score(1)<=10):
                s = ret.score_diff()
                if s>0:
                    ret.winner = 1
                    ret.v_winner = 1
                else:
                    ret.winner = 0
                    ret.v_winner = 0
        return ret

        
    def display(self, reverse = False):
        print("Turn", self.num, "Player", self.turn)
        for j in (range(0, 10) if reverse else range(9, -1, -1)):
            for i in range(0, 9):
                x = self.board[i][j]
                if x == 0:
                    print(" . ", end='')
                elif x<10:
                    print(" ", x, " ", sep = '', end = '')
                else:
                    print(x, ' ', sep='', end='')
            print(" ", j)
        print()
        for i in range(0, 9):
            print(' ', i, ' ', sep='', end='')
        print()
        print(self.score(0), "   vs   ", self.score(1))
        print()

    def __hash__(self):
        if self.prehash==None:
            hasher = list()
            for l in self.board:
                hasher.append(tuple(l))
            hasher.append(self.turn)
            hasher.append(self.num)
            hasher.append(self.winner)
            hasher.append(tuple(self.lastmoves))
            self.prehash=hash(tuple(hasher))
        return self.prehash

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        if self.turn!=other.turn or self.num!=other.num or self.winner!=other.winner:
            return False
        for i in range(0, 8):
            if self.lastmoves[i] != other.lastmoves[i]:
                return False
        for i in range(1, 15):
            if len(self.place[i])!=len(other.place[i]):
                return False
            for x in self.place[i]:
                yes = False
                for y in other.place[i]:
                    if x==y:
                        yes=True
                        break
                if not yes:
                    return False
        return True

    def totensor(self, device):
        ret = torch.zeros((1, 21, 9, 10), device = device)
        myscore = (self.score(self.turn)-40)/30
        yourscore = (self.score(1-self.turn)-40)/30
        if self.turn==0:
            index = torch.tensor([self.board], dtype = torch.long, device = device)
            ret[0, 0:14] = torch.zeros((15, 9, 10), device = device).scatter_(0, index, 1.0)[1:15]
        else:
            index = torch.tensor([[[(x+1)^1 for x in reversed(ll)] for ll in self.board]], dtype = torch.long, device = device)
            ret[0, 0:14]=torch.zeros((16, 9, 10), device = device).scatter_(0, index, 1.0)[2:16]
            ret[0, 14]=1.0
        ret[0, 15]=myscore-yourscore
        ret[0, 16]=self.num/200
        ret[0, 17, 4, 1] = 1.0
        ret[0, 18, 4, 8] = 1.0
        ret[0, 19] = myscore
        ret[0, 20] = yourscore
        return ret

def getlogProbsTorch(state, probs):
    device = probs.device
    moves = state.validMoves()
    mnums = []
    if state.turn==0:
        for (a, b) in moves:
            c = (b[0]-a[0], b[1]-a[1])
            if a[0]==-1:
                mnums.append(movecode[c]*90 + state.place[13][0][0]*10 + state.place[13][0][1])
            else:
                mnums.append(movecode[c]*90 + a[0]*10 + a[1])
    else:
        for a, b in moves:
            c = (b[0]-a[0], a[1]-b[1])
            if a[0]==-1:
                mnums.append(movecode[c]*90 + state.place[14][0][0]*10 + state.place[14][0][1])
            else:
                mnums.append(movecode[c]*90 + a[0]*10 + 9 - a[1])
    ret = torch.take(probs, torch.tensor(mnums, dtype = torch.long, device = device))
    ret = F.log_softmax(ret, dim = -1)
    return ret

def softmax(x):
    y = np.exp(x-np.max(x))
    y /= np.sum(y)
    return y

def getProbsNp(state, logits):
    moves = state.validMoves()
    mnums = []
    logits = logits.detach().cpu().numpy()
    if state.turn==0:
        for a, b in moves:
            c = (b[0]-a[0], b[1]-a[1])
            if a[0]==-1:
                mnums.append(movecode[c]*90 + state.place[13][0][0]*10 + state.place[13][0][1])
            else:
                mnums.append(movecode[c]*90 + a[0]*10 + a[1])
    else:
        for a, b in moves:
            c = (b[0]-a[0], a[1]-b[1])
            if a[0]==-1:
                mnums.append(movecode[c]*90 + state.place[14][0][0]*10 + state.place[14][0][1])
            else:
                mnums.append(movecode[c]*90 + a[0]*10 + 9 - a[1])
    ret = np.take(logits, mnums)
    ret = softmax(ret)
    return ret

class TreeEdge:
    def __init__(self, move, n, par):
        self.num = n
        self.parnode = par
        self.Node=None
        self.move=move

class TreeNode:    
    def __init__(self, state, V, edges, logits = None, paredge = None):
        self.state = state
        self.V=V
        self.V0=V
        self.N = 1
        self.searchnum = 0
        self.root=False
        self.winedge = None
        self.paredge = paredge
        if logits == None:
            self.P = None
            self.Q = None
            self.edges=edges
            self.edgenum = len(edges) if edges != None else 0
            self.NN = None
            self.logits = logits
            self.losenum = 1
        else:
            moves = state.validMoves()
            self.edgenum = len(moves)
            self.P = getProbsNp(state, logits)
            self.Q = np.zeros(self.edgenum)
            self.NN = np.ones(self.edgenum)
            edges = [TreeEdge(m, i, self) for i, m in enumerate(moves)]
            self.edges=edges
            self.losenum = self.edgenum

    def SelectMove(self):
        C = 5*np.sqrt(self.N)
        i = np.argmax(C*self.P/self.NN+self.Q)
        return i, self.edges[i]

    def argPrior(self, k):
        C = -5*np.sqrt(self.N)
        a = C*self.P/self.NN-self.Q
        if k == self.edgenum:
            ind = np.arange(0, self.edgenum)
        else:
            ind = np.argpartition(a, k)[:k]
        return ind[np.argsort(a[ind])]

    # def MakeEdges(self):
    #     moves = self.state.validMoves()
    #     self.edgenum = len(moves)
    #     self.P = getProbsNp(self.state, self.logits)
    #     self.Q = np.zeros(self.edgenum)
    #     self.NN = np.ones(self.edgenum)
    #     edges = [TreeEdge(m, i, self) for i, m in enumerate(moves)]
    #     self.edges=edges
    #     self.logits=None
    #     self.losenum = self.edgenum

class MCTree:
    def __init__(self):
        self.cur = None

    def Add(self, state, V, edges, logits = None, paredge = None):
        newNode = TreeNode(state, V, edges, logits, paredge)
        if paredge!=None:
            paredge.Node = newNode
        return newNode

    def MoveOn(self, buffer = None, greedy = False):
        node = self.cur
        if node.winedge != None:
            p = np.zeros(len(node.edges))
            p[node.winedge[0]] = 1
        elif greedy:
            mx = np.max(node.NN)
            p = np.where(node.NN==mx, 1.0, 0.0)
        else:
            p = node.NN-1
        p /= np.sum(p)
        if buffer!=None:
            buffer.append((node.state, p, node.V))
        edge = node.edges[choice(range(len(node.edges)), p = p)]
        ret_move = edge.move
        self.cur = edge.Node
        self.cur.root = True
        self.cur.paredge = None
        return ret_move, p
        

def Leaf_Multi(model, leaf, begin = False):
    leaf_nonterminal = []
    for s, tree, edge in leaf:
        if s.winner==-1:
            leaf_nonterminal.append((s, tree, edge))
        else:
            node = tree.Add(s, 1.0 if s.turn==s.winner else -1.0, list(), None, edge)
            if begin:
                tree.cur = node
    if len(leaf_nonterminal)==0:
        return
    leaf_list = []
    # for i in range(0, len(leaf_nonterminal), 1024):
    #     leaf_list.append(leaf_nonterminal[i:min(i+1024, len(leaf_nonterminal))])
    # for leaf in leaf_list:
    #     inputs = torch.cat([s.totensor(model.device) for s, _, _ in leaf], dim = 0)
    #     vals, logits = model(inputs)
    #     for i, (s, tree, edge) in enumerate(leaf):
    #         node = tree.Add(s, vals[i, 0].item(), None, logits[i], edge)
    #         #node.MakeEdges()
    #         if begin:
    #             tree.cur = node
    inputs = torch.cat([s.totensor(model.device) for s, _, _ in leaf_nonterminal], dim = 0)
    vals, logits = model(inputs)
    for i, (s, tree, edge) in enumerate(leaf_nonterminal):
        node = tree.Add(s, vals[i, 0].item(), None, logits[i], edge)
        #node.MakeEdges()
        if begin:
            tree.cur = node

class NormConv2d(nn.Module):
    def __init__(self, in_chn_num, out_chn_num, kernel_size, padding = 0, groups = 1):
        super(NormConv2d, self).__init__()
        self.norm = nn.BatchNorm2d(in_chn_num)
        self.conv = nn.Conv2d(in_chn_num, out_chn_num, kernel_size, padding = padding, groups = groups, bias = False)
    
    def forward(self, x):
        return self.conv(F.relu(self.norm(x), inplace = True))

class SEBlock(nn.Module):
    def __init__(self, chn_num, mid_num):
        super(SEBlock, self).__init__()
        self.norm = nn.BatchNorm2d(chn_num)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = nn.Linear(chn_num, mid_num)
        self.layer2 = nn.Linear(mid_num, chn_num)
    
    def forward(self, x):
        y = F.relu(self.norm(x), inplace = True)
        y = self.avgPool(y).squeeze(-1).squeeze(-1)
        y = F.relu(self.layer1(y), inplace = True)
        y = torch.sigmoid(self.layer2(y))
        y = y.unsqueeze(-1).unsqueeze(-1)
        return x*y

class Inception_Block(nn.Module):
    def __init__(self, in_chn = 256):
        super(Inception_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chn)
        self.midnum = in_chn//4
        self.conv3x3 = nn.Conv2d(self.midnum*2, self.midnum*2, 3, padding = 1, bias = False)
        self.conv9x1 = nn.Conv2d(self.midnum, self.midnum, (9, 1), padding = (4, 0), bias = False)
        self.conv1x9 = nn.Conv2d(self.midnum, self.midnum, (1, 9), padding = (0, 4), bias = False)
    
    def forward(self, x):
        x = F.relu(self.bn1(x), inplace = True)
        out3x3 = self.conv3x3(x[:, 0:self.midnum*2])
        out9x1 = self.conv9x1(x[:, self.midnum*2:self.midnum*3])
        out1x9 = self.conv1x9(x[:, self.midnum*3:self.midnum*4])
        out = torch.cat([out3x3, out9x1, out1x9], dim = -3)
        return out

class ResNet_Block3(nn.Module):
    def __init__(self, in_chn = 256):
        super(ResNet_Block3, self).__init__()
        self.inception = Inception_Block(in_chn)
        self.conv = NormConv2d(in_chn, in_chn, 1)
        self.seblock = SEBlock(in_chn, in_chn//16)
    
    def forward(self, x):
        y = self.seblock(self.conv(self.inception(x)))
        return x + y

class AlphaJanggi(nn.Module):
    def __init__(self):
        super(AlphaJanggi, self).__init__()
        self.first_conv = nn.Conv2d(21, 256, 3, padding = 1, bias=True)
        self.res_tower = nn.Sequential(*[ResNet_Block3(256) for _ in range(10)])
        self.val_head = nn.Sequential(NormConv2d(256, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(-3, -1), 
                                    nn.Linear(90*4, 256, bias=True), nn.ReLU(), nn.Linear(256, 1, bias=True), nn.Tanh())
        self.prob_head = nn.Sequential(NormConv2d(256, 256, 3, padding = 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, len(movecode), 1))
        self.device = "cpu"
  
    def forward(self, x):
        x2=self.first_conv(x)
        x3=self.res_tower(x2)
        val=self.val_head(x3)
        logits=self.prob_head(x3)
        logits = logits.flatten(-3, -1)
        return val, logits


def SymMove(m):
    a, b = m
    if a[0] == -1:
        return ((-1, -1), (-1, -1))
    return ((8-a[0], a[1]), (8-b[0], b[1]))

def SymBoard(state):
    ret = copy.deepcopy(state)
    for i in range(0, 9):
        for j in range(0, 10):
            ret.board[i][j]=state.board[8-i][j]
    if state.moves != None:
        ret.moves = [SymMove(m) for m in state.moves]
    ret.prehash=None
    return ret

def Print(Node, topk = 5):
    K = Node.edgenum
    if Node.state.v_winner==Node.state.turn:
        print("Winning State")
        print([Node.winedge[1].move])
        return [Node.winedge[1].move]
    elif Node.state.v_winner==1-Node.state.turn:
        print("Losing State")
    sum = np.sum(Node.NN-1)
    moves = [(Node.NN[i]-1, i) for i in range(K)]
    moves.sort()
    movess = [(Node.edges[i].move, n/sum, i) for n, i in reversed(moves)]
    if topk == -1:
        topk = K
    else:
        topk = min(topk, K)
    print("{:.5f}\t{:.5f}".format(Node.V, Node.V0))
    for m, n, i in movess[:topk]:
        print(*m, "{:.5f}\t{:.5f}\t{:.5f}".format(n, Node.P[i], Node.Q[i]))

def ExpandTree_Batches(model, trees, think_time = 4096, verbose = False):
    stack = list()
    leaf = list()
    edges = list()
    for i, (tree, _) in enumerate(trees):
        node = tree.cur
        node.root = True
        #node.P = (node.P*3+np.where(node.P!=0.0, np.random.dirichlet([0.03] * node.edgenum), 0))/4
        if i==0 and verbose:
            node.state.display()
    while True:
        playing_num = 0
        for i, (tree, buffer) in enumerate(trees):
            node = tree.cur
            if node.state.winner!=-1:
                continue
            if node.state.v_winner!=-1:
                while node.state.winner==-1:
                    tree.MoveOn(buffer = buffer)
                    node = tree.cur
                    if i==0 and verbose:
                        node.state.display()
                continue
            playing_num+=1
            if node.N>=think_time:
                if i==0 and verbose:
                    Print(node)
                tree.MoveOn(buffer = buffer)
                node = tree.cur
                if i==0 and verbose:
                    node.state.display()
                if node.state.v_winner!=-1:
                    while node.state.winner==-1:
                        tree.MoveOn(buffer = buffer)
                        node = tree.cur
                        if i==0 and verbose:
                            node.state.display()
                    continue
                #node.P = (node.P*3+np.where(node.P!=0.0, np.random.dirichlet([0.03] * node.edgenum), 0))/4
                node.searchnum = 0
                for e in node.edges:
                    if e.Node==None:
                        s = node.state.move(*e.move)
                        leaf.append((s, tree, e))
                        edges.append(e)
                        node.searchnum+=1
            elif think_time >=2048:
                if node.N<think_time//8:
                    stack.append((node, 64))
                elif node.N<think_time//8*5:
                    stack.append((node, 32))
                elif node.N<think_time//4*3:
                    stack.append((node, 8))
                elif node.N<think_time//8*7:
                    stack.append((node, 4))
                else:
                    stack.append((node, 1))
            else:
                if node.N<think_time//8:
                    stack.append((node, 32))
                elif node.N<think_time//2:
                    stack.append((node, 16))
                elif node.N<think_time//4*3:
                    stack.append((node, 4))
                else:
                    stack.append((node, 1))


        if playing_num==0:
            break

        while len(stack)>0:
            node, num = stack[-1]
            stack.pop()
            searchnum = min(num, node.losenum)
            if node.root:
                if node.N<think_time//8:
                    if searchnum > 8:
                        searchnum = 8
                elif node.N<think_time//8*3:
                #elif node.N<think_time//4*3:
                    if searchnum > 4:
                        searchnum = 4
                elif node.N<think_time//2:
                #elif node.N<think_time//8*7:
                    if searchnum > 2:
                        searchnum = 2
                else:
                    searchnum = 1
            else:
                if node.N<think_time//64:
                #if node.N<think_time//32:
                    if searchnum > 4:
                        searchnum = 4
                elif node.N<think_time//8:
                    if searchnum > 2:
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

@torch.no_grad()
def Alpha_SelfPlay_Multi(model, n = 16, think_time = 4096, verbose = False):
    samples = []
    trees = [(MCTree(), []) for _ in range(n)]
    start = [(State((choice(4), choice(4))), tree, None) for tree, _ in trees]
    Leaf_Multi(model, start, begin = True)
    ExpandTree_Batches(model, trees, think_time, verbose)
    
    for tree, buf in trees:
        buf.reverse()
        avgV = 1.0 if tree.cur.state.winner == buf[0][0].turn else -1.0
        lamb = 0.99
        for state, p, V in buf:
            samples.append((state, p, avgV))
            avgV += (V-avgV)*(1-lamb)
            avgV = -avgV
    torch.cuda.empty_cache()
    return samples

def Train(model, optimizer, samples, sample_num = 2048, batch_size = 64):
    device = model.device
    MSEloss = nn.MSELoss(reduction = 'mean')
    indices = choice(len(samples), sample_num, replace = False)
    steps = 0
    for i in range(0, sample_num, batch_size):
        goal_val = []
        states = []
        for j in indices[i:i+batch_size]:
            states.append(samples[j][0])
            states.append(SymBoard(samples[j][0]))
            goal_val.append([samples[j][2]])
            goal_val.append([samples[j][2]])
        input = torch.cat([s.totensor(device) for s in states], dim = 0)
        output_val, output_logits = model(input)
        goal_val = torch.tensor(goal_val, dtype = torch.float32, device = device)
        loss_val = MSEloss(output_val, goal_val)
        loss_list = []
        for k, j in enumerate(indices[i:i+batch_size]):
            log_myprob = getlogProbsTorch(states[2*k], output_logits[2*k]) + getlogProbsTorch(states[2*k+1], output_logits[2*k+1])
            y_prob = torch.tensor(samples[j][1], dtype = torch.float32, device = device)
            loss_prob = torch.dot(log_myprob, y_prob)
            loss_list.append(-loss_prob)

        loss_p = torch.mean(torch.stack(loss_list))/2
        loss = loss_val+loss_p
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1
    return steps

def SaveSamples(samples, path):
    packs = []
    for s, p, V in samples:
        pack = [[s.place, s.validMoves(), s.num, s.turn], p, V]
        packs.append(pack)
    with open(path, 'wb') as fp:
        pickle.dump(packs, fp)

def LoadSamples(path):
    samples = []
    with open(path, 'rb') as fp:
        packs = pickle.load(fp)
    for ss, p, V in packs:
        s = State(copy_place = ss[0])
        s.moves = ss[1]
        s.num = ss[2]
        s.turn = ss[3]
        samples.append((s, p, V))
    return samples